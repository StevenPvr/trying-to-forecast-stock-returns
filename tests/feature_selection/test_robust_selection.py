from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.cache import FeatureSelectionRuntimeCache
from core.src.meta_model.feature_selection.io import build_feature_selection_metadata_from_frame
from core.src.meta_model.feature_selection.correlation import (
    run_incremental_distance_correlation_pruning,
    run_incremental_linear_correlation_pruning,
    run_target_distance_correlation_filter,
)
from core.src.meta_model.feature_selection import sfi as sfi_module
from core.src.meta_model.feature_selection.cv import SelectionFold
from core.src.meta_model.feature_selection.objective import FoldEconomicScore, SubsetEconomicScore
from core.src.meta_model.feature_selection.scoring import BacktestFeatureSubsetScorer
from core.src.meta_model.feature_selection.sfi import build_sfi_score_frame
from core.src.meta_model.model_contract import DATE_COLUMN, MODEL_TARGET_COLUMN, SPLIT_COLUMN, TICKER_COLUMN
from core.src.meta_model.model_contract import REALIZED_RETURN_COLUMN


class _DummyCache:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.reset_index(drop=True)

    def feature_coverage_fraction(self, feature_name: str) -> float:
        values = cast(pd.Series, self._frame[feature_name]).to_numpy(dtype=np.float64, copy=False)
        return float(np.mean(np.isfinite(values)))

    def build_sampled_feature_frame(
        self,
        feature_names: list[str],
        *,
        sample_size: int,
    ) -> pd.DataFrame:
        del sample_size
        return pd.DataFrame(self._frame.loc[:, feature_names].copy())

    def build_feature_frame(
        self,
        feature_names: list[str],
        *,
        row_indices: np.ndarray | None = None,
    ) -> pd.DataFrame:
        columns = [
            DATE_COLUMN,
            TICKER_COLUMN,
            SPLIT_COLUMN,
            MODEL_TARGET_COLUMN,
            REALIZED_RETURN_COLUMN,
            *feature_names,
        ]
        frame = pd.DataFrame(self._frame.loc[:, columns].copy())
        if row_indices is None:
            return frame
        return pd.DataFrame(frame.take(np.asarray(row_indices, dtype=np.int64)).reset_index(drop=True))

    def get_feature_array(self, feature_name: str) -> np.ndarray:
        return self.get_feature_array_slice(feature_name)

    def get_feature_array_slice(
        self,
        feature_name: str,
        *,
        row_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        feature_series = cast(pd.Series, self._frame[feature_name])
        values = feature_series.to_numpy(dtype=np.float32, copy=True)
        if row_indices is None:
            return values
        return values[np.asarray(row_indices, dtype=np.int64)]

    def build_temporal_sample_row_indices(
        self,
        sample_size: int,
        *,
        minimum_date_count: int = 1,
    ) -> np.ndarray:
        if sample_size <= 0 or len(self._frame) <= sample_size:
            return np.arange(len(self._frame), dtype=np.int64)
        date_values = pd.to_datetime(cast(pd.Series, self._frame[DATE_COLUMN])).to_numpy(copy=False)
        unique_dates = pd.Index(pd.to_datetime(date_values).drop_duplicates().sort_values())
        date_count = min(len(unique_dates), max(minimum_date_count, sample_size // 2))
        sampled_dates = pd.Index(
            unique_dates.take(
                np.unique(np.linspace(0, len(unique_dates) - 1, num=date_count, dtype=np.int64)),
            ),
        )
        sampled_index = np.flatnonzero(
            cast(pd.Series, self._frame[DATE_COLUMN]).isin(sampled_dates).to_numpy(),
        )
        if sampled_index.size <= sample_size:
            return sampled_index
        sampled_positions = np.unique(
            np.linspace(0, sampled_index.size - 1, num=sample_size, dtype=np.int64),
        )
        return sampled_index[sampled_positions]

    @property
    def train_row_count(self) -> int:
        return len(self._frame)


def _build_subset_score(feature_names: list[str]) -> SubsetEconomicScore:
    best_feature = max(feature_names)
    objective = {
        "feature_best": 0.05,
        "feature_dup": 0.04,
        "feature_curve": 0.03,
        "feature_noise": -0.01,
    }.get(best_feature, 0.0)
    fold_scores = (
        FoldEconomicScore(
            index=1,
            weight=1.0,
            net_pnl_after_costs=objective,
            alpha_over_benchmark_net=max(objective, 0.0),
            turnover_annualized=0.40,
            max_drawdown=-0.05,
            daily_rank_ic_mean=objective,
            daily_rank_ic_ir=objective,
            daily_top_bottom_spread_mean=max(objective, 0.0),
        ),
    )
    return SubsetEconomicScore(
        feature_names=tuple(sorted(feature_names)),
        objective_score=objective,
        weighted_net_pnl_after_costs=objective,
        weighted_alpha_over_benchmark_net=max(objective, 0.0),
        weighted_turnover_annualized=0.40,
        weighted_max_drawdown=-0.05,
        positive_fold_share=1.0 if objective > 0.0 else 0.0,
        median_fold_net_pnl=objective,
        lower_quartile_fold_net_pnl=objective,
        is_valid=objective > 0.0,
        fold_scores=fold_scores,
        pnl_positive_fold_share=1.0 if objective > 0.0 else 0.0,
        weighted_daily_rank_ic_mean=objective,
        weighted_daily_rank_ic_ir=objective,
        weighted_daily_top_bottom_spread_mean=max(objective, 0.0),
    )


def _make_train_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=24, freq="B")
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        curve_value = float(index ** 2)
        for ticker, base in (("AAA", 0.0), ("BBB", 1.0)):
            target = float(index % 3) + base
            rows.append(
                {
                    DATE_COLUMN: date,
                    TICKER_COLUMN: ticker,
                    SPLIT_COLUMN: "train",
                    MODEL_TARGET_COLUMN: target,
                    "feature_best": float(index) + base,
                    "feature_dup": (float(index) + base) * 1.0,
                    "feature_curve": curve_value + base,
                    "feature_noise": float((-1) ** index),
                },
            )
    return pd.DataFrame(rows)


def test_build_sfi_score_frame_applies_coverage_threshold() -> None:
    frame = _make_train_frame()
    frame.loc[frame.index[::3], "feature_noise"] = np.nan
    cache = _DummyCache(frame)

    score_frame = build_sfi_score_frame(
        cast(Any, cache),
        ["feature_best", "feature_noise"],
        _build_subset_score,
        FeatureSelectionConfig(sfi_min_coverage_fraction=0.90),
    )

    best_row = cast(pd.DataFrame, score_frame.loc[score_frame["feature_name"] == "feature_best"]).iloc[0]
    noise_row = cast(pd.DataFrame, score_frame.loc[score_frame["feature_name"] == "feature_noise"]).iloc[0]
    assert bool(best_row["passes_sfi"])
    assert not bool(noise_row["passes_coverage"])
    assert str(noise_row["sfi_drop_reason"]) == "low_coverage"


def test_build_sfi_score_frame_requires_selection_gates() -> None:
    frame = _make_train_frame()
    cache = _DummyCache(frame)

    def gated_out_scorer(feature_names: list[str]) -> SubsetEconomicScore:
        del feature_names
        fold_score = FoldEconomicScore(
            index=1,
            weight=1.0,
            net_pnl_after_costs=0.01,
            alpha_over_benchmark_net=0.01,
            turnover_annualized=0.40,
            max_drawdown=-0.05,
            daily_rank_ic_mean=0.01,
            daily_rank_ic_ir=-0.25,
            daily_top_bottom_spread_mean=-0.02,
        )
        return SubsetEconomicScore(
            feature_names=("feature_best",),
            objective_score=0.01,
            weighted_net_pnl_after_costs=0.01,
            weighted_alpha_over_benchmark_net=0.01,
            weighted_turnover_annualized=0.40,
            weighted_max_drawdown=-0.05,
            positive_fold_share=0.0,
            median_fold_net_pnl=0.01,
            lower_quartile_fold_net_pnl=0.01,
            is_valid=False,
            fold_scores=(fold_score,),
            weighted_daily_rank_ic_mean=0.01,
            weighted_daily_rank_ic_ir=-0.25,
            weighted_daily_top_bottom_spread_mean=-0.02,
        )

    score_frame = build_sfi_score_frame(
        cast(Any, cache),
        ["feature_best"],
        gated_out_scorer,
        FeatureSelectionConfig(sfi_min_coverage_fraction=0.90),
    )

    row = cast(pd.DataFrame, score_frame.loc[score_frame["feature_name"] == "feature_best"]).iloc[0]
    assert not bool(row["passes_sfi"])
    assert str(row["sfi_drop_reason"]) == "failed_selection_gates"


def test_build_sfi_score_frame_ignores_subset_backtest_guardrails() -> None:
    frame = _make_train_frame()
    cache = _DummyCache(frame)

    def sfi_viable_scorer(feature_names: list[str]) -> SubsetEconomicScore:
        del feature_names
        fold_score = FoldEconomicScore(
            index=1,
            weight=1.0,
            net_pnl_after_costs=-0.02,
            alpha_over_benchmark_net=-0.01,
            turnover_annualized=0.40,
            max_drawdown=-0.05,
            daily_rank_ic_mean=0.01,
            daily_rank_ic_ir=0.15,
            daily_top_bottom_spread_mean=-0.02,
        )
        return SubsetEconomicScore(
            feature_names=("feature_best",),
            objective_score=0.01,
            weighted_net_pnl_after_costs=-0.02,
            weighted_alpha_over_benchmark_net=-0.01,
            weighted_turnover_annualized=0.40,
            weighted_max_drawdown=-0.05,
            positive_fold_share=1.0,
            median_fold_net_pnl=-0.02,
            lower_quartile_fold_net_pnl=-0.02,
            is_valid=True,
            fold_scores=(fold_score,),
            weighted_daily_rank_ic_mean=0.01,
            weighted_daily_rank_ic_ir=0.15,
            weighted_daily_top_bottom_spread_mean=-0.02,
        )

    score_frame = build_sfi_score_frame(
        cast(Any, cache),
        ["feature_best"],
        sfi_viable_scorer,
        FeatureSelectionConfig(sfi_min_coverage_fraction=0.90),
    )

    row = cast(pd.DataFrame, score_frame.loc[score_frame["feature_name"] == "feature_best"]).iloc[0]
    assert bool(row["passes_sfi"])
    assert str(row["sfi_drop_reason"]) == "retained"


def test_build_sfi_score_frame_falls_back_to_threads_when_process_pool_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _make_train_frame()
    cache = _DummyCache(frame)
    fallback_rows = [
        {
            "feature_name": "feature_best",
            "feature_family": "other",
            "feature_stem": "feature_best",
            "coverage_fraction": 1.0,
            "objective_score": 0.05,
            "daily_rank_ic_mean": 0.05,
            "daily_rank_ic_ir": 0.05,
            "daily_top_bottom_spread_mean": 0.05,
            "positive_fold_share": 1.0,
            "passes_coverage": True,
            "passes_sfi": True,
            "sfi_drop_reason": "retained",
        },
    ]

    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.sfi._can_use_process_pool",
        lambda cache, scorer: True,
    )
    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.sfi._score_features_parallel_process",
        lambda cache, scorer, feature_names, config, worker_count: (_ for _ in ()).throw(
            PermissionError("process pool blocked")
        ),
    )
    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.sfi._score_features_parallel",
        lambda cache, scorer, feature_names, config, worker_count: fallback_rows,
    )

    score_frame = build_sfi_score_frame(
        cast(Any, cache),
        ["feature_best"],
        _build_subset_score,
        FeatureSelectionConfig(),
    )

    assert score_frame["feature_name"].tolist() == ["feature_best"]
    assert score_frame["objective_score"].tolist() == [0.05]


def test_sfi_worker_count_uses_config_parallel_workers() -> None:
    config = FeatureSelectionConfig(parallel_workers=12)

    assert sfi_module._resolve_sfi_worker_count(config, 40) == 12
    assert sfi_module._resolve_sfi_worker_count(config, 3) == 3


def test_sfi_worker_config_disables_nested_cpu_oversubscription() -> None:
    config = FeatureSelectionConfig(parallel_workers=12)

    worker_config = sfi_module._build_sfi_worker_config(config, worker_count=12)

    assert worker_config.parallel_workers == 1
    assert worker_config.state_evaluation_workers == 1


def test_sfi_thread_backend_rebalances_inner_scorer_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2020-01-01", periods=16, freq="B"),
            TICKER_COLUMN: ["AAA"] * 16,
            SPLIT_COLUMN: ["train"] * 16,
            MODEL_TARGET_COLUMN: np.linspace(-1.0, 1.0, 16),
            REALIZED_RETURN_COLUMN: np.linspace(0.0, 0.01, 16),
            "feature_signal": np.linspace(0.0, 1.0, 16),
            "feature_other": np.linspace(1.0, 0.0, 16),
        },
    )
    folds = [
        SelectionFold(
            index=1,
            weight=1.0,
            train_indices=np.asarray(list(range(0, 8)), dtype=np.int64),
            validation_indices=np.asarray(list(range(8, 10)), dtype=np.int64),
            train_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-10")),
            validation_start_date=cast(pd.Timestamp, pd.Timestamp("2020-01-13")),
            validation_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-14")),
        ),
        SelectionFold(
            index=2,
            weight=1.0,
            train_indices=np.asarray(list(range(0, 9)), dtype=np.int64),
            validation_indices=np.asarray(list(range(9, 11)), dtype=np.int64),
            train_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-10")),
            validation_start_date=cast(pd.Timestamp, pd.Timestamp("2020-01-13")),
            validation_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-14")),
        ),
        SelectionFold(
            index=3,
            weight=1.0,
            train_indices=np.asarray(list(range(0, 10)), dtype=np.int64),
            validation_indices=np.asarray(list(range(10, 12)), dtype=np.int64),
            train_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-10")),
            validation_start_date=cast(pd.Timestamp, pd.Timestamp("2020-01-13")),
            validation_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-14")),
        ),
        SelectionFold(
            index=4,
            weight=1.0,
            train_indices=np.asarray(list(range(0, 11)), dtype=np.int64),
            validation_indices=np.asarray(list(range(11, 13)), dtype=np.int64),
            train_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-10")),
            validation_start_date=cast(pd.Timestamp, pd.Timestamp("2020-01-13")),
            validation_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-14")),
        ),
    ]
    scorer = BacktestFeatureSubsetScorer(
        cast(Any, _DummyCache(frame)),
        folds,
        FeatureSelectionConfig(parallel_workers=12, null_bootstrap_count=0),
    )
    observed_scorer_settings: list[tuple[int, int, int]] = []

    def fake_score_single_feature(
        cache: object,
        scorer: object,
        feature_name: str,
        config: FeatureSelectionConfig,
    ) -> dict[str, object]:
        del cache, feature_name, config
        typed_scorer = cast(BacktestFeatureSubsetScorer, scorer)
        observed_scorer_settings.append(
            (
                typed_scorer._state_worker_budget,
                typed_scorer._fold_worker_count,
                typed_scorer._model_thread_count,
            ),
        )
        return {
            "feature_name": "feature_signal",
            "feature_family": "other",
            "feature_stem": "feature_signal",
            "coverage_fraction": 1.0,
            "objective_score": 0.1,
            "daily_rank_ic_mean": 0.1,
            "daily_rank_ic_ir": 0.1,
            "daily_top_bottom_spread_mean": 0.1,
            "positive_fold_share": 1.0,
            "passes_coverage": True,
            "passes_sfi": True,
            "sfi_drop_reason": "retained",
        }

    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.sfi._score_single_feature",
        fake_score_single_feature,
    )

    sfi_module._score_features_parallel(
        cast(Any, _DummyCache(frame)),
        scorer,
        ["feature_signal", "feature_other"],
        FeatureSelectionConfig(parallel_workers=12, null_bootstrap_count=0),
        worker_count=12,
    )

    assert observed_scorer_settings
    assert set(observed_scorer_settings) == {(1, 1, 1)}


def test_linear_pruning_keeps_highest_sfi_representative() -> None:
    frame = _make_train_frame()
    cache = _DummyCache(frame)
    sfi_frame = pd.DataFrame(
        {
            "feature_name": ["feature_best", "feature_dup", "feature_noise"],
            "feature_family": ["other", "other", "other"],
            "feature_stem": ["feature_best", "feature_dup", "feature_noise"],
            "objective_score": [0.05, 0.04, 0.01],
            "daily_rank_ic_mean": [0.05, 0.04, 0.01],
            "coverage_fraction": [1.0, 1.0, 1.0],
            "passes_sfi": [True, True, True],
        },
    )

    survivors, audit = run_incremental_linear_correlation_pruning(
        cast(Any, cache),
        sfi_frame,
        FeatureSelectionConfig(group_sample_size=128, linear_correlation_threshold=0.95),
    )

    assert "feature_best" in survivors
    assert "feature_dup" not in survivors
    dropped = cast(pd.DataFrame, audit.loc[audit["decision"] == "drop"])
    assert "feature_dup" in set(cast(pd.Series, dropped["feature_name"]).astype(str))


def test_distance_pruning_drops_nonlinear_duplicate() -> None:
    frame = _make_train_frame()
    cache = _DummyCache(frame)
    sfi_frame = pd.DataFrame(
        {
            "feature_name": ["feature_best", "feature_curve", "feature_noise"],
            "feature_family": ["other", "other", "other"],
            "feature_stem": ["feature_best", "feature_curve", "feature_noise"],
            "objective_score": [0.05, 0.03, 0.01],
            "daily_rank_ic_mean": [0.05, 0.03, 0.01],
            "coverage_fraction": [1.0, 1.0, 1.0],
            "passes_sfi": [True, True, True],
        },
    )

    survivors, audit = run_incremental_distance_correlation_pruning(
        cast(Any, cache),
        sfi_frame,
        ["feature_best", "feature_curve", "feature_noise"],
        FeatureSelectionConfig(distance_correlation_threshold=0.80, distance_correlation_sample_size=48),
    )

    assert "feature_best" in survivors
    assert "feature_curve" not in survivors
    assert "feature_noise" in survivors
    dropped = cast(pd.DataFrame, audit.loc[audit["decision"] == "drop"])
    assert "feature_curve" in set(cast(pd.Series, dropped["feature_name"]).astype(str))


def test_target_distance_correlation_filter_keeps_only_features_linked_to_train_target() -> None:
    frame = _make_train_frame()
    frame[REALIZED_RETURN_COLUMN] = frame[MODEL_TARGET_COLUMN]
    frame["feature_signal"] = frame[MODEL_TARGET_COLUMN]
    frame["feature_dead"] = 1.0
    cache = _DummyCache(frame)
    sfi_frame = pd.DataFrame(
        {
            "feature_name": ["feature_signal", "feature_dead"],
            "feature_family": ["other", "other"],
            "feature_stem": ["feature_signal", "feature_dead"],
            "objective_score": [0.05, 0.04],
            "daily_rank_ic_mean": [0.05, 0.04],
            "coverage_fraction": [1.0, 1.0],
            "passes_sfi": [True, True],
        },
    )

    survivors, audit = run_target_distance_correlation_filter(
        cast(Any, cache),
        sfi_frame,
        ["feature_signal", "feature_dead"],
        FeatureSelectionConfig(target_distance_correlation_threshold=0.005, parallel_workers=2),
    )

    assert survivors == ["feature_signal"]
    signal_row = cast(pd.DataFrame, audit.loc[audit["feature_name"] == "feature_signal"]).iloc[0]
    dead_row = cast(pd.DataFrame, audit.loc[audit["feature_name"] == "feature_dead"]).iloc[0]
    assert bool(signal_row["passes_target_correlation"])
    assert not bool(dead_row["passes_target_correlation"])
    assert str(dead_row["drop_reason"]) == "low_target_distance_correlation"


def test_target_distance_correlation_filter_avoids_threaded_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _make_train_frame()
    frame[REALIZED_RETURN_COLUMN] = frame[MODEL_TARGET_COLUMN]
    frame["feature_signal"] = frame[MODEL_TARGET_COLUMN]
    cache = _DummyCache(frame)
    sfi_frame = pd.DataFrame(
        {
            "feature_name": ["feature_signal"],
            "feature_family": ["other"],
            "feature_stem": ["feature_signal"],
            "objective_score": [0.05],
            "daily_rank_ic_mean": [0.05],
            "coverage_fraction": [1.0],
            "passes_sfi": [True],
        },
    )

    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.correlation._score_target_distance_correlation_rows",
        lambda cache, feature_names, target_values, threshold, row_indices: [
            {
                "feature_name": "feature_signal",
                "target_distance_correlation": 1.0,
                "passes_target_correlation": True,
                "drop_reason": "retained",
            },
        ],
    )

    survivors, audit = run_target_distance_correlation_filter(
        cast(Any, cache),
        sfi_frame,
        ["feature_signal"],
        FeatureSelectionConfig(target_distance_correlation_threshold=0.005, parallel_workers=12),
    )

    assert survivors == ["feature_signal"]
    assert audit["feature_name"].tolist() == ["feature_signal"]


def test_target_distance_correlation_filter_uses_temporally_distributed_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _make_train_frame()
    frame[REALIZED_RETURN_COLUMN] = frame[MODEL_TARGET_COLUMN]
    frame["feature_signal"] = frame[MODEL_TARGET_COLUMN]
    cache = _DummyCache(frame)
    sfi_frame = pd.DataFrame(
        {
            "feature_name": ["feature_signal"],
            "feature_family": ["other"],
            "feature_stem": ["feature_signal"],
            "objective_score": [0.05],
            "daily_rank_ic_mean": [0.05],
            "coverage_fraction": [1.0],
            "passes_sfi": [True],
        },
    )
    captured_row_indices: list[np.ndarray] = []

    def _capture_rows(
        cache: object,
        feature_names: list[str],
        target_values: np.ndarray,
        *,
        threshold: float,
        row_indices: np.ndarray,
    ) -> list[dict[str, object]]:
        del cache, feature_names, target_values, threshold
        captured_row_indices.append(np.asarray(row_indices, dtype=np.int64))
        return [
            {
                "feature_name": "feature_signal",
                "target_distance_correlation": 1.0,
                "passes_target_correlation": True,
                "drop_reason": "retained",
            },
        ]

    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.correlation._score_target_distance_correlation_rows",
        _capture_rows,
    )

    survivors, audit = run_target_distance_correlation_filter(
        cast(Any, cache),
        sfi_frame,
        ["feature_signal"],
        FeatureSelectionConfig(
            target_distance_correlation_threshold=0.005,
            target_distance_correlation_sample_size=10,
        ),
    )

    assert survivors == ["feature_signal"]
    assert audit["feature_name"].tolist() == ["feature_signal"]
    assert len(captured_row_indices) == 1
    sampled_dates = pd.Index(
        pd.to_datetime(
            cache.build_feature_frame([], row_indices=captured_row_indices[0])[DATE_COLUMN],
        ).drop_duplicates().sort_values(),
    )
    assert captured_row_indices[0].size == 10
    assert len(sampled_dates) >= 5
    assert sampled_dates[0] == pd.Timestamp("2020-01-01")
    assert sampled_dates[-1] == pd.Timestamp("2020-02-03")


def test_sfi_process_pool_is_disabled_for_in_memory_cache() -> None:
    frame = _make_train_frame()
    frame[REALIZED_RETURN_COLUMN] = frame[MODEL_TARGET_COLUMN]
    metadata = build_feature_selection_metadata_from_frame(frame)
    cache = FeatureSelectionRuntimeCache(
        frame,
        metadata,
        random_seed=7,
        max_cache_gib=0.05,
    )

    assert cache._dataset_path is None
    assert not __import__(
        "core.src.meta_model.feature_selection.sfi",
        fromlist=["_can_use_process_pool"],
    )._can_use_process_pool(cache, object())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
