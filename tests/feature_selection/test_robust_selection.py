from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.correlation import (
    run_incremental_distance_correlation_pruning,
    run_incremental_linear_correlation_pruning,
    run_target_distance_correlation_filter,
)
from core.src.meta_model.feature_selection.cv import SelectionFold
from core.src.meta_model.feature_selection.mda import run_mda_selection
from core.src.meta_model.feature_selection.objective import FoldEconomicScore, SubsetEconomicScore
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
    ) -> pd.DataFrame:
        columns = [
            DATE_COLUMN,
            TICKER_COLUMN,
            SPLIT_COLUMN,
            MODEL_TARGET_COLUMN,
            REALIZED_RETURN_COLUMN,
            *feature_names,
        ]
        return pd.DataFrame(self._frame.loc[:, columns].copy())

    def get_feature_array(self, feature_name: str) -> np.ndarray:
        feature_series = cast(pd.Series, self._frame[feature_name])
        return feature_series.to_numpy(dtype=np.float32, copy=True)


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
        lambda cache, feature_names, target_values, threshold: [
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


def test_run_mda_selection_marks_positive_importance_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.to_datetime(
        ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
    )
    frame = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            TICKER_COLUMN: ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            SPLIT_COLUMN: ["train"] * 6,
            MODEL_TARGET_COLUMN: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            REALIZED_RETURN_COLUMN: [0.0, 0.01, 0.0, 0.01, 0.0, 0.01],
            "feature_best": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "feature_noise": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
    )
    cache = _DummyCache(frame)
    sfi_frame = pd.DataFrame(
        {
            "feature_name": ["feature_best", "feature_noise"],
            "feature_family": ["other", "other"],
            "feature_stem": ["feature_best", "feature_noise"],
            "objective_score": [0.05, 0.01],
            "daily_rank_ic_mean": [0.05, 0.01],
            "coverage_fraction": [1.0, 1.0],
            "passes_sfi": [True, True],
        },
    )
    folds = [
        SelectionFold(
            index=1,
            weight=1.0,
            train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
            validation_indices=np.asarray([4, 5], dtype=np.int64),
            train_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-02")),
            validation_start_date=cast(pd.Timestamp, pd.Timestamp("2020-01-03")),
            validation_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-03")),
        ),
    ]

    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.mda.fit_model",
        lambda model_spec, train_frame, feature_names: type(
            "Artifact",
            (),
            {
                "feature_names": list(feature_names),
                "training_metadata": {"target_column": MODEL_TARGET_COLUMN},
                "fitted_object": None,
            },
        )(),
    )
    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.mda.predict_model",
        lambda artifact, validation_frame, feature_names: validation_frame["feature_best"].to_numpy(dtype=np.float64, copy=False),
    )
    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.mda.score_predicted_validation",
        lambda predicted_validation, backtest_config, cost_config, target_column=MODEL_TARGET_COLUMN: {
            "daily_rank_ic_mean": float(
                np.mean(
                    predicted_validation["prediction"].to_numpy(dtype=np.float64, copy=False)
                    == predicted_validation[target_column].to_numpy(dtype=np.float64, copy=False),
                ),
            ),
        },
    )
    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.mda._permute_feature_within_dates",
        lambda validation_frame, feature_name, seed: pd.DataFrame(
            validation_frame.assign(
                **{
                    feature_name: (
                        validation_frame[feature_name].to_numpy(dtype=np.float64, copy=False)[::-1]
                        if feature_name == "feature_best"
                        else validation_frame[feature_name].to_numpy(dtype=np.float64, copy=False)
                    ),
                },
            ),
        ),
    )

    result = run_mda_selection(
        cast(Any, cache),
        folds,
        sfi_frame,
        ["feature_best", "feature_noise"],
        FeatureSelectionConfig(mda_permutation_repeats=2, proxy_training_rounds=8),
    )

    selected_names = set(result.selected_feature_names)
    assert "feature_best" in selected_names


def test_run_mda_selection_scores_all_survivors_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            DATE_COLUMN: pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            TICKER_COLUMN: ["AAA", "BBB", "AAA", "BBB"],
            SPLIT_COLUMN: ["train"] * 4,
            MODEL_TARGET_COLUMN: [0.0, 1.0, 0.0, 1.0],
            REALIZED_RETURN_COLUMN: [0.0, 0.01, 0.0, 0.01],
            "feature_a": [0.0, 1.0, 0.0, 1.0],
            "feature_b": [1.0, 0.0, 1.0, 0.0],
            "feature_c": [0.5, 0.5, 0.5, 0.5],
        },
    )
    cache = _DummyCache(frame)
    sfi_frame = pd.DataFrame(
        {
            "feature_name": ["feature_a", "feature_b", "feature_c"],
            "feature_family": ["other", "other", "other"],
            "feature_stem": ["feature_a", "feature_b", "feature_c"],
            "objective_score": [0.3, 0.2, 0.1],
            "daily_rank_ic_mean": [0.3, 0.2, 0.1],
            "coverage_fraction": [1.0, 1.0, 1.0],
            "passes_sfi": [True, True, True],
        },
    )
    folds = [
        SelectionFold(
            index=1,
            weight=1.0,
            train_indices=np.asarray([0, 1], dtype=np.int64),
            validation_indices=np.asarray([2, 3], dtype=np.int64),
            train_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-01")),
            validation_start_date=cast(pd.Timestamp, pd.Timestamp("2020-01-02")),
            validation_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-02")),
        ),
    ]

    scored_feature_names: list[str] = []

    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.mda._build_fold_context",
        lambda *args, **kwargs: {
            "fold": folds[0],
            "artifact": object(),
            "validation_frame": frame,
            "baseline_score": 0.0,
        },
    )
    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.mda._score_single_feature_mda",
        lambda feature_name, fold_contexts, config: (
            scored_feature_names.append(feature_name)
            or {
                "feature_name": feature_name,
                "mda_mean_delta_objective": 0.001,
                "mda_std_delta_objective": 0.0,
                "mda_fold_positive_share": 1.0,
                "mda_repeat_count": 1,
                "selected": True,
                "mda_drop_reason": "selected",
            }
        ),
    )

    result = run_mda_selection(
        cast(Any, cache),
        folds,
        sfi_frame,
        ["feature_a", "feature_b", "feature_c"],
        FeatureSelectionConfig(mda_max_features=1, proxy_training_rounds=8),
    )

    assert sorted(scored_feature_names) == ["feature_a", "feature_b", "feature_c"]
    assert result.final_scores["feature_name"].tolist() == ["feature_a", "feature_b", "feature_c"]


def test_run_mda_selection_parallelizes_feature_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            DATE_COLUMN: pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            TICKER_COLUMN: ["AAA", "BBB", "AAA", "BBB"],
            SPLIT_COLUMN: ["train"] * 4,
            MODEL_TARGET_COLUMN: [0.0, 1.0, 0.0, 1.0],
            REALIZED_RETURN_COLUMN: [0.0, 0.01, 0.0, 0.01],
            "feature_a": [0.0, 1.0, 0.0, 1.0],
            "feature_b": [1.0, 0.0, 1.0, 0.0],
            "feature_c": [0.5, 0.5, 0.5, 0.5],
            "feature_d": [0.2, 0.8, 0.2, 0.8],
        },
    )
    cache = _DummyCache(frame)
    sfi_frame = pd.DataFrame(
        {
            "feature_name": ["feature_a", "feature_b", "feature_c", "feature_d"],
            "feature_family": ["other", "other", "other", "other"],
            "feature_stem": ["feature_a", "feature_b", "feature_c", "feature_d"],
            "objective_score": [0.4, 0.3, 0.2, 0.1],
            "daily_rank_ic_mean": [0.4, 0.3, 0.2, 0.1],
            "coverage_fraction": [1.0, 1.0, 1.0, 1.0],
            "passes_sfi": [True, True, True, True],
        },
    )
    folds = [
        SelectionFold(
            index=1,
            weight=1.0,
            train_indices=np.asarray([0, 1], dtype=np.int64),
            validation_indices=np.asarray([2, 3], dtype=np.int64),
            train_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-01")),
            validation_start_date=cast(pd.Timestamp, pd.Timestamp("2020-01-02")),
            validation_end_date=cast(pd.Timestamp, pd.Timestamp("2020-01-02")),
        ),
    ]
    thread_ids: set[int] = set()

    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.mda._build_fold_context",
        lambda *args, **kwargs: {
            "fold": folds[0],
            "artifact": object(),
            "validation_frame": frame,
            "baseline_score": 0.0,
        },
    )

    def _fake_score_single_feature_mda(
        feature_name: str,
        fold_contexts: list[dict[str, object]],
        config: FeatureSelectionConfig,
    ) -> dict[str, object]:
        del fold_contexts, config
        time.sleep(0.02)
        thread_ids.add(threading.get_ident())
        return {
            "feature_name": feature_name,
            "mda_mean_delta_objective": 0.001,
            "mda_std_delta_objective": 0.0,
            "mda_fold_positive_share": 1.0,
            "mda_repeat_count": 1,
            "selected": True,
            "mda_drop_reason": "selected",
        }

    monkeypatch.setattr(
        "core.src.meta_model.feature_selection.mda._score_single_feature_mda",
        _fake_score_single_feature_mda,
    )

    result = run_mda_selection(
        cast(Any, cache),
        folds,
        sfi_frame,
        ["feature_a", "feature_b", "feature_c", "feature_d"],
        FeatureSelectionConfig(parallel_workers=4, proxy_training_rounds=8),
    )

    assert len(thread_ids) > 1
    assert result.final_scores["feature_name"].tolist() == [
        "feature_a",
        "feature_b",
        "feature_c",
        "feature_d",
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
