from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.paths import FEATURE_CORR_PCA_OUTPUT_PARQUET
from core.src.meta_model.feature_selection_lag.greedy_forward_selection.main import (
    build_candidate_feature_columns,
    create_retained_features_summary,
    create_selected_features_summary,
    load_selection_scaffold,
    load_train_feature_series,
    load_ranked_candidate_feature_columns,
    run_greedy_forward_selection,
)


def _make_feature_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2010-01-01", periods=140)
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(("AAA", "BBB"), start=1):
        signal_1 = np.sin(np.linspace(0.0, 8.0, len(dates))) + (0.2 * ticker_index)
        signal_5 = np.cos(np.linspace(0.0, 5.0, len(dates))) * 0.6
        noise = np.zeros(len(dates), dtype=float)
        close = 100.0 + np.cumsum(rng.normal(0.05, 0.01, len(dates)))
        ticker_frame = pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "stock_close_price": close,
            "signal_a_lag_1d": signal_1,
            "signal_b_lag_5d": signal_5,
            "noise_lag_1d": noise,
            "context_feature": np.zeros(len(dates), dtype=float),
        })
        rows.extend(ticker_frame.to_dict(orient="records"))
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def _make_selection_scaffold(feature_dataset: pd.DataFrame) -> pd.DataFrame:
    target = (
        0.7 * feature_dataset["signal_a_lag_1d"].to_numpy()
        + 0.4 * feature_dataset["signal_b_lag_5d"].to_numpy()
    )
    scaffold = feature_dataset.loc[:, ["date", "ticker"]].copy()
    scaffold["row_position"] = np.arange(len(scaffold), dtype=np.int64)
    scaffold["target_main"] = target
    cutoff_train = int(len(scaffold) * 0.7)
    cutoff_val = int(len(scaffold) * 0.85)
    scaffold["dataset_split"] = "test"
    scaffold.loc[: cutoff_train - 1, "dataset_split"] = "train"
    scaffold.loc[cutoff_train: cutoff_val - 1, "dataset_split"] = "val"
    return scaffold.loc[:, ["row_position", "date", "ticker", "target_main", "dataset_split"]]


class FakeCatBoostRegressor:
    def __init__(self, **kwargs) -> None:
        self.coefficients: np.ndarray | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series],
        early_stopping_rounds: int,
        use_best_model: bool,
        verbose: bool = False,
    ) -> "FakeCatBoostRegressor":
        del eval_set, early_stopping_rounds, use_best_model, verbose
        design = np.asarray(X, dtype=float)
        augmented = np.column_stack([np.ones(len(design)), design])
        target = np.asarray(y, dtype=float)
        self.coefficients = np.linalg.pinv(augmented) @ target
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model must be fitted before prediction.")
        design = np.asarray(X, dtype=float)
        augmented = np.column_stack([np.ones(len(design)), design])
        return augmented @ self.coefficients


class TestBuildCandidateFeatureColumns:
    def test_discovers_all_non_identifier_features(self, tmp_path: Path) -> None:
        path = tmp_path / "features.parquet"
        _make_feature_dataset().to_parquet(path, index=False)

        result = build_candidate_feature_columns(path)

        assert result == [
            "context_feature",
            "noise_lag_1d",
            "signal_a_lag_1d",
            "signal_b_lag_5d",
            "stock_close_price",
        ]

    def test_prioritizes_ta_and_quant_features_before_other_families(self, tmp_path: Path) -> None:
        path = tmp_path / "prioritized_features.parquet"
        pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=2, freq="B"),
            "ticker": ["AAA", "AAA"],
            "macro_signal": [1.0, 2.0],
            "calendar_event": [0.0, 1.0],
            "quant_alpha": [0.2, 0.3],
            "ta_rsi": [50.0, 55.0],
            "ta_macd": [0.1, 0.2],
            "quant_beta": [1.2, 1.1],
        }).to_parquet(path, index=False)

        result = build_candidate_feature_columns(path)

        assert result == [
            "ta_macd",
            "ta_rsi",
            "quant_alpha",
            "quant_beta",
            "macro_signal",
            "calendar_event",
        ]

    def test_orders_remaining_feature_families_after_ta_and_quant(self, tmp_path: Path) -> None:
        path = tmp_path / "prioritized_other_families.parquet"
        pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=2, freq="B"),
            "ticker": ["AAA", "AAA"],
            "cross_asset_spy": [1.0, 1.1],
            "calendar_fomc": [0.0, 1.0],
            "macro_rate": [4.0, 4.1],
            "sentiment_aaii": [0.2, 0.25],
            "company_sector_code": [1.0, 1.0],
            "stock_close_price": [100.0, 101.0],
            "other_feature": [7.0, 8.0],
        }).to_parquet(path, index=False)

        result = build_candidate_feature_columns(path)

        assert result == [
            "macro_rate",
            "calendar_fomc",
            "sentiment_aaii",
            "cross_asset_spy",
            "company_sector_code",
            "other_feature",
            "stock_close_price",
        ]


class TestLoadTrainFeatureSeries:
    def test_reads_only_requested_feature_column(self, tmp_path: Path) -> None:
        path = tmp_path / "features.parquet"
        dataset = _make_feature_dataset()
        scaffold = _make_selection_scaffold(dataset)
        dataset.to_parquet(path, index=False)
        original_read_parquet = pd.read_parquet
        captured_columns: list[tuple[str, ...] | None] = []

        def spy_read_parquet(*args, **kwargs):
            captured_columns.append(tuple(kwargs.get("columns", [])) or None)
            return original_read_parquet(*args, **kwargs)

        with patch(
            "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.pd.read_parquet",
            side_effect=spy_read_parquet,
        ):
            series = load_train_feature_series(path, scaffold, "signal_a_lag_1d")

        assert ("signal_a_lag_1d",) in captured_columns
        assert len(series) == len(scaffold)


class TestLoadRankedCandidateFeatureColumns:
    def test_discovers_available_features_without_sfi_dependency(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        _make_feature_dataset().to_parquet(feature_path, index=False)

        result = load_ranked_candidate_feature_columns(feature_path)

        assert result == [
            "context_feature",
            "noise_lag_1d",
            "signal_a_lag_1d",
            "signal_b_lag_5d",
            "stock_close_price",
        ]


class TestCreateSelectedFeaturesSummary:
    def test_keeps_only_rows_marked_as_selected(self) -> None:
        evaluations = pd.DataFrame([
            {
                "iteration": 1,
                "candidate_feature_name": "signal_a_lag_1d",
                "selected_at_iteration": True,
                "trial_rmse": 0.10,
                "previous_best_rmse": 0.12,
                "marginal_rmse_gain": 0.02,
            },
            {
                "iteration": 1,
                "candidate_feature_name": "noise_lag_1d",
                "selected_at_iteration": False,
                "trial_rmse": 0.11,
                "previous_best_rmse": 0.12,
                "marginal_rmse_gain": 0.01,
            },
            {
                "iteration": 2,
                "candidate_feature_name": "signal_b_lag_5d",
                "selected_at_iteration": True,
                "trial_rmse": 0.08,
                "previous_best_rmse": 0.10,
                "marginal_rmse_gain": 0.02,
            },
        ])

        result = create_selected_features_summary(evaluations)

        assert list(result["candidate_feature_name"]) == ["signal_a_lag_1d", "signal_b_lag_5d"]


class TestCreateRetainedFeaturesSummary:
    def test_keeps_only_features_selected_by_greedy(self) -> None:
        selected_features = pd.DataFrame([
            {"candidate_feature_name": "signal_a_lag_1d"},
            {"candidate_feature_name": "signal_b_lag_5d"},
        ])

        result = create_retained_features_summary(selected_features)

        assert list(result["feature_name"]) == ["signal_a_lag_1d", "signal_b_lag_5d"]


class TestRunLagGreedyForwardSelection:
    def test_uses_small_catboost_with_best_model_early_stopping(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        dataset = _make_feature_dataset()
        dataset.to_parquet(feature_path, index=False)
        scaffold = _make_selection_scaffold(dataset)

        captured_init_kwargs: dict[str, object] = {}
        captured_fit_kwargs: dict[str, object] = {}

        class CapturingCatBoostRegressor:
            def __init__(self, **kwargs) -> None:
                captured_init_kwargs.update(kwargs)

            def fit(
                self,
                X: pd.DataFrame,
                y: pd.Series,
                *,
                eval_set: tuple[pd.DataFrame, pd.Series],
                early_stopping_rounds: int,
                use_best_model: bool,
                verbose: bool = False,
            ) -> "CapturingCatBoostRegressor":
                del X, y
                captured_fit_kwargs.update({
                    "eval_set": eval_set,
                    "early_stopping_rounds": early_stopping_rounds,
                    "use_best_model": use_best_model,
                    "verbose": verbose,
                })
                return self

            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(X), dtype=float)

        with (
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_selection_scaffold",
                return_value=scaffold,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_catboost_regressor_class",
                return_value=CapturingCatBoostRegressor,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_ranked_candidate_feature_columns",
                return_value=["signal_a_lag_1d"],
            ),
        ):
            run_greedy_forward_selection(feature_path)

        assert captured_init_kwargs == {
            "random_seed": 7,
            "allow_writing_files": False,
            "iterations": 3000,
            "depth": 4,
            "learning_rate": 0.03,
            "loss_function": "RMSE",
        }
        assert captured_fit_kwargs["early_stopping_rounds"] == 100
        assert captured_fit_kwargs["use_best_model"] is True
        assert captured_fit_kwargs["verbose"] is False
        assert isinstance(captured_fit_kwargs["eval_set"], tuple)

    def test_uses_feature_corr_pca_output_as_default_input(self) -> None:
        default_path = inspect.signature(run_greedy_forward_selection).parameters["feature_parquet_path"].default
        assert default_path == FEATURE_CORR_PCA_OUTPUT_PARQUET

    def test_starts_when_first_feature_beats_baseline_then_retries_earlier_candidates(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        dataset = _make_feature_dataset()
        dataset.to_parquet(feature_path, index=False)
        scaffold = _make_selection_scaffold(dataset)

        with (
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_selection_scaffold",
                return_value=scaffold,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_catboost_regressor_class",
                return_value=FakeCatBoostRegressor,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_ranked_candidate_feature_columns",
                return_value=["noise_lag_1d", "signal_b_lag_5d", "signal_a_lag_1d"],
            ),
        ):
            evaluation_scores, selected_features = run_greedy_forward_selection(feature_path)

        assert list(selected_features["candidate_feature_name"][:2]) == [
            "signal_b_lag_5d",
            "signal_a_lag_1d",
        ]
        assert list(evaluation_scores["candidate_feature_name"][:4]) == [
            "noise_lag_1d",
            "signal_b_lag_5d",
            "noise_lag_1d",
            "signal_a_lag_1d",
        ]

    def test_retries_all_remaining_candidates_after_a_run_adds_new_survivors(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        dataset = _make_feature_dataset()
        dataset.to_parquet(feature_path, index=False)
        scaffold = _make_selection_scaffold(dataset)

        score_map = {
            ("ta_seed",): 0.40,
            ("ta_seed", "macro_retry"): 0.40,
            ("ta_seed", "quant_runner_up"): 0.30,
            ("ta_seed", "quant_runner_up", "macro_retry"): 0.20,
        }

        def fake_load_train_feature_series(
            path: Path,
            train_rows: pd.DataFrame,
            feature_name: str,
        ) -> pd.Series:
            del path
            offset = float(len(feature_name))
            return pd.Series(
                np.arange(len(train_rows), dtype=float) + offset,
                index=train_rows.index,
                name=feature_name,
            )

        def fake_score_feature_subset(
            feature_frame: pd.DataFrame,
            feature_names: list[str],
            holdout_window: object,
            model_config: object | None = None,
        ) -> dict[str, object]:
            del feature_frame, holdout_window, model_config
            trial_rmse = score_map[tuple(feature_names)]
            return {
                "mean_rmse": trial_rmse,
                "baseline_rmse": 1.0,
                "mean_r2": 0.0,
                "train_row_count": 10,
                "validation_row_count": 5,
            }

        with (
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_selection_scaffold",
                return_value=scaffold,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_ranked_candidate_feature_columns",
                return_value=["ta_seed", "macro_retry", "quant_runner_up"],
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_train_feature_series",
                side_effect=fake_load_train_feature_series,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.score_feature_subset",
                side_effect=fake_score_feature_subset,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.build_filtered_feature_dataset",
                return_value=pd.DataFrame(),
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main._save_greedy_forward_selection_outputs",
                return_value=None,
            ),
        ):
            evaluation_scores, selected_features = run_greedy_forward_selection(feature_path)

        assert list(selected_features["candidate_feature_name"]) == [
            "ta_seed",
            "quant_runner_up",
            "macro_retry",
        ]
        assert list(evaluation_scores["candidate_feature_name"]) == [
            "ta_seed",
            "macro_retry",
            "quant_runner_up",
            "macro_retry",
        ]
        assert list(evaluation_scores["selected_at_iteration"]) == [True, False, True, True]

    def test_runs_greedy_forward_selection_and_filters_output_dataset(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        scores_parquet = tmp_path / "greedy_scores.parquet"
        scores_csv = tmp_path / "greedy_scores.csv"
        selected_parquet = tmp_path / "greedy_selected.parquet"
        selected_csv = tmp_path / "greedy_selected.csv"
        filtered_parquet = tmp_path / "greedy_filtered.parquet"
        filtered_csv = tmp_path / "greedy_filtered.csv"
        dataset = _make_feature_dataset()
        dataset["empty_lag_2d"] = np.nan
        scaffold = _make_selection_scaffold(dataset)
        dataset.to_parquet(feature_path, index=False)

        with (
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_selection_scaffold",
                return_value=scaffold,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_catboost_regressor_class",
                return_value=FakeCatBoostRegressor,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.GREEDY_FORWARD_SELECTION_SCORES_PARQUET",
                scores_parquet,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.GREEDY_FORWARD_SELECTION_SCORES_CSV",
                scores_csv,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.GREEDY_FORWARD_SELECTION_SELECTED_FEATURES_PARQUET",
                selected_parquet,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.GREEDY_FORWARD_SELECTION_SELECTED_FEATURES_CSV",
                selected_csv,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_PARQUET",
                filtered_parquet,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_CSV",
                filtered_csv,
            ),
        ):
            evaluation_scores, selected_features = run_greedy_forward_selection(feature_path)

        assert scores_parquet.exists()
        assert scores_csv.exists()
        assert selected_parquet.exists()
        assert selected_csv.exists()
        assert filtered_parquet.exists()
        assert filtered_csv.exists()
        assert list(selected_features["candidate_feature_name"]) == [
            "signal_a_lag_1d",
            "signal_b_lag_5d",
        ]
        assert set(evaluation_scores["candidate_feature_name"]) == {
            "context_feature",
            "empty_lag_2d",
            "noise_lag_1d",
            "signal_a_lag_1d",
            "signal_b_lag_5d",
            "stock_close_price",
        }
        selected_names = set(
            evaluation_scores.loc[
                evaluation_scores["selected_at_iteration"],
                "candidate_feature_name",
            ].tolist(),
        )
        assert selected_names == {"signal_a_lag_1d", "signal_b_lag_5d"}
        assert evaluation_scores.loc[
            evaluation_scores["candidate_feature_name"] == "context_feature",
            "rejection_reason",
        ].iloc[0] == "constant_feature"
        assert evaluation_scores.loc[
            evaluation_scores["candidate_feature_name"] == "empty_lag_2d",
            "rejection_reason",
        ].iloc[0] == "empty_feature_frame"
        filtered_dataset = pd.read_parquet(filtered_parquet)
        retained_features = {
            column_name
            for column_name in filtered_dataset.columns
            if column_name not in {"date", "ticker", "stock_close_price", "target_main"}
        }
        assert retained_features == {
            "signal_a_lag_1d",
            "signal_b_lag_5d",
        }
        assert bool((evaluation_scores["selected_at_iteration"].sum()) == 2)

    def test_rejects_candidate_when_catboost_reports_all_features_constant_or_ignored(
        self,
        tmp_path: Path,
    ) -> None:
        feature_path = tmp_path / "features.parquet"
        dataset = _make_feature_dataset()
        dataset["fragile_feature"] = np.linspace(0.0, 1.0, len(dataset))
        dataset.to_parquet(feature_path, index=False)
        scaffold = _make_selection_scaffold(dataset)

        class FailingCatBoostRegressor:
            def __init__(self, **kwargs) -> None:
                del kwargs

            def fit(
                self,
                X: pd.DataFrame,
                y: pd.Series,
                *,
                eval_set: tuple[pd.DataFrame, pd.Series],
                early_stopping_rounds: int,
                use_best_model: bool,
                verbose: bool = False,
            ) -> "FailingCatBoostRegressor":
                del X, y, eval_set, early_stopping_rounds, use_best_model, verbose
                raise RuntimeError("All features are either constant or ignored.")

            def predict(self, X: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(X), dtype=float)

        with (
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_selection_scaffold",
                return_value=scaffold,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_catboost_regressor_class",
                return_value=FailingCatBoostRegressor,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_ranked_candidate_feature_columns",
                return_value=["fragile_feature"],
            ),
        ):
            evaluation_scores, selected_features = run_greedy_forward_selection(feature_path)

        assert selected_features.empty
        assert evaluation_scores.loc[0, "candidate_feature_name"] == "fragile_feature"
        assert not bool(evaluation_scores.loc[0, "selected_at_iteration"])
        assert evaluation_scores.loc[0, "rejection_reason"] == "catboost_all_features_constant_or_ignored"

    def test_rejects_candidate_when_no_valid_train_or_validation_rows_remain(
        self,
        tmp_path: Path,
    ) -> None:
        feature_path = tmp_path / "features.parquet"
        dataset = _make_feature_dataset()
        dataset["fragile_feature"] = np.linspace(0.0, 1.0, len(dataset))
        dataset.to_parquet(feature_path, index=False)
        scaffold = _make_selection_scaffold(dataset)

        with (
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_selection_scaffold",
                return_value=scaffold,
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.load_ranked_candidate_feature_columns",
                return_value=["fragile_feature"],
            ),
            patch(
                "core.src.meta_model.feature_selection_lag.greedy_forward_selection.main.score_feature_subset",
                side_effect=ValueError("No valid train/validation rows remained for subset scoring."),
            ),
        ):
            evaluation_scores, selected_features = run_greedy_forward_selection(feature_path)

        assert selected_features.empty
        assert evaluation_scores.loc[0, "candidate_feature_name"] == "fragile_feature"
        assert not bool(evaluation_scores.loc[0, "selected_at_iteration"])
        assert evaluation_scores.loc[0, "rejection_reason"] == "no_valid_train_validation_rows"
