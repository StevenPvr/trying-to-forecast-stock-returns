# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportArgumentType=false
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.model_contract import MODEL_TARGET_COLUMN
from core.src.meta_model.optimize_parameters import main as optimize_main
from core.src.meta_model.optimize_parameters.cv import WalkForwardFold, build_walk_forward_folds
from core.src.meta_model.optimize_parameters.dataset import load_preprocessed_dataset
from core.src.meta_model.optimize_parameters.config import (
    DEFAULT_BOOST_ROUNDS,
    EARLY_STOPPING_ROUNDS,
    OptimizationConfig,
)
from core.src.meta_model.optimize_parameters.objective import aggregate_fold_rank_ic, aggregate_fold_rmse
from core.src.meta_model.optimize_parameters.parallelism import resolve_parallelism
from core.src.meta_model.optimize_parameters.robustness import (
    aggregate_robust_objective,
    build_train_windows,
    compute_complexity_penalty,
)
from core.src.meta_model.optimize_parameters.search_space import suggest_xgboost_params
from core.src.meta_model.optimize_parameters.selection import select_one_standard_error_trial


def _with_model_target(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched[MODEL_TARGET_COLUMN] = enriched["target_main"].astype(float)
    return enriched


def _make_preprocessed_df() -> pd.DataFrame:
    train_dates = pd.date_range("2018-01-01", periods=6, freq="B")
    val_dates = pd.date_range("2019-02-01", periods=10, freq="B")
    rows: list[dict[str, object]] = []
    for date in train_dates:
        rows.append({
            "date": date,
            "ticker": "AAA",
            "target_main": 0.01,
            "dataset_split": "train",
            "feature_a": 1.0,
            "feature_b": 2.0,
        })
    for idx, date in enumerate(val_dates, start=1):
        rows.append({
            "date": date,
            "ticker": "AAA",
            "target_main": 0.01 * idx,
            "dataset_split": "val",
            "feature_a": float(idx),
            "feature_b": float(idx * 2),
        })
    return _with_model_target(
        pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True),
    )


def _make_larger_train_df() -> pd.DataFrame:
    train_dates = pd.date_range("2018-01-01", periods=20, freq="B")
    val_dates = pd.date_range("2019-02-01", periods=10, freq="B")
    rows: list[dict[str, object]] = []
    for idx, date in enumerate(train_dates, start=1):
        rows.append({
            "date": date,
            "ticker": "AAA",
            "target_main": 0.001 * idx,
            "dataset_split": "train",
            "feature_a": float(idx),
            "feature_b": float(idx * 2),
        })
    for idx, date in enumerate(val_dates, start=1):
        rows.append({
            "date": date,
            "ticker": "AAA",
            "target_main": 0.01 * idx,
            "dataset_split": "val",
            "feature_a": float(idx),
            "feature_b": float(idx * 2),
        })
    return _with_model_target(
        pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True),
    )


def _make_embargo_df() -> pd.DataFrame:
    train_dates = pd.date_range("2018-01-01", periods=5, freq="B")
    val_dates = pd.date_range("2018-01-08", periods=4, freq="B")
    rows: list[dict[str, object]] = []
    for idx, date in enumerate(train_dates, start=1):
        rows.append({
            "date": date,
            "ticker": "AAA",
            "target_main": 0.001 * idx,
            "dataset_split": "train",
            "feature_a": float(idx),
            "feature_b": float(idx * 2),
        })
    for idx, date in enumerate(val_dates, start=1):
        rows.append({
            "date": date,
            "ticker": "AAA",
            "target_main": 0.01 * idx,
            "dataset_split": "val",
            "feature_a": float(idx),
            "feature_b": float(idx * 2),
        })
    return _with_model_target(
        pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True),
    )


def _make_preprocessed_with_test_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, start, periods in (
        ("train", "2018-01-01", 4),
        ("val", "2019-02-01", 4),
        ("test", "2022-02-01", 4),
    ):
        for idx, date in enumerate(pd.date_range(start, periods=periods, freq="B"), start=1):
            rows.append({
                "date": date,
                "ticker": "AAA",
                "target_main": 0.01 * idx,
                "dataset_split": split_name,
                "feature_a": float(idx),
                "feature_b": float(idx * 2),
            })
    return _with_model_target(
        pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True),
    )


class TestBuildWalkForwardFolds:
    def test_builds_expanding_train_and_ordered_validation_blocks(self) -> None:
        data = _make_preprocessed_df()

        folds = build_walk_forward_folds(data, fold_count=5, target_horizon_days=1)

        assert len(folds) == 5
        assert [fold.weight for fold in folds] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert folds[0].train_end_date < folds[0].validation_end_date
        assert folds[-1].validation_end_date == pd.Timestamp("2019-02-14")
        assert folds[1].train_row_count > folds[0].train_row_count
        assert all(fold.validation_row_count == 2 for fold in folds)

    def test_applies_forward_label_embargo_before_each_validation_block(self) -> None:
        data = _make_embargo_df()

        folds = build_walk_forward_folds(data, fold_count=2, target_horizon_days=2)

        assert folds[0].train_end_date == pd.Timestamp("2018-01-03")
        assert folds[0].train_row_count == 3
        assert folds[1].train_end_date == pd.Timestamp("2018-01-05")
        assert folds[1].train_row_count == 5


class TestAggregateFoldRmse:
    def test_combines_equal_weight_fold_average_and_stability_penalty(self) -> None:
        score = aggregate_fold_rmse(
            fold_rmse=[0.30, 0.25, 0.22, 0.20, 0.18],
            fold_weights=[1.0, 2.0, 3.0, 4.0, 5.0],
            stability_penalty_alpha=0.10,
        )

        assert round(score["mean_rmse"], 6) == 0.230000
        assert round(score["std_rmse"], 6) == 0.041952
        assert round(score["objective_score"], 6) == 0.234195


class TestAggregateFoldRankIc:
    def test_combines_rank_ic_with_stability_penalties(self) -> None:
        aggregate = aggregate_fold_rank_ic(
            fold_rank_ic=[0.04, 0.03, 0.05, 0.02, 0.01],
            fold_window_std=[0.01, 0.02, 0.01, 0.03, 0.02],
            stability_penalty_alpha=0.10,
            train_window_stability_alpha=0.05,
            complexity_penalty=0.01,
            objective_standard_error=0.002,
        )

        assert round(aggregate["mean_rank_ic"], 6) == 0.030000
        assert round(aggregate["std_rank_ic"], 6) == 0.014142
        assert round(aggregate["objective_score"], 6) == -0.017686


class TestResolveParallelism:
    def test_uses_all_detected_cores_when_total_cores_not_provided(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr(
            "core.src.meta_model.runtime_parallelism.os.cpu_count",
            lambda: 12,
        )

        plan = resolve_parallelism(total_cores=None, fold_count=5)

        assert plan.total_cores == 12
        assert plan.fold_workers == 5
        assert plan.threads_per_fold == 2

    def test_allocates_threads_across_parallel_folds(self) -> None:
        plan = resolve_parallelism(total_cores=11, fold_count=5)

        assert plan.total_cores == 11
        assert plan.fold_workers == 5
        assert plan.threads_per_fold == 2

    def test_caps_fold_workers_when_feature_matrix_is_large(self) -> None:
        plan = resolve_parallelism(
            11,
            5,
            2_500 * 1024 * 1024,
        )

        assert plan.total_cores == 11
        assert plan.fold_workers == 1
        assert plan.threads_per_fold == 11

    def test_forces_single_fold_worker_in_cuda_mode(self) -> None:
        plan = resolve_parallelism(
            16,
            5,
            accelerator="cuda",
        )

        assert plan.total_cores == 16
        assert plan.fold_workers == 1
        assert plan.threads_per_fold == 16


class TestOptimizationDefaults:
    def test_uses_generous_xgboost_early_stopping_defaults(self) -> None:
        assert DEFAULT_BOOST_ROUNDS == 3000
        assert EARLY_STOPPING_ROUNDS == 100


class TestFoldEvaluationBackend:
    @staticmethod
    def _thread_backend_should_not_run(*_: Any, **__: Any) -> list[dict[str, int]]:
        raise AssertionError("thread backend should not be used")

    def test_prefers_process_backend_when_available(self, monkeypatch) -> None:
        dataset_bundle: Any = object()
        fold_contexts = [object(), object()]
        calls: list[str] = []

        monkeypatch.setattr(optimize_main, "_process_pool_available", lambda: True)
        monkeypatch.setattr(
            optimize_main,
            "_install_process_fold_context",
            lambda dataset_bundle, booster_params, optimization_config, fold_matrix_cache_by_index: calls.append("install"),
        )
        monkeypatch.setattr(
            optimize_main,
            "_clear_process_fold_context",
            lambda: calls.append("clear"),
        )
        monkeypatch.setattr(
            optimize_main,
            "_evaluate_trial_folds_in_process_pool",
            lambda folds, *, parallelism_plan: [{"fold_index": 2}, {"fold_index": 1}],
        )
        monkeypatch.setattr(
            optimize_main,
            "_evaluate_trial_folds_in_thread_pool",
            self._thread_backend_should_not_run,
        )

        results = optimize_main._evaluate_trial_folds(
            dataset_bundle,
            fold_contexts,
            {"eta": 0.03},
            OptimizationConfig(),
            resolve_parallelism(total_cores=4, fold_count=2),
        )

        assert results == [{"fold_index": 2}, {"fold_index": 1}]
        assert calls == ["install", "clear"]

    def test_falls_back_to_thread_backend_when_process_backend_is_unavailable(self, monkeypatch) -> None:
        dataset_bundle: Any = object()
        fold_contexts = [object(), object()]

        monkeypatch.setattr(optimize_main, "_process_pool_available", lambda: False)
        monkeypatch.setattr(
            optimize_main,
            "_evaluate_trial_folds_in_thread_pool",
            lambda dataset_bundle, folds, booster_params, optimization_config, *, parallelism_plan, fold_matrix_cache_by_index: [{"fold_index": 1}],
        )

        results = optimize_main._evaluate_trial_folds(
            dataset_bundle,
            fold_contexts,
            {"eta": 0.03},
            OptimizationConfig(),
            resolve_parallelism(total_cores=4, fold_count=2),
        )

        assert results == [{"fold_index": 1}]


class _RecordingTrial:
    def __init__(self) -> None:
        self.float_calls: list[tuple[str, float, float, bool]] = []
        self.int_calls: list[tuple[str, int, int]] = []

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float:
        self.float_calls.append((name, low, high, log))
        return low

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.int_calls.append((name, low, high))
        return low


class TestSearchSpace:
    def test_uses_more_parsimonious_xgboost_ranges(self) -> None:
        trial = _RecordingTrial()

        params = suggest_xgboost_params(
            trial,
            threads_per_fold=4,
            random_seed=7,
            accelerator="cpu",
            gpu_device_id=0,
        )

        assert params["tree_method"] == "hist"
        assert params["nthread"] == 4
        assert ("eta", 0.01, 0.08, True) in trial.float_calls
        assert ("min_child_weight", 0.5, 64.0, True) in trial.float_calls
        assert ("subsample", 0.60, 0.90, False) in trial.float_calls
        assert ("colsample_bytree", 0.40, 0.80, False) in trial.float_calls
        assert ("gamma", 1e-3, 20.0, True) in trial.float_calls
        assert ("lambda", 1e-2, 100.0, True) in trial.float_calls
        assert ("alpha", 1e-3, 25.0, True) in trial.float_calls
        assert ("max_depth", 2, 4) in trial.int_calls
        assert ("max_bin", 64, 256) in trial.int_calls

    def test_injects_cuda_device_parameters_when_accelerator_is_cuda(self) -> None:
        trial = _RecordingTrial()

        params = suggest_xgboost_params(
            trial,
            threads_per_fold=4,
            random_seed=7,
            accelerator="cuda",
            gpu_device_id=2,
        )

        assert params["device"] == "cuda"
        assert params["gpu_id"] == 2


class TestBuildTrainWindows:
    def test_includes_full_tail_and_random_windows(self) -> None:
        data = _make_larger_train_df()
        train_indices = data.index[data["dataset_split"] == "train"].to_numpy(dtype=int)

        windows = build_train_windows(
            data,
            train_indices,
            random_seed=7,
            recent_tail_fraction=0.67,
            random_window_count=1,
            random_window_min_fraction=0.60,
        )

        labels = [window.label for window in windows]
        assert labels[0] == "full"
        assert "recent_tail" in labels
        assert "random_window_1" in labels


class TestRobustObjective:
    def test_penalizes_window_instability_and_complexity(self) -> None:
        aggregate = aggregate_robust_objective(
            fold_rmse=[0.30, 0.25, 0.22, 0.20, 0.18],
            fold_weights=[1.0, 2.0, 3.0, 4.0, 5.0],
            fold_window_std=[0.05, 0.04, 0.03, 0.02, 0.01],
            stability_penalty_alpha=0.10,
            train_window_stability_alpha=0.05,
            complexity_penalty=0.01,
        )

        assert round(aggregate["mean_rmse"], 6) == 0.230000
        assert round(aggregate["std_rmse"], 6) == 0.041952
        assert round(aggregate["window_std_mean"], 6) == 0.030000
        assert round(aggregate["objective_score"], 6) == 0.245695

    def test_standard_error_reflects_full_objective_and_not_rmse_only(self) -> None:
        aggregate = aggregate_robust_objective(
            fold_rmse=[0.20, 0.20, 0.20, 0.20, 0.20],
            fold_weights=[1.0, 2.0, 3.0, 4.0, 5.0],
            fold_window_std=[0.00, 0.02, 0.00, 0.04, 0.00],
            stability_penalty_alpha=0.10,
            train_window_stability_alpha=0.50,
            complexity_penalty=0.0,
        )

        assert aggregate["objective_standard_error"] > 0.0


class TestComplexityPenalty:
    def test_grows_with_depth_and_iterations(self) -> None:
        light = compute_complexity_penalty(
            max_depth=3,
            average_best_iteration=120.0,
            boost_rounds=3000,
            penalty_alpha=0.01,
        )
        heavy = compute_complexity_penalty(
            max_depth=9,
            average_best_iteration=900.0,
            boost_rounds=3000,
            penalty_alpha=0.01,
        )

        assert heavy > light


class TestOneStandardErrorSelection:
    def test_prefers_simpler_trial_within_one_standard_error(self) -> None:
        trials = pd.DataFrame([
            {
                "trial_number": 0,
                "objective_score": 0.2000,
                "objective_standard_error": 0.0100,
                "complexity_penalty": 0.0200,
            },
            {
                "trial_number": 1,
                "objective_score": 0.2050,
                "objective_standard_error": 0.0200,
                "complexity_penalty": 0.0050,
            },
        ])

        selected = select_one_standard_error_trial(trials)

        assert selected.trial_number == 1


class TestOptimizationDatasetLoading:
    def test_excludes_test_split_rows_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "preprocessed.parquet"
        _make_preprocessed_with_test_df().to_parquet(path, index=False)

        loaded = load_preprocessed_dataset(path)

        assert set(loaded["dataset_split"].astype(str).unique()) == {"train", "val"}
        assert len(loaded) == 8


@dataclass
class _FakeBooster:
    best_score: float
    best_iteration: int


class _FakeDMatrix:
    def __init__(self, data: Any, label: Any, feature_names: list[str] | None = None) -> None:
        self.data = data
        self.label = label
        self.feature_names = feature_names or []

    def get_label(self) -> Any:
        return self.label


class _FakeCallbackModule:
    @staticmethod
    def EarlyStopping(**_: Any) -> tuple[str, str]:
        return ("callback", "early_stopping")


class _FakeXGBoostModule:
    DMatrix = _FakeDMatrix
    callback = _FakeCallbackModule()

    @staticmethod
    def train(
        *,
        params: dict[str, Any],
        dtrain: _FakeDMatrix,
        num_boost_round: int,
        evals: list[tuple[_FakeDMatrix, str]],
        custom_metric: Any,
        maximize: bool,
        evals_result: dict[str, dict[str, list[float]]],
        callbacks: list[Any],
        verbose_eval: bool,
    ) -> _FakeBooster:
        del dtrain, callbacks, verbose_eval, maximize
        validation_matrix = evals[0][0]
        score = float(
            0.15
            - abs(float(params["eta"]) - 0.03)
            - 0.01 * int(params["max_depth"])
            - 0.001 * float(validation_matrix.label.mean())
        )
        metric_name, metric_value = custom_metric(
            np.full_like(validation_matrix.label, fill_value=score, dtype=float),
            validation_matrix,
        )
        evals_result["validation"] = {metric_name: [metric_value]}
        return _FakeBooster(best_score=score, best_iteration=min(num_boost_round, 12 + int(params["max_depth"])))


class _FakeTrial:
    def __init__(self, number: int, suggestions: dict[str, float | int]) -> None:
        self.number = number
        self._suggestions = suggestions
        self.params: dict[str, float | int] = {}
        self.user_attrs: dict[str, Any] = {}
        self.value: float | None = None
        self.state = "COMPLETE"

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float:
        del low, high, log
        value = float(self._suggestions[name])
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int) -> int:
        del low, high
        value = int(self._suggestions[name])
        self.params[name] = value
        return value

    def set_user_attr(self, name: str, value: Any) -> None:
        self.user_attrs[name] = value


class _FakeStudy:
    def __init__(self, trial_suggestions: list[dict[str, float | int]]) -> None:
        self._trial_suggestions = trial_suggestions
        self.trials: list[_FakeTrial] = []
        self.best_trial: _FakeTrial | None = None
        self.best_value: float | None = None

    def optimize(self, objective: Any, n_trials: int, n_jobs: int, show_progress_bar: bool) -> None:
        del n_jobs, show_progress_bar
        for trial_number, suggestions in enumerate(self._trial_suggestions[:n_trials]):
            trial = _FakeTrial(trial_number, suggestions)
            trial.value = float(objective(trial))
            self.trials.append(trial)
        if not self.trials:
            raise ValueError("Fake study requires at least one trial.")
        best_trial = min(self.trials, key=self._trial_value)
        self.best_trial = best_trial
        self.best_value = self._trial_value(best_trial)

    @staticmethod
    def _trial_value(trial: _FakeTrial) -> float:
        if trial.value is None:
            raise ValueError("Fake trial has no objective value.")
        return float(trial.value)


class _FakeSampler:
    def __init__(self, seed: int) -> None:
        self.seed = seed


class _FakeOptunaModule:
    class samplers:
        TPESampler = _FakeSampler

    def __init__(self, trial_suggestions: list[dict[str, float | int]]) -> None:
        self._trial_suggestions = trial_suggestions

    def create_study(self, *, study_name: str, direction: str, sampler: Any) -> _FakeStudy:
        del study_name, direction, sampler
        return _FakeStudy(self._trial_suggestions)


class TestOptimizeXGBoostParametersIntegration:
    def test_runs_end_to_end_and_persists_trial_contract(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        _make_preprocessed_df().to_parquet(dataset_path, index=False)
        captured: dict[str, Any] = {}
        fake_optuna = _FakeOptunaModule([
            {
                "eta": 0.02,
                "max_depth": 2,
                "min_child_weight": 1.0,
                "subsample": 0.75,
                "colsample_bytree": 0.50,
                "gamma": 0.01,
                "lambda": 1.0,
                "alpha": 0.01,
                "max_bin": 64,
            },
            {
                "eta": 0.06,
                "max_depth": 4,
                "min_child_weight": 8.0,
                "subsample": 0.85,
                "colsample_bytree": 0.70,
                "gamma": 0.10,
                "lambda": 5.0,
                "alpha": 0.10,
                "max_bin": 128,
            },
        ])

        def _capture_outputs(trials_frame: pd.DataFrame, best_params: dict[str, Any]) -> None:
            captured["trials_frame"] = trials_frame
            captured["best_params"] = best_params

        monkeypatch.setattr(optimize_main, "load_optuna_module", lambda: fake_optuna)
        monkeypatch.setattr(optimize_main, "load_xgboost_module", lambda: _FakeXGBoostModule())
        monkeypatch.setattr(optimize_main, "save_optimization_outputs", _capture_outputs)
        monkeypatch.setattr(optimize_main, "_process_pool_available", lambda: False)

        trials_frame, best_payload = optimize_main.optimize_xgboost_parameters(
            dataset_path,
            optimization_config=OptimizationConfig(trial_count=2, fold_count=5, target_horizon_days=1),
        )

        assert len(trials_frame) == 2
        assert "fold_1_daily_rank_ic" in trials_frame.columns
        assert "mean_daily_rank_ic" in trials_frame.columns
        assert "objective_standard_error" in trials_frame.columns
        assert "selected_trial_one_standard_error" in best_payload
        assert best_payload["selected_trial_one_standard_error"]["trial_number"] in {0, 1}
        assert captured["trials_frame"].equals(trials_frame)
        assert captured["best_params"] == best_payload
        assert best_payload["acceleration"]["accelerator"] == "cpu"

    def test_accepts_custom_study_name_and_output_paths(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        trials_parquet_path = tmp_path / "custom_trials.parquet"
        trials_csv_path = tmp_path / "custom_trials.csv"
        best_params_path = tmp_path / "custom_best_params.json"
        _make_preprocessed_df().to_parquet(dataset_path, index=False)
        captured: dict[str, Any] = {}

        class _CapturingOptunaModule(_FakeOptunaModule):
            def create_study(self, *, study_name: str, direction: str, sampler: Any) -> _FakeStudy:
                captured["study_name"] = study_name
                return super().create_study(study_name=study_name, direction=direction, sampler=sampler)

        fake_optuna = _CapturingOptunaModule([
            {
                "eta": 0.02,
                "max_depth": 2,
                "min_child_weight": 1.0,
                "subsample": 0.75,
                "colsample_bytree": 0.50,
                "gamma": 0.01,
                "lambda": 1.0,
                "alpha": 0.01,
                "max_bin": 64,
            },
        ])

        def _capture_outputs(
            trials_frame: pd.DataFrame,
            best_params: dict[str, Any],
            *,
            trials_parquet_path: Path,
            trials_csv_path: Path,
            best_params_path: Path,
        ) -> None:
            captured["trials_frame"] = trials_frame
            captured["best_params"] = best_params
            captured["trials_parquet_path"] = trials_parquet_path
            captured["trials_csv_path"] = trials_csv_path
            captured["best_params_path"] = best_params_path

        monkeypatch.setattr(optimize_main, "load_optuna_module", lambda: fake_optuna)
        monkeypatch.setattr(optimize_main, "load_xgboost_module", lambda: _FakeXGBoostModule())
        monkeypatch.setattr(optimize_main, "save_optimization_outputs", _capture_outputs)
        monkeypatch.setattr(optimize_main, "_process_pool_available", lambda: False)

        optimize_main.optimize_xgboost_parameters(
            dataset_path,
            optimization_config=OptimizationConfig(trial_count=1, fold_count=5, target_horizon_days=1),
            study_name="secondary_future_trend_5d",
            trials_parquet_path=trials_parquet_path,
            trials_csv_path=trials_csv_path,
            best_params_path=best_params_path,
        )

        assert captured["study_name"] == "secondary_future_trend_5d"
        assert captured["trials_parquet_path"] == trials_parquet_path
        assert captured["trials_csv_path"] == trials_csv_path
        assert captured["best_params_path"] == best_params_path


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
