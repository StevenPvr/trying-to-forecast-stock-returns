from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import pytest
from concurrent.futures import Future
from sklearn.metrics import matthews_corrcoef

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.meta_labeling import features as meta_features
from core.src.meta_model.meta_labeling import labels as meta_labels
from core.src.meta_model.meta_labeling import main as meta_main
from core.src.meta_model.meta_labeling import metrics as meta_metrics
from core.src.meta_model.model_registry.main import ModelSpec


def _dataset() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    rows: list[dict[str, object]] = []
    for idx, date in enumerate(dates, start=1):
        split = "train" if idx <= 7 else "val"
        for ticker, shift in [("AAA", 0.2), ("BBB", -0.1)]:
            rows.append({
                "date": date,
                "ticker": ticker,
                "dataset_split": split,
                "feature_a": float(idx + shift),
                "feature_b": float((idx * 2) + shift),
                "target_week_hold_excess_log_return": float(0.01 if ticker == "AAA" else -0.02),
                "target_week_hold_5sessions_close_log_return": float(0.03 if ticker == "AAA" else -0.01),
                "target_week_hold_5sessions_net_log_return": float(0.02 if ticker == "AAA" else -0.015),
            })
    return pd.DataFrame(rows)


def _meta_panel() -> pd.DataFrame:
    dataset = pd.DataFrame(_dataset().copy())
    dataset["prediction"] = np.where(
        dataset["ticker"].astype(str) == "AAA",
        0.5,
        -0.5,
    ).astype(np.float64)
    aaa_mask = dataset["ticker"].astype(str) == "AAA"
    alternating_positive = (np.arange(len(dataset), dtype=np.int64) % 2) == 0
    dataset.loc[aaa_mask & alternating_positive, "target_week_hold_5sessions_net_log_return"] = 0.02
    dataset.loc[aaa_mask & ~alternating_positive, "target_week_hold_5sessions_net_log_return"] = -0.02
    return meta_labels.attach_meta_labels(
        meta_features.build_primary_context_columns(dataset),
        minimum_target_net_return=0.0,
    )


def test_resolve_primary_burn_start_uses_unique_train_dates() -> None:
    data = _dataset()

    burn_start = meta_main._resolve_primary_burn_start_date(data, burn_fraction=0.20)

    assert burn_start == pd.Timestamp("2024-01-03")


def test_build_meta_labels_marks_only_positive_primary_candidates() -> None:
    frame = pd.DataFrame({
        "primary_prediction": [0.4, 0.2, -0.3],
        "target_week_hold_5sessions_net_log_return": [0.02, -0.01, 0.03],
    })

    labeled = meta_labels.attach_meta_labels(frame, minimum_target_net_return=0.0)

    assert labeled["meta_candidate"].tolist() == [1, 1, 0]
    assert labeled["meta_label"].tolist() == [1, 0, 0]


def test_build_meta_labels_respects_minimum_net_return_threshold() -> None:
    frame = pd.DataFrame({
        "primary_prediction": [0.4, 0.2, 0.3],
        "target_week_hold_5sessions_net_log_return": [0.0030, 0.0015, 0.0020],
    })

    labeled = meta_labels.attach_meta_labels(frame, minimum_target_net_return=0.002)

    assert labeled["meta_candidate"].tolist() == [1, 1, 1]
    assert labeled["meta_label"].tolist() == [1, 0, 0]


def test_refine_primary_signal_uses_positive_confidence_only() -> None:
    frame = pd.DataFrame({
        "prediction": [0.5, -0.2],
        "expected_return_5d": [0.10, -0.04],
        "meta_probability": [0.75, 0.40],
    })

    refined = meta_features.attach_refined_signal_columns(frame)

    assert refined["meta_confidence"].tolist() == pytest.approx([0.75, 0.0])
    assert refined["refined_prediction"].tolist() == pytest.approx([0.375, 0.0])
    assert refined["refined_expected_return_5d"].tolist() == pytest.approx([0.075, 0.0])


def test_binary_logloss_filters_nan_values() -> None:
    frame = pd.DataFrame({
        "meta_label": [1, 0, float("nan"), 1],
        "meta_probability": [0.9, 0.1, 0.5, float("nan")],
    })

    observed = meta_metrics.binary_logloss(frame)

    assert np.isfinite(observed)
    assert observed > 0.0


def test_binary_logloss_returns_inf_when_all_nan() -> None:
    frame = pd.DataFrame({
        "meta_label": [float("nan"), float("nan")],
        "meta_probability": [float("nan"), float("nan")],
    })

    observed = meta_metrics.binary_logloss(frame)

    assert observed == float("inf")


def test_binary_mcc_uses_decision_threshold() -> None:
    frame = pd.DataFrame({
        "meta_label": [1, 1, 0, 0],
        "meta_probability": [0.90, 0.40, 0.60, 0.10],
    })

    observed = meta_metrics.binary_mcc(frame, threshold=0.5, use_balanced_class_weights=False)

    assert observed == pytest.approx(0.0)


def test_binary_mcc_can_use_balanced_class_weights() -> None:
    frame = pd.DataFrame({
        "meta_label": [1, 1, 1, 0],
        "meta_probability": [0.90, 0.80, 0.40, 0.45],
    })

    observed = meta_metrics.binary_mcc(frame, threshold=0.5, use_balanced_class_weights=True)
    expected = matthews_corrcoef(
        np.array([1, 1, 1, 0], dtype=np.int64),
        np.array([1, 1, 0, 0], dtype=np.int64),
        sample_weight=np.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0], dtype=np.float64),
    )

    assert observed == pytest.approx(expected)
    assert observed > meta_metrics.binary_mcc(frame, threshold=0.5, use_balanced_class_weights=False)


def test_build_primary_oos_panels_excludes_burn_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _dataset()
    model_spec = ModelSpec(model_name="ridge", params={"alpha": 1.0})

    def fake_fit_model(
        spec: ModelSpec,
        train_frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> object:
        del spec, feature_columns
        return {"mean_feature": float(train_frame["feature_a"].mean())}

    def fake_predict_model(
        artifact: object,
        frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> np.ndarray:
        del feature_columns
        mean_feature = float(dict(artifact)["mean_feature"])
        return frame["feature_a"].to_numpy(dtype=np.float64) - mean_feature

    monkeypatch.setattr(meta_main, "fit_model", fake_fit_model)
    monkeypatch.setattr(meta_main, "predict_model", fake_predict_model)

    train_tail, val_panel = meta_main._build_primary_oos_panels(
        data,
        feature_columns=["feature_a", "feature_b"],
        model_spec=model_spec,
        burn_fraction=0.20,
    )

    assert not train_tail.empty
    assert not val_panel.empty
    assert pd.Timestamp(train_tail["date"].min()) >= pd.Timestamp("2024-01-03")
    assert set(train_tail["dataset_split"].astype(str).unique()) == {"train"}
    assert set(val_panel["dataset_split"].astype(str).unique()) == {"val"}
    assert "primary_prediction" in train_tail.columns
    assert "primary_prediction_rank_cs" in val_panel.columns


def test_build_primary_oos_panels_parallelizes_with_single_thread_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _dataset()
    model_spec = ModelSpec(
        model_name="xgboost",
        params={"eta": 0.1, "nthread": 7},
        training_rounds=25,
    )
    submitted_specs: list[ModelSpec] = []

    def fake_build_primary_prediction_part(
        *,
        data: pd.DataFrame,
        feature_columns: list[str],
        model_spec: ModelSpec,
        prediction_date: pd.Timestamp,
    ) -> pd.DataFrame:
        del data, feature_columns
        submitted_specs.append(model_spec)
        return pd.DataFrame({
            "date": [prediction_date],
            "ticker": ["AAA"],
            "dataset_split": ["train" if prediction_date <= pd.Timestamp("2024-01-09") else "val"],
            "primary_prediction": [0.1],
            "primary_prediction_rank_cs": [1.0],
            "primary_prediction_zscore_cs": [0.0],
            "primary_prediction_abs": [0.1],
            "primary_prediction_sign": [1.0],
        })

    class _InlineExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> "_InlineExecutor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type, exc, tb
            return False

        def submit(self, fn: object, *args: object, **kwargs: object) -> Future[pd.DataFrame]:
            future: Future[pd.DataFrame] = Future()
            callable_fn = fn
            if not callable(callable_fn):
                raise TypeError("submit expected a callable")
            result = callable_fn(*args, **kwargs)
            future.set_result(result)
            return future

    def fake_as_completed(
        futures: list[Future[pd.DataFrame]],
    ) -> list[Future[pd.DataFrame]]:
        return futures

    monkeypatch.setattr(meta_main, "_build_primary_prediction_part", fake_build_primary_prediction_part)
    monkeypatch.setattr(meta_main, "ThreadPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(meta_main, "as_completed", fake_as_completed)

    train_tail, val_panel = meta_main._build_primary_oos_panels(
        data,
        feature_columns=["feature_a", "feature_b"],
        model_spec=model_spec,
        burn_fraction=0.20,
        parallel_workers=2,
    )

    assert not train_tail.empty
    assert not val_panel.empty
    assert submitted_specs
    assert all(spec.params["nthread"] == 1 for spec in submitted_specs)
    assert all(spec.training_rounds == 25 for spec in submitted_specs)


def test_load_or_build_primary_oos_panels_reuses_cached_parquets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cached_train = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03"]),
        "ticker": ["AAA"],
        "dataset_split": ["train"],
        "prediction": [0.1],
        "primary_prediction": [0.1],
        "primary_prediction_rank_cs": [1.0],
        "primary_prediction_zscore_cs": [0.0],
        "primary_prediction_abs": [0.1],
        "primary_prediction_sign": [1.0],
        "target_week_hold_5sessions_net_log_return": [0.02],
    })
    cached_val = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10"]),
        "ticker": ["BBB"],
        "dataset_split": ["val"],
        "prediction": [0.2],
        "primary_prediction": [0.2],
        "primary_prediction_rank_cs": [1.0],
        "primary_prediction_zscore_cs": [0.0],
        "primary_prediction_abs": [0.2],
        "primary_prediction_sign": [1.0],
        "target_week_hold_5sessions_net_log_return": [0.01],
    })
    train_path = tmp_path / "primary_train.parquet"
    val_path = tmp_path / "primary_val.parquet"
    cached_train.to_parquet(train_path, index=False)
    cached_val.to_parquet(val_path, index=False)

    def fail_if_called(
        data: pd.DataFrame,
        *,
        feature_columns: list[str],
        model_spec: ModelSpec,
        burn_fraction: float,
        parallel_workers: int | None = None,
    ) -> NoReturn:
        del data, feature_columns, model_spec, burn_fraction, parallel_workers
        raise AssertionError("primary OOS builder should not run when cache exists")

    monkeypatch.setattr(meta_main, "META_PRIMARY_OOS_TRAIN_TAIL_PARQUET", train_path)
    monkeypatch.setattr(meta_main, "META_PRIMARY_OOS_VAL_PARQUET", val_path)
    monkeypatch.setattr(meta_main, "_build_primary_oos_panels", fail_if_called)

    train_tail, val_panel = meta_main._load_or_build_primary_oos_panels(
        _dataset(),
        feature_columns=["feature_a", "feature_b"],
        model_spec=ModelSpec(model_name="ridge", params={"alpha": 1.0}),
        burn_fraction=0.20,
        parallel_workers=2,
    )

    assert train_tail.equals(cached_train)
    assert val_panel.equals(cached_val)


def test_load_or_build_primary_oos_panels_rebuilds_cache_missing_required_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cached_train = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03"]),
        "ticker": ["AAA"],
        "dataset_split": ["train"],
        "prediction": [0.1],
        "primary_prediction": [0.1],
        "primary_prediction_rank_cs": [1.0],
        "primary_prediction_zscore_cs": [0.0],
        "primary_prediction_abs": [0.1],
        "primary_prediction_sign": [1.0],
    })
    cached_val = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10"]),
        "ticker": ["BBB"],
        "dataset_split": ["val"],
        "prediction": [0.2],
        "primary_prediction": [0.2],
        "primary_prediction_rank_cs": [1.0],
        "primary_prediction_zscore_cs": [0.0],
        "primary_prediction_abs": [0.2],
        "primary_prediction_sign": [1.0],
    })
    train_path = tmp_path / "primary_train_missing_net.parquet"
    val_path = tmp_path / "primary_val_missing_net.parquet"
    cached_train.to_parquet(train_path, index=False)
    cached_val.to_parquet(val_path, index=False)

    rebuilt_train = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03"]),
        "ticker": ["AAA"],
        "dataset_split": ["train"],
        "prediction": [0.1],
        "primary_prediction": [0.1],
        "primary_prediction_rank_cs": [1.0],
        "primary_prediction_zscore_cs": [0.0],
        "primary_prediction_abs": [0.1],
        "primary_prediction_sign": [1.0],
        "target_week_hold_5sessions_net_log_return": [0.02],
    })
    rebuilt_val = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10"]),
        "ticker": ["BBB"],
        "dataset_split": ["val"],
        "prediction": [0.2],
        "primary_prediction": [0.2],
        "primary_prediction_rank_cs": [1.0],
        "primary_prediction_zscore_cs": [0.0],
        "primary_prediction_abs": [0.2],
        "primary_prediction_sign": [1.0],
        "target_week_hold_5sessions_net_log_return": [0.01],
    })

    def fake_rebuild(
        data: pd.DataFrame,
        *,
        feature_columns: list[str],
        model_spec: ModelSpec,
        burn_fraction: float,
        parallel_workers: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        del data, feature_columns, model_spec, burn_fraction, parallel_workers
        return rebuilt_train, rebuilt_val

    monkeypatch.setattr(meta_main, "META_PRIMARY_OOS_TRAIN_TAIL_PARQUET", train_path)
    monkeypatch.setattr(meta_main, "META_PRIMARY_OOS_VAL_PARQUET", val_path)
    monkeypatch.setattr(meta_main, "_build_primary_oos_panels", fake_rebuild)

    train_tail, val_panel = meta_main._load_or_build_primary_oos_panels(
        _dataset(),
        feature_columns=["feature_a", "feature_b"],
        model_spec=ModelSpec(model_name="ridge", params={"alpha": 1.0}),
        burn_fraction=0.20,
        parallel_workers=2,
    )

    assert train_tail.equals(rebuilt_train)
    assert val_panel.equals(rebuilt_val)


def test_build_meta_oof_predictions_reuses_selected_training_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta_train_panel = _meta_panel()
    fit_calls: list[dict[str, object]] = []

    def fake_fit_meta_model(
        train_frame: pd.DataFrame,
        feature_columns: list[str],
        *,
        label_column: str,
        params: dict[str, object],
        training_rounds: int,
        early_stopping_rounds: int | None = None,
        early_stopping_validation_fraction: float = 0.10,
        minimum_training_rounds: int = 1,
    ) -> object:
        del train_frame, feature_columns, label_column, params, early_stopping_validation_fraction, minimum_training_rounds
        fit_calls.append({
            "training_rounds": training_rounds,
            "early_stopping_rounds": early_stopping_rounds,
        })
        return object()

    monkeypatch.setattr(meta_main, "fit_meta_model", fake_fit_meta_model)
    monkeypatch.setattr(meta_main, "has_binary_label_support", lambda frame: True)
    monkeypatch.setattr(
        meta_main,
        "predict_meta_model",
        lambda artifact, frame: np.full(len(frame), 0.55, dtype=np.float64),
    )

    oof_predictions = meta_main._build_meta_oof_predictions(
        meta_train_panel,
        meta_feature_columns=["feature_a", "feature_b"],
        runtime=meta_main.MetaLabelingConfig(),
        best_params={
            "learning_rate": 0.1,
            "max_depth": 4,
            "min_child_weight": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "lambda": 1.0,
            "alpha": 0.0,
            "max_bin": 128,
            "scale_pos_weight": 1.0,
            "selected_training_rounds": 42,
        },
    )

    assert not oof_predictions.empty
    assert fit_calls
    assert all(call["training_rounds"] == 42 for call in fit_calls)
    assert all(call["early_stopping_rounds"] is None for call in fit_calls)
    assert set(oof_predictions.loc[oof_predictions["ticker"] == "BBB", "meta_probability"].tolist()) == {0.0}


def test_fit_final_meta_outputs_reuses_selected_training_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta_train_panel = _meta_panel()
    meta_val_panel = pd.DataFrame(meta_train_panel.loc[meta_train_panel["dataset_split"] == "val"].copy())
    fit_calls: list[dict[str, object]] = []

    def fake_fit_meta_model(
        train_frame: pd.DataFrame,
        feature_columns: list[str],
        *,
        label_column: str,
        params: dict[str, object],
        training_rounds: int,
        early_stopping_rounds: int | None = None,
        early_stopping_validation_fraction: float = 0.10,
        minimum_training_rounds: int = 1,
    ) -> object:
        del train_frame, feature_columns, label_column, params, early_stopping_validation_fraction, minimum_training_rounds
        fit_calls.append({
            "training_rounds": training_rounds,
            "early_stopping_rounds": early_stopping_rounds,
        })
        return object()

    monkeypatch.setattr(meta_main, "fit_meta_model", fake_fit_meta_model)
    monkeypatch.setattr(
        meta_main,
        "predict_meta_model",
        lambda artifact, frame: np.full(len(frame), 0.56, dtype=np.float64),
    )
    monkeypatch.setattr(
        meta_main,
        "serialize_meta_model_artifact",
        lambda artifact: {"artifact": "ok"},
    )

    payload, scored_val = meta_main._fit_final_meta_outputs(
        meta_train_panel,
        meta_val_panel,
        meta_feature_columns=["feature_a", "feature_b"],
        runtime=meta_main.MetaLabelingConfig(),
        best_params={
            "learning_rate": 0.1,
            "max_depth": 4,
            "min_child_weight": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "lambda": 1.0,
            "alpha": 0.0,
            "max_bin": 128,
            "scale_pos_weight": 1.0,
            "selected_training_rounds": 37,
        },
    )

    assert payload == {"artifact": "ok"}
    assert not scored_val.empty
    assert fit_calls == [{"training_rounds": 37, "early_stopping_rounds": None}]
    assert set(scored_val.loc[scored_val["ticker"] == "BBB", "meta_probability"].tolist()) == {0.0}


def test_calibration_holdout_dates_disjoint_from_optuna_folds() -> None:
    meta_train_panel = _meta_panel()
    train_only = pd.DataFrame(meta_train_panel.loc[meta_train_panel["dataset_split"] == "train"].copy())
    unique_dates = pd.Index(pd.to_datetime(train_only["date"]).drop_duplicates().sort_values())
    holdout_fraction = 0.20
    holdout_count = max(1, int(np.ceil(len(unique_dates) * holdout_fraction)))
    holdout_dates = set(unique_dates[-holdout_count:].tolist())
    optuna_panel = pd.DataFrame(train_only.loc[~train_only["date"].isin(holdout_dates)].copy())

    from core.src.meta_model.meta_labeling.study import build_meta_folds

    folds = build_meta_folds(optuna_panel, fold_count=4)

    for _train_cutoff, fold_dates in folds:
        fold_date_set = set(fold_dates)
        overlap = fold_date_set & holdout_dates
        assert not overlap, f"Calibration holdout dates leaked into Optuna folds: {overlap}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
