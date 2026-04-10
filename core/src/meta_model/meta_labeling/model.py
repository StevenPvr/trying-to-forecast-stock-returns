from __future__ import annotations

"""Binary meta-model fit/predict/serialization utilities."""

import logging
import time
from dataclasses import dataclass, field
import base64
import math
from typing import Any

import numpy as np
import pandas as pd

from core.src.meta_model.optimize_parameters.search_space import load_xgboost_module
from core.src.meta_model.xgboost_dmatrix import build_xgboost_dmatrix, prepare_xgboost_feature_frame

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetaModelArtifact:
    feature_names: list[str]
    params: dict[str, Any]
    training_rounds: int
    fitted_object: Any
    training_metadata: dict[str, Any] = field(default_factory=dict)


def _split_early_stopping_frame(
    train_frame: pd.DataFrame,
    *,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    ordered = pd.DataFrame(train_frame.sort_values(["date", "ticker"]).reset_index(drop=True))
    unique_dates = pd.Index(pd.to_datetime(ordered["date"]).drop_duplicates().sort_values())
    if len(unique_dates) < 3:
        return ordered, None
    eval_date_count = max(1, int(math.ceil(len(unique_dates) * validation_fraction)))
    eval_date_count = min(eval_date_count, len(unique_dates) - 1)
    eval_dates = pd.Index(unique_dates[-eval_date_count:])
    fit_frame = pd.DataFrame(ordered.loc[~pd.to_datetime(ordered["date"]).isin(eval_dates)].copy())
    eval_frame = pd.DataFrame(ordered.loc[pd.to_datetime(ordered["date"]).isin(eval_dates)].copy())
    if fit_frame.empty or eval_frame.empty:
        return ordered, None
    return fit_frame, eval_frame


def _build_label_vector(
    frame: pd.DataFrame,
    *,
    label_column: str,
) -> np.ndarray:
    return pd.to_numeric(frame[label_column], errors="coerce").to_numpy(dtype=np.float64)


def _build_training_matrix_bundle(
    xgb: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    label_column: str,
) -> tuple[pd.DataFrame, np.ndarray, Any]:
    feature_frame = prepare_xgboost_feature_frame(frame, feature_columns)
    labels = _build_label_vector(frame, label_column=label_column)
    finite_label_mask = np.isfinite(labels)
    if not bool(finite_label_mask.all()):
        feature_frame = pd.DataFrame(feature_frame.loc[finite_label_mask].copy())
        labels = labels[finite_label_mask]
    if labels.size == 0:
        raise ValueError("meta-model training frame has no finite label values.")
    matrix = build_xgboost_dmatrix(
        xgb,
        feature_frame,
        labels.astype(np.float32, copy=False),
    )
    return feature_frame, labels, matrix


def _resolve_selected_training_rounds(
    search_booster: Any,
    *,
    training_rounds: int,
    minimum_training_rounds: int,
) -> int:
    best_iteration = getattr(search_booster, "best_iteration", None)
    if best_iteration is None:
        return max(int(minimum_training_rounds), int(training_rounds))
    selected_rounds = int(best_iteration) + 1
    return max(
        int(minimum_training_rounds),
        min(selected_rounds, int(training_rounds)),
    )


def fit_meta_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    label_column: str,
    params: dict[str, Any],
    training_rounds: int,
    early_stopping_rounds: int | None = None,
    early_stopping_validation_fraction: float = 0.10,
    minimum_training_rounds: int = 1,
) -> MetaModelArtifact:
    started_at: float = time.perf_counter()
    xgb = load_xgboost_module()
    fitted_rounds = int(training_rounds)
    fit_frame = pd.DataFrame(train_frame.copy())
    eval_frame: pd.DataFrame | None = None
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        fit_frame, eval_frame = _split_early_stopping_frame(
            train_frame,
            validation_fraction=early_stopping_validation_fraction,
        )
    if eval_frame is not None:
        _, _, search_train_matrix = _build_training_matrix_bundle(
            xgb,
            fit_frame,
            feature_columns,
            label_column=label_column,
        )
        _, _, eval_matrix = _build_training_matrix_bundle(
            xgb,
            eval_frame,
            feature_columns,
            label_column=label_column,
        )
        search_booster = xgb.train(
            params=dict(params),
            dtrain=search_train_matrix,
            num_boost_round=training_rounds,
            evals=[(eval_matrix, "validation")],
            early_stopping_rounds=int(early_stopping_rounds),
            verbose_eval=False,
        )
        fitted_rounds = _resolve_selected_training_rounds(
            search_booster,
            training_rounds=training_rounds,
            minimum_training_rounds=minimum_training_rounds,
        )
    _, _, matrix = _build_training_matrix_bundle(
        xgb,
        train_frame,
        feature_columns,
        label_column=label_column,
    )
    booster = xgb.train(
        params=dict(params),
        dtrain=matrix,
        num_boost_round=fitted_rounds,
        verbose_eval=False,
    )
    elapsed: float = time.perf_counter() - started_at
    LOGGER.info(
        "Meta-model fitted: train_rows=%d | features=%d | rounds=%d | early_stopping=%s | elapsed=%.2fs",
        len(train_frame),
        len(feature_columns),
        fitted_rounds,
        "on" if eval_frame is not None else "off",
        elapsed,
    )
    return MetaModelArtifact(
        feature_names=list(feature_columns),
        params=dict(params),
        training_rounds=fitted_rounds,
        fitted_object=booster,
        training_metadata={
            "selected_training_rounds": int(fitted_rounds),
            "max_training_rounds": int(training_rounds),
            "early_stopping_rounds": int(early_stopping_rounds or 0),
            "early_stopping_validation_fraction": float(early_stopping_validation_fraction),
            "early_stopping_used": bool(eval_frame is not None),
            "minimum_training_rounds": int(minimum_training_rounds),
        },
    )


def predict_meta_model(
    artifact: MetaModelArtifact,
    frame: pd.DataFrame,
) -> np.ndarray:
    xgb = load_xgboost_module()
    feature_frame = prepare_xgboost_feature_frame(frame, artifact.feature_names)
    matrix = build_xgboost_dmatrix(xgb, feature_frame, label=None)
    predictions: np.ndarray = np.asarray(artifact.fitted_object.predict(matrix), dtype=np.float64)
    LOGGER.info(
        "Meta-model predicted: rows=%d | mean_probability=%.4f | std=%.4f",
        len(predictions),
        float(np.mean(predictions)),
        float(np.std(predictions)),
    )
    return predictions


def serialize_meta_model_artifact(
    artifact: MetaModelArtifact,
) -> dict[str, object]:
    raw_model = artifact.fitted_object.save_raw()
    payload = base64.b64encode(bytes(raw_model)).decode("ascii")
    return {
        "feature_names": artifact.feature_names,
        "params": artifact.params,
        "training_rounds": int(artifact.training_rounds),
        "training_metadata": dict(artifact.training_metadata),
        "raw_model_b64": payload,
    }


def deserialize_meta_model_artifact(payload: dict[str, object]) -> MetaModelArtifact:
    xgb = load_xgboost_module()
    raw_model_b64 = str(payload["raw_model_b64"])
    booster = xgb.Booster()
    booster.load_model(bytearray(base64.b64decode(raw_model_b64.encode("ascii"))))
    return MetaModelArtifact(
        feature_names=[str(value) for value in payload.get("feature_names", [])],
        params={str(key): value for key, value in dict(payload.get("params", {})).items()},
        training_rounds=int(payload.get("training_rounds", 0)),
        training_metadata={
            str(key): value
            for key, value in dict(payload.get("training_metadata", {})).items()
        },
        fitted_object=booster,
    )
