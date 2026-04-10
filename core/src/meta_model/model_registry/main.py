from __future__ import annotations

"""Unified model registry: fit and predict for Ridge, ElasticNet, XGBoost, LightGBM, factor composite."""

import logging
from dataclasses import dataclass, field
import importlib.util
from typing import Any

import numpy as np
import pandas as pd

LOGGER: logging.Logger = logging.getLogger(__name__)

from core.src.meta_model.model_contract import DATE_COLUMN, MODEL_TARGET_COLUMN
from core.src.meta_model.optimize_parameters.search_space import load_xgboost_module
from core.src.meta_model.xgboost_dmatrix import build_xgboost_dmatrix, prepare_xgboost_feature_frame


@dataclass(frozen=True)
class ModelSpec:
    """Specification for training a single model (name, hyper-params, target)."""

    model_name: str
    params: dict[str, Any] = field(default_factory=dict)
    target_column: str = MODEL_TARGET_COLUMN
    training_rounds: int | None = None


@dataclass(frozen=True)
class ModelArtifact:
    """Trained model artifact containing the fitted object and metadata."""

    model_name: str
    feature_names: list[str]
    training_metadata: dict[str, Any]
    fitted_object: Any


def _lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def build_default_model_specs(
    *,
    xgboost_params: dict[str, Any],
    xgboost_training_rounds: int,
) -> list[ModelSpec]:
    """Build the canonical list of model specs for the evaluation pipeline."""
    return [
        ModelSpec(model_name="ridge", params={"alpha": 1.0}),
        ModelSpec(
            model_name="elastic_net",
            params={"alpha": 0.05, "l1_ratio": 0.25, "max_iter": 300, "tol": 1e-5},
        ),
        ModelSpec(model_name="factor_composite"),
        ModelSpec(
            model_name="xgboost",
            params=dict(xgboost_params),
            training_rounds=xgboost_training_rounds,
        ),
        *(
            [
                ModelSpec(
                    model_name="lightgbm",
                    params={
                        "objective": "regression",
                        "learning_rate": 0.03,
                        "num_leaves": 31,
                        "feature_fraction": 0.70,
                        "bagging_fraction": 0.80,
                        "bagging_freq": 1,
                        "min_data_in_leaf": 64,
                        "verbosity": -1,
                    },
                    training_rounds=min(xgboost_training_rounds, 300),
                ),
            ]
            if _lightgbm_available()
            else []
        ),
    ]


def _fit_ridge_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    alpha: float,
    target_column: str,
) -> ModelArtifact:
    x = train_frame.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=False)
    y = train_frame[target_column].to_numpy(dtype=np.float64, copy=False)
    feature_mean = np.nanmean(x, axis=0)
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0)
    x_imputed = np.where(np.isfinite(x), x, feature_mean)
    feature_std = x_imputed.std(axis=0, ddof=0)
    safe_std = np.where(feature_std <= 1e-12, 1.0, feature_std)
    x_standardized = (x_imputed - feature_mean) / safe_std
    target_mean = float(y.mean())
    y_centered = y - target_mean
    gram = x_standardized.T @ x_standardized
    ridge_penalty = alpha * np.eye(len(feature_columns), dtype=np.float64)
    coefficients = np.linalg.solve(gram + ridge_penalty, x_standardized.T @ y_centered)
    fitted_object = {
        "feature_mean": feature_mean,
        "feature_std": safe_std,
        "coefficients": coefficients,
        "target_mean": target_mean,
    }
    return ModelArtifact(
        model_name="ridge",
        feature_names=list(feature_columns),
        training_metadata={"alpha": alpha, "target_column": target_column},
        fitted_object=fitted_object,
    )


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _fit_elastic_net_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    target_column: str,
) -> ModelArtifact:
    x = train_frame.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=False)
    y = train_frame[target_column].to_numpy(dtype=np.float64, copy=False)
    feature_mean = np.nanmean(x, axis=0)
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0)
    x_imputed = np.where(np.isfinite(x), x, feature_mean)
    feature_std = x_imputed.std(axis=0, ddof=0)
    safe_std = np.where(feature_std <= 1e-12, 1.0, feature_std)
    x_standardized = (x_imputed - feature_mean) / safe_std
    target_mean = float(y.mean())
    y_centered = y - target_mean
    coefficients = np.zeros(len(feature_columns), dtype=np.float64)
    squared_norm = np.mean(np.square(x_standardized), axis=0)
    for _ in range(max_iter):
        max_update = 0.0
        for column_index in range(len(feature_columns)):
            residual = (
                y_centered
                - x_standardized @ coefficients
                + x_standardized[:, column_index] * coefficients[column_index]
            )
            rho = float(np.mean(x_standardized[:, column_index] * residual))
            denominator = squared_norm[column_index] + alpha * (1.0 - l1_ratio)
            updated_value = _soft_threshold(rho, alpha * l1_ratio) / max(denominator, 1e-12)
            max_update = max(max_update, abs(updated_value - coefficients[column_index]))
            coefficients[column_index] = updated_value
        if max_update <= tol:
            break
    fitted_object = {
        "feature_mean": feature_mean,
        "feature_std": safe_std,
        "coefficients": coefficients,
        "target_mean": target_mean,
    }
    return ModelArtifact(
        model_name="elastic_net",
        feature_names=list(feature_columns),
        training_metadata={
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "max_iter": max_iter,
            "tol": tol,
            "target_column": target_column,
        },
        fitted_object=fitted_object,
    )


def _select_factor_composite_columns(feature_columns: list[str]) -> dict[str, list[str]]:
    buckets = {
        "momentum": [
            column for column in feature_columns
            if "momentum" in column or "mom_" in column
        ],
        "reversal": [
            column for column in feature_columns
            if "reversal" in column
        ],
        "volatility": [
            column for column in feature_columns
            if "volatility" in column or column.startswith("vol_")
        ],
        "size": [
            column for column in feature_columns
            if "market_cap" in column or "enterprise_value" in column
        ],
        "liquidity": [
            column for column in feature_columns
            if "trading_volume" in column or "dollar_volume" in column
        ],
    }
    if any(buckets.values()):
        return buckets
    raise ValueError(
        "factor_composite requires at least one semantic bucket among momentum, reversal, volatility, size, or liquidity.",
    )


def _fit_factor_composite_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
) -> ModelArtifact:
    selected_columns = _select_factor_composite_columns(feature_columns)
    feature_frame = train_frame.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=False)
    feature_mean = np.nanmean(feature_frame, axis=0)
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0)
    return ModelArtifact(
        model_name="factor_composite",
        feature_names=list(feature_columns),
        training_metadata={"selected_columns": selected_columns},
        fitted_object={"selected_columns": selected_columns, "feature_mean": feature_mean},
    )


def _fit_xgboost_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    params: dict[str, Any],
    target_column: str,
    training_rounds: int,
) -> ModelArtifact:
    xgb = load_xgboost_module()
    feature_frame = prepare_xgboost_feature_frame(train_frame, feature_columns)
    label_values = pd.to_numeric(
        train_frame[target_column],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    finite_label_mask = np.isfinite(label_values)
    if not bool(finite_label_mask.all()):
        feature_frame = pd.DataFrame(feature_frame.loc[finite_label_mask].copy())
        label_values = label_values[finite_label_mask]
    if label_values.size == 0:
        raise ValueError("xgboost training frame has no finite target values.")
    matrix = build_xgboost_dmatrix(
        xgb,
        feature_frame,
        label_values.astype(np.float32, copy=False),
    )
    booster = xgb.train(
        params=dict(params),
        dtrain=matrix,
        num_boost_round=training_rounds,
        verbose_eval=False,
    )
    return ModelArtifact(
        model_name="xgboost",
        feature_names=list(feature_columns),
        training_metadata={
            "params": dict(params),
            "target_column": target_column,
            "training_rounds": training_rounds,
        },
        fitted_object=booster,
    )


def _fit_lightgbm_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    params: dict[str, Any],
    target_column: str,
    training_rounds: int,
) -> ModelArtifact:
    if not _lightgbm_available():
        raise ValueError("lightgbm is not available in the current environment.")
    import lightgbm as lgb  # noqa: F811 -- conditional import guarded by availability check

    dataset = lgb.Dataset(
        train_frame.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=False),
        label=train_frame[target_column].to_numpy(dtype=np.float64, copy=False),
        feature_name=feature_columns,
        free_raw_data=True,
    )
    booster = lgb.train(
        params=dict(params),
        train_set=dataset,
        num_boost_round=training_rounds,
    )
    return ModelArtifact(
        model_name="lightgbm",
        feature_names=list(feature_columns),
        training_metadata={
            "params": dict(params),
            "target_column": target_column,
            "training_rounds": training_rounds,
        },
        fitted_object=booster,
    )


def fit_model(
    spec: ModelSpec,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
) -> ModelArtifact:
    """Dispatch training to the appropriate model backend and return the artifact."""
    if spec.model_name == "ridge":
        return _fit_ridge_model(
            train_frame,
            feature_columns,
            alpha=float(spec.params.get("alpha", 1.0)),
            target_column=spec.target_column,
        )
    if spec.model_name == "elastic_net":
        return _fit_elastic_net_model(
            train_frame,
            feature_columns,
            alpha=float(spec.params.get("alpha", 0.05)),
            l1_ratio=float(spec.params.get("l1_ratio", 0.25)),
            max_iter=int(spec.params.get("max_iter", 300)),
            tol=float(spec.params.get("tol", 1e-5)),
            target_column=spec.target_column,
        )
    if spec.model_name == "factor_composite":
        return _fit_factor_composite_model(train_frame, feature_columns)
    if spec.model_name == "xgboost":
        if spec.training_rounds is None:
            raise ValueError("xgboost ModelSpec requires training_rounds.")
        return _fit_xgboost_model(
            train_frame,
            feature_columns,
            params=spec.params,
            target_column=spec.target_column,
            training_rounds=spec.training_rounds,
        )
    if spec.model_name == "lightgbm":
        if spec.training_rounds is None:
            raise ValueError("lightgbm ModelSpec requires training_rounds.")
        return _fit_lightgbm_model(
            train_frame,
            feature_columns,
            params=spec.params,
            target_column=spec.target_column,
            training_rounds=spec.training_rounds,
        )
    raise ValueError(f"Unsupported model_name: {spec.model_name}")


def _predict_ridge_model(
    artifact: ModelArtifact,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    model = dict(artifact.fitted_object)
    x = frame.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=False)
    x_imputed = np.where(np.isfinite(x), x, model["feature_mean"])
    x_standardized = (x_imputed - model["feature_mean"]) / model["feature_std"]
    return x_standardized @ model["coefficients"] + float(model["target_mean"])


def _cross_sectional_zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(np.zeros(len(values), dtype=np.float64), index=values.index)
    return (values - float(values.mean())) / std


def _predict_factor_composite_model(
    artifact: ModelArtifact,
    frame: pd.DataFrame,
) -> np.ndarray:
    fitted_object = dict(artifact.fitted_object)
    selected_columns = fitted_object["selected_columns"]
    imputed_frame = frame.loc[:, artifact.feature_names].copy()
    feature_mean = np.asarray(fitted_object["feature_mean"], dtype=np.float64)
    for column_index, column_name in enumerate(artifact.feature_names):
        imputed_frame[column_name] = imputed_frame[column_name].fillna(float(feature_mean[column_index]))
    scored = frame.loc[:, [DATE_COLUMN]].copy()
    scored["prediction"] = 0.0
    sign_map = {
        "momentum": 1.0,
        "reversal": 1.0,
        "volatility": -1.0,
        "size": -1.0,
        "liquidity": 1.0,
    }
    for bucket_name, columns in selected_columns.items():
        if not columns:
            continue
        bucket_values = imputed_frame.loc[:, columns].mean(axis=1)
        scored["prediction"] += (
            bucket_values.groupby(frame[DATE_COLUMN], sort=False).transform(_cross_sectional_zscore)
            * sign_map[bucket_name]
        )
    return scored["prediction"].to_numpy(dtype=np.float64, copy=False)


def _predict_xgboost_model(
    artifact: ModelArtifact,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    xgb = load_xgboost_module()
    feature_frame = prepare_xgboost_feature_frame(frame, feature_columns)
    matrix = build_xgboost_dmatrix(xgb, feature_frame, label=None)
    return np.asarray(artifact.fitted_object.predict(matrix), dtype=np.float64)


def predict_model(
    artifact: ModelArtifact,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Generate predictions using the trained *artifact* on *frame*."""
    if artifact.model_name == "ridge":
        return _predict_ridge_model(artifact, frame, feature_columns)
    if artifact.model_name == "elastic_net":
        return _predict_ridge_model(artifact, frame, feature_columns)
    if artifact.model_name == "factor_composite":
        return _predict_factor_composite_model(artifact, frame)
    if artifact.model_name == "xgboost":
        return _predict_xgboost_model(artifact, frame, feature_columns)
    if artifact.model_name == "lightgbm":
        return np.asarray(
            artifact.fitted_object.predict(frame.loc[:, feature_columns].to_numpy(dtype=np.float64)),
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported model_name: {artifact.model_name}")


__all__ = [
    "ModelArtifact",
    "ModelSpec",
    "build_default_model_specs",
    "fit_model",
    "predict_model",
]
