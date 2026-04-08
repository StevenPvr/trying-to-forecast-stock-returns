"""Shared XGBoost DMatrix construction with native categorical support."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.src.meta_model.model_contract import is_structural_categorical_feature_column


def prepare_xgboost_feature_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Align dtypes for XGBoost: structural columns as ``category``, numerics as float32."""
    subset = pd.DataFrame(frame.loc[:, feature_columns].copy())
    for column_name in feature_columns:
        column_series = subset[column_name]
        if is_structural_categorical_feature_column(column_name):
            if isinstance(column_series.dtype, pd.CategoricalDtype):
                continue
            string_series = column_series.astype("string")
            subset[column_name] = string_series.fillna("__missing__").astype("category")
            continue
        if pd.api.types.is_bool_dtype(column_series):
            subset[column_name] = column_series.astype(np.float32)
            continue
        numeric = pd.to_numeric(column_series, errors="coerce").astype(np.float32)
        values = numeric.to_numpy(dtype=np.float64, copy=False)
        finite_mask = np.isfinite(values)
        if not bool(finite_mask.all()):
            numeric = numeric.where(finite_mask, np.nan)
        subset[column_name] = numeric
    return subset


def build_xgboost_dmatrix(
    xgb_module: Any,
    features: pd.DataFrame | np.ndarray,
    label: np.ndarray | None,
    *,
    feature_names: list[str] | None = None,
) -> Any:
    """Build a DMatrix from a pandas frame (with category dtypes) or a dense float matrix."""
    if isinstance(features, pd.DataFrame):
        kwargs: dict[str, Any] = {"data": features}
        if label is not None:
            kwargs["label"] = np.ascontiguousarray(label, dtype=np.float32)
        try:
            return xgb_module.DMatrix(**kwargs, enable_categorical=True)
        except TypeError:
            return xgb_module.DMatrix(**kwargs)
    dense = np.ascontiguousarray(features)
    kw2: dict[str, Any] = {
        "data": dense,
        "feature_names": feature_names,
    }
    if label is not None:
        kw2["label"] = np.ascontiguousarray(label, dtype=np.float32)
    return xgb_module.DMatrix(**kw2)
