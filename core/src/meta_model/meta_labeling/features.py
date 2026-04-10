from __future__ import annotations

"""Signal refinement and derived feature utilities for meta-labeling."""

import logging

import numpy as np
import pandas as pd

from core.src.meta_model.evaluate.backtest import EXPECTED_RETURN_COLUMN
from core.src.meta_model.model_contract import DATE_COLUMN, PREDICTION_COLUMN

LOGGER: logging.Logger = logging.getLogger(__name__)

PRIMARY_PREDICTION_COLUMN: str = "primary_prediction"
PRIMARY_PREDICTION_RANK_CS_COLUMN: str = "primary_prediction_rank_cs"
PRIMARY_PREDICTION_ZSCORE_CS_COLUMN: str = "primary_prediction_zscore_cs"
PRIMARY_PREDICTION_ABS_COLUMN: str = "primary_prediction_abs"
PRIMARY_PREDICTION_SIGN_COLUMN: str = "primary_prediction_sign"
META_PROBABILITY_COLUMN: str = "meta_probability"
META_CONFIDENCE_COLUMN: str = "meta_confidence"
REFINED_PREDICTION_COLUMN: str = "refined_prediction"
REFINED_EXPECTED_RETURN_COLUMN: str = "refined_expected_return_5d"


def build_primary_context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = pd.DataFrame(frame.copy())
    prediction = pd.to_numeric(enriched[PREDICTION_COLUMN], errors="coerce").fillna(0.0)
    enriched[PRIMARY_PREDICTION_COLUMN] = prediction.to_numpy(dtype=np.float64)
    grouped = enriched.groupby(DATE_COLUMN, sort=False)[PRIMARY_PREDICTION_COLUMN]
    rank = grouped.rank(method="average", pct=True) - 0.5
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    zscore = ((prediction - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    enriched[PRIMARY_PREDICTION_RANK_CS_COLUMN] = rank.to_numpy(dtype=np.float64)
    enriched[PRIMARY_PREDICTION_ZSCORE_CS_COLUMN] = zscore.to_numpy(dtype=np.float64)
    enriched[PRIMARY_PREDICTION_ABS_COLUMN] = prediction.abs().to_numpy(dtype=np.float64)
    enriched[PRIMARY_PREDICTION_SIGN_COLUMN] = np.sign(
        prediction.to_numpy(dtype=np.float64),
    ).astype(np.float64)
    unique_dates: int = int(enriched[DATE_COLUMN].nunique())
    LOGGER.info(
        "Primary context columns built: rows=%d | dates=%d",
        len(enriched),
        unique_dates,
    )
    return enriched


def build_meta_feature_columns(base_feature_columns: list[str]) -> list[str]:
    ordered = [
        *base_feature_columns,
        PRIMARY_PREDICTION_COLUMN,
        PRIMARY_PREDICTION_RANK_CS_COLUMN,
        PRIMARY_PREDICTION_ZSCORE_CS_COLUMN,
        PRIMARY_PREDICTION_ABS_COLUMN,
        PRIMARY_PREDICTION_SIGN_COLUMN,
    ]
    deduped = list(dict.fromkeys(ordered))
    LOGGER.info(
        "Meta-feature columns: base=%d | total=%d (after dedup)",
        len(base_feature_columns),
        len(deduped),
    )
    return deduped


def attach_refined_signal_columns(
    frame: pd.DataFrame,
    *,
    strategy: str = "binary_gate",
    soft_shifted_floor: float = 0.45,
    rank_blend_lambda: float = 0.50,
) -> pd.DataFrame:
    from core.src.meta_model.meta_labeling.refinement import compute_refined_signal

    return compute_refined_signal(
        frame,
        strategy=strategy,
        soft_shifted_floor=soft_shifted_floor,
        rank_blend_lambda=rank_blend_lambda,
    )
