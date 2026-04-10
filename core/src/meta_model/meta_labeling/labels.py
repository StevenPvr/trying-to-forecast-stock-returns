from __future__ import annotations

"""Meta-label construction utilities."""

import logging

import numpy as np
import pandas as pd

from core.src.meta_model.meta_labeling.features import PRIMARY_PREDICTION_COLUMN
from core.src.meta_model.model_contract import WEEK_HOLD_NET_RETURN_COLUMN

LOGGER: logging.Logger = logging.getLogger(__name__)

META_LABEL_COLUMN: str = "meta_label"
META_CANDIDATE_COLUMN: str = "meta_candidate"


def attach_meta_labels(
    frame: pd.DataFrame,
    *,
    primary_prediction_threshold: float = 0.0,
    minimum_target_net_return: float = 0.001,
) -> pd.DataFrame:
    labeled = pd.DataFrame(frame.copy())
    primary_prediction = pd.to_numeric(
        labeled[PRIMARY_PREDICTION_COLUMN],
        errors="coerce",
    ).fillna(0.0)
    realized_net_return = pd.to_numeric(
        labeled[WEEK_HOLD_NET_RETURN_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    candidate_mask = primary_prediction.to_numpy(dtype=np.float64) > float(primary_prediction_threshold)
    label_mask = candidate_mask & (realized_net_return > float(minimum_target_net_return))
    labeled[META_CANDIDATE_COLUMN] = candidate_mask.astype(np.int64)
    labeled[META_LABEL_COLUMN] = label_mask.astype(np.int64)
    candidate_count: int = int(np.sum(candidate_mask))
    positive_count: int = int(np.sum(label_mask))
    total_count: int = len(labeled)
    negative_count: int = candidate_count - positive_count
    LOGGER.info(
        "Meta-labels attached: rows=%d | candidates=%d (%.1f%%) | positive=%d (%.1f%% of candidates) | negative=%d (%.1f%% of candidates)",
        total_count,
        candidate_count,
        100.0 * candidate_count / total_count if total_count > 0 else 0.0,
        positive_count,
        100.0 * positive_count / candidate_count if candidate_count > 0 else 0.0,
        negative_count,
        100.0 * negative_count / candidate_count if candidate_count > 0 else 0.0,
    )
    return labeled


def select_meta_candidate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    candidate_mask = pd.to_numeric(
        frame[META_CANDIDATE_COLUMN],
        errors="coerce",
    ).fillna(0.0) > 0.5
    return pd.DataFrame(frame.loc[candidate_mask].copy())
