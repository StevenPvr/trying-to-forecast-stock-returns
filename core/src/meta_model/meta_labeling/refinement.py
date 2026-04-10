from __future__ import annotations

"""Pluggable refinement strategies for meta-labeling signal transformation.

Each strategy transforms (primary_prediction, meta_probability) into
(meta_confidence, refined_prediction, refined_expected_return).
"""

import logging

import numpy as np
import pandas as pd

from core.src.meta_model.evaluate.backtest import EXPECTED_RETURN_COLUMN
from core.src.meta_model.meta_labeling.features import (
    META_CONFIDENCE_COLUMN,
    META_PROBABILITY_COLUMN,
    PRIMARY_PREDICTION_COLUMN,
    REFINED_EXPECTED_RETURN_COLUMN,
    REFINED_PREDICTION_COLUMN,
)
from core.src.meta_model.model_contract import DATE_COLUMN, PREDICTION_COLUMN

LOGGER: logging.Logger = logging.getLogger(__name__)

KNOWN_STRATEGIES: tuple[str, ...] = (
    "binary_gate",
    "candidate_only",
    "soft_shifted",
    "rank_blend",
    "no_meta",
)
DEFAULT_REFINEMENT_STRATEGY: str = "binary_gate"
DEFAULT_SOFT_SHIFTED_FLOOR: float = 0.45
DEFAULT_RANK_BLEND_LAMBDA: float = 0.50
_BINARY_GATE_THRESHOLD: float = 0.5


def _cross_sectional_zscore(series: pd.Series, dates: pd.Series) -> np.ndarray:
    """Cross-sectional zscore per trading date."""
    grouped = series.groupby(dates)
    mean = grouped.transform("mean")
    std = grouped.transform("std").clip(lower=1e-8)
    zscore = ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return zscore.to_numpy(dtype=np.float64)


def _apply_binary_gate(
    primary: np.ndarray,
    meta_prob: np.ndarray,
    expected: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    confidence = np.where(meta_prob >= _BINARY_GATE_THRESHOLD, meta_prob, 0.0)
    refined = primary * confidence
    refined_exp = expected * confidence if expected is not None else None
    return confidence, refined, refined_exp


def _apply_candidate_only(
    primary: np.ndarray,
    meta_prob: np.ndarray,
    expected: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    confidence = np.where(primary > 0.0, 1.0, 0.0)
    refined = primary * confidence
    refined_exp = expected * confidence if expected is not None else None
    return confidence, refined, refined_exp


def _apply_soft_shifted(
    primary: np.ndarray,
    meta_prob: np.ndarray,
    expected: np.ndarray | None,
    *,
    floor: float = DEFAULT_SOFT_SHIFTED_FLOOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    denominator: float = max(1.0 - floor, 1e-8)
    confidence = np.clip((meta_prob - floor) / denominator, 0.0, 1.0)
    refined = primary * confidence
    refined_exp = expected * confidence if expected is not None else None
    return confidence, refined, refined_exp


def _apply_rank_blend(
    primary_series: pd.Series,
    meta_prob_series: pd.Series,
    dates: pd.Series,
    *,
    blend_lambda: float = DEFAULT_RANK_BLEND_LAMBDA,
) -> np.ndarray:
    z_primary = _cross_sectional_zscore(primary_series, dates)
    z_meta = _cross_sectional_zscore(meta_prob_series, dates)
    return z_primary + blend_lambda * z_meta


def _apply_no_meta(
    primary: np.ndarray,
    meta_prob: np.ndarray,
    expected: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    confidence = np.ones_like(primary)
    return confidence, primary.copy(), expected.copy() if expected is not None else None


def compute_refined_signal(
    frame: pd.DataFrame,
    *,
    strategy: str = DEFAULT_REFINEMENT_STRATEGY,
    soft_shifted_floor: float = DEFAULT_SOFT_SHIFTED_FLOOR,
    rank_blend_lambda: float = DEFAULT_RANK_BLEND_LAMBDA,
) -> pd.DataFrame:
    """Apply the named refinement strategy and attach signal columns."""
    if strategy not in KNOWN_STRATEGIES:
        raise ValueError(
            f"Unknown refinement strategy {strategy!r}. "
            f"Must be one of {KNOWN_STRATEGIES}.",
        )
    enriched = pd.DataFrame(frame.copy())
    primary_col = (
        PRIMARY_PREDICTION_COLUMN
        if PRIMARY_PREDICTION_COLUMN in enriched.columns
        else PREDICTION_COLUMN
    )
    primary_prediction = pd.to_numeric(
        enriched[primary_col], errors="coerce",
    ).fillna(0.0)
    meta_probability = pd.to_numeric(
        enriched[META_PROBABILITY_COLUMN], errors="coerce",
    ).fillna(0.0)
    primary_arr = primary_prediction.to_numpy(dtype=np.float64)
    meta_arr = meta_probability.to_numpy(dtype=np.float64)
    has_expected = EXPECTED_RETURN_COLUMN in enriched.columns
    expected_arr: np.ndarray | None = (
        pd.to_numeric(enriched[EXPECTED_RETURN_COLUMN], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        if has_expected
        else None
    )

    if strategy == "binary_gate":
        confidence, refined, refined_exp = _apply_binary_gate(
            primary_arr, meta_arr, expected_arr,
        )
    elif strategy == "candidate_only":
        confidence, refined, refined_exp = _apply_candidate_only(
            primary_arr, meta_arr, expected_arr,
        )
    elif strategy == "soft_shifted":
        confidence, refined, refined_exp = _apply_soft_shifted(
            primary_arr, meta_arr, expected_arr, floor=soft_shifted_floor,
        )
    elif strategy == "rank_blend":
        dates = enriched[DATE_COLUMN]
        blended = _apply_rank_blend(
            primary_prediction, meta_probability, dates,
            blend_lambda=rank_blend_lambda,
        )
        confidence = np.ones(len(enriched), dtype=np.float64)
        refined = blended
        refined_exp = expected_arr
    elif strategy == "no_meta":
        confidence, refined, refined_exp = _apply_no_meta(
            primary_arr, meta_arr, expected_arr,
        )

    enriched[META_CONFIDENCE_COLUMN] = confidence
    enriched[REFINED_PREDICTION_COLUMN] = refined
    enriched[REFINED_EXPECTED_RETURN_COLUMN] = (
        refined_exp if refined_exp is not None else 0.0
    )
    mean_confidence: float = float(np.mean(confidence))
    mean_refined: float = float(np.mean(np.abs(refined)))
    LOGGER.info(
        "Refined signal [%s]: rows=%d | mean_confidence=%.4f | mean_abs_refined=%.6f",
        strategy,
        len(enriched),
        mean_confidence,
        mean_refined,
    )
    return enriched
