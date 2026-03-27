from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from core.src.meta_model.data.constants import (
    CROSS_SECTION_OUTLIER_MAD_THRESHOLD,
    DEFAULT_RETURN_COL_CANDIDATES,
    ELEVATED_RETURN_THRESHOLD,
    EXTREME_RETURN_THRESHOLD,
    TICKER_OUTLIER_MAD_THRESHOLD,
    TICKER_OUTLIER_MIN_PERIODS,
    TICKER_OUTLIER_ROLLING_WINDOW,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def _resolve_return_column(df: pd.DataFrame) -> str | None:
    for candidate in DEFAULT_RETURN_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _append_reason(
    reasons: pd.Series,
    mask: pd.Series,
    label: str,
) -> pd.Series:
    updated: pd.Series = reasons.copy()
    empty_mask: pd.Series = mask & (updated == "")
    non_empty_mask: pd.Series = mask & (updated != "")
    updated.loc[empty_mask] = label
    current_values: pd.Series = updated.loc[non_empty_mask].astype(str)
    combined_values: list[str] = [
        f"{value}|{label}"
        for value in current_values.tolist()
    ]
    updated.loc[non_empty_mask] = combined_values
    return updated


def _detect_data_error_flags(
    df: pd.DataFrame,
    return_col: str | None,
) -> tuple[pd.Series, pd.Series]:
    flags: pd.Series = pd.Series(False, index=df.index)
    reasons: pd.Series = pd.Series("", index=df.index, dtype="string")

    if "stock_trading_volume" in df.columns:
        negative_volume_mask: pd.Series = df["stock_trading_volume"] < 0
        flags = flags | negative_volume_mask
        reasons = _append_reason(reasons, negative_volume_mask, "NEGATIVE_VOLUME")

    if return_col is not None:
        finite_mask: pd.Series = pd.Series(
            np.isfinite(df[return_col].to_numpy(dtype=float, copy=False)),
            index=df.index,
        )
        invalid_return_mask: pd.Series = (
            df[return_col].notna() & ~finite_mask
        )
        flags = flags | invalid_return_mask
        reasons = _append_reason(reasons, invalid_return_mask, "INVALID_RETURN")

    return flags.astype(bool), reasons


def _hampel_ticker_flags(series: pd.Series) -> pd.Series:
    median = cast(pd.Series, series.rolling(
        window=TICKER_OUTLIER_ROLLING_WINDOW,
        min_periods=TICKER_OUTLIER_MIN_PERIODS,
    ).median())
    mad = cast(pd.Series, (series - median).abs().rolling(
        window=TICKER_OUTLIER_ROLLING_WINDOW,
        min_periods=TICKER_OUTLIER_MIN_PERIODS,
    ).median())
    robust_sigma: pd.Series = 1.4826 * mad
    distance: pd.Series = (series - median).abs()
    flags: pd.Series = (
        (robust_sigma > 0)
        & (distance > (TICKER_OUTLIER_MAD_THRESHOLD * robust_sigma))
        & series.notna()
    )
    return flags.fillna(False)


def _detect_ticker_extreme_flags(df: pd.DataFrame, return_col: str) -> pd.Series:
    flags = cast(
        pd.Series,
        df.groupby("ticker", sort=False)[return_col].transform(_hampel_ticker_flags),
    )
    return flags.astype(bool)


def _detect_cross_section_extreme_flags(df: pd.DataFrame, return_col: str) -> pd.Series:
    median = cast(
        pd.Series,
        df.groupby("date", sort=False)[return_col].transform("median"),
    )
    deviation: pd.Series = (df[return_col] - median).abs()
    mad = cast(
        pd.Series,
        deviation.groupby(df["date"], sort=False).transform("median"),
    )
    robust_sigma: pd.Series = 1.4826 * mad
    zero_mad_fallback: pd.Series = (
        (robust_sigma == 0)
        & (deviation > 0)
        & (df[return_col].abs() >= EXTREME_RETURN_THRESHOLD)
        & df[return_col].notna()
    )
    flags: pd.Series = (
        (robust_sigma > 0)
        & (deviation > (CROSS_SECTION_OUTLIER_MAD_THRESHOLD * robust_sigma))
        & df[return_col].notna()
    )
    return (flags | zero_mad_fallback).fillna(False).astype(bool)


def _compute_return_based_flags(
    df: pd.DataFrame,
    return_col: str | None,
) -> tuple[pd.Series, pd.Series]:
    ticker_extreme_flag: pd.Series = pd.Series(False, index=df.index)
    cross_section_extreme_flag: pd.Series = pd.Series(False, index=df.index)

    if return_col is None or "ticker" not in df.columns or "date" not in df.columns:
        LOGGER.warning(
            "Could not compute return-based outlier flags (missing return/date/ticker columns).",
        )
        return ticker_extreme_flag, cross_section_extreme_flag

    ticker_extreme_flag = _detect_ticker_extreme_flags(df, return_col)
    cross_section_extreme_flag = _detect_cross_section_extreme_flags(df, return_col)
    return ticker_extreme_flag, cross_section_extreme_flag


def _build_outlier_reasons(
    data_error_reasons: pd.Series,
    ticker_extreme_flag: pd.Series,
    cross_section_extreme_flag: pd.Series,
) -> pd.Series:
    reasons: pd.Series = data_error_reasons.copy()
    reasons = _append_reason(reasons, ticker_extreme_flag, "TICKER_RETURN_EXTREME")
    reasons = _append_reason(
        reasons,
        cross_section_extreme_flag,
        "CROSS_SECTION_RETURN_EXTREME",
    )
    return reasons


def _build_outlier_severity(
    df: pd.DataFrame,
    return_col: str | None,
    combined_flag: pd.Series,
    data_error_flag: pd.Series,
) -> pd.Series:
    severity: pd.Series = pd.Series("normal", index=df.index, dtype="string")
    if return_col is None:
        severity.loc[data_error_flag] = "data_error"
        return severity

    abs_returns = cast(pd.Series, df[return_col]).abs()
    elevated_mask: pd.Series = (
        combined_flag
        & (abs_returns >= ELEVATED_RETURN_THRESHOLD)
        & (abs_returns < EXTREME_RETURN_THRESHOLD)
        & ~data_error_flag
    )
    extreme_mask: pd.Series = (
        combined_flag & (abs_returns >= EXTREME_RETURN_THRESHOLD) & ~data_error_flag
    )
    severity.loc[elevated_mask] = "elevated"
    severity.loc[extreme_mask] = "extreme"
    severity.loc[data_error_flag] = "data_error"
    return severity


def apply_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add outlier and data-quality flags without modifying original feature values."""
    transformed: pd.DataFrame = df.copy()
    return_col: str | None = _resolve_return_column(transformed)

    data_error_flag: pd.Series
    data_error_reasons: pd.Series
    data_error_flag, data_error_reasons = _detect_data_error_flags(
        transformed,
        return_col,
    )

    ticker_extreme_flag: pd.Series
    cross_section_extreme_flag: pd.Series
    ticker_extreme_flag, cross_section_extreme_flag = _compute_return_based_flags(
        transformed,
        return_col,
    )

    combined_flag: pd.Series = data_error_flag | ticker_extreme_flag | cross_section_extreme_flag
    reasons: pd.Series = _build_outlier_reasons(
        data_error_reasons,
        ticker_extreme_flag,
        cross_section_extreme_flag,
    )
    severity: pd.Series = _build_outlier_severity(
        transformed,
        return_col,
        combined_flag,
        data_error_flag,
    )

    transformed["data_error_flag"] = data_error_flag
    transformed["ticker_return_extreme_flag"] = ticker_extreme_flag
    transformed["cross_section_return_extreme_flag"] = cross_section_extreme_flag
    transformed["is_outlier_flag"] = combined_flag
    transformed["outlier_severity"] = severity
    transformed["outlier_reason"] = reasons

    LOGGER.info(
        "Outlier flags: total=%d (%.2f%%), data_error=%d, ticker_extreme=%d, cross_section_extreme=%d",
        int(combined_flag.sum()),
        100.0 * float(combined_flag.mean()) if len(combined_flag) > 0 else 0.0,
        int(data_error_flag.sum()),
        int(ticker_extreme_flag.sum()),
        int(cross_section_extreme_flag.sum()),
    )
    return transformed
