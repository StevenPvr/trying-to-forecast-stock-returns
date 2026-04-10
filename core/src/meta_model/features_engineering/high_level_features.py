from __future__ import annotations

"""High-level composite features: sector-relative, signal interactions, and alpha composites."""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from core.src.meta_model.broker_xtb.specs import (
    BrokerSpecProvider,
    XtbInstrumentSpec,
    build_default_spec_provider,
)
from core.src.meta_model.data.data_reference.reference_pipeline import ensure_earnings_history_output
from core.src.meta_model.data.paths import REFERENCE_EARNINGS_HISTORY_CSV
from core.src.meta_model.features_engineering.config import TRADING_DAYS_PER_YEAR

LOGGER: logging.Logger = logging.getLogger(__name__)

REQUIRED_PRICE_COLUMNS: tuple[str, ...] = (
    "stock_open_price",
    "stock_high_price",
    "stock_low_price",
    "stock_close_price",
)
OPTIONAL_CONTEXT_COLUMNS: tuple[str, ...] = (
    "company_sector",
    "company_industry",
    "stock_trading_volume",
)
EARNINGS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "announcement_date",
    "announcement_session",
    "fiscal_year",
    "fiscal_quarter",
)
TEMP_PREFIX: str = "__hl_"
FAR_EARNINGS_DISTANCE: float = 252.0
_SUPPORTED_EARNINGS_SESSIONS: frozenset[str] = frozenset({"before_open", "after_close", "unknown"})
RSI_OVERBOUGHT_THRESHOLD: float = 70.0
RSI_OVERSOLD_THRESHOLD: float = 30.0
VOLATILITY_TENSION_THRESHOLD: float = 1.0
OPEN_STRETCH_ATR_THRESHOLD: float = 1.0
HIGH_RELATIVE_COST_THRESHOLD: float = 0.25


def add_high_level_features(
    data: pd.DataFrame,
    *,
    earnings_path: Path = REFERENCE_EARNINGS_HISTORY_CSV,
    spec_provider: BrokerSpecProvider | None = None,
) -> pd.DataFrame:
    resolved_earnings_path = (
        ensure_earnings_history_output(earnings_output_csv=REFERENCE_EARNINGS_HISTORY_CSV)
        if earnings_path == REFERENCE_EARNINGS_HISTORY_CSV
        else earnings_path
    )
    prepared = _prepare_input_dataset(data)
    enriched = _add_price_primitives(prepared)
    enriched = _add_xtb_features(enriched, spec_provider or build_default_spec_provider())
    enriched = _add_sector_features(enriched)
    enriched = _add_open_features(enriched)
    enriched = _add_regime_features(enriched)
    enriched = _add_earnings_features(enriched, resolved_earnings_path)
    enriched = _add_signal_intersection_features(enriched)
    return _finalize_dataset(enriched)


def _prepare_input_dataset(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column_name for column_name in REQUIRED_PRICE_COLUMNS if column_name not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required high-level feature columns: {', '.join(missing_columns)}")
    prepared = pd.DataFrame(data.copy())
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared["ticker"] = prepared["ticker"].astype(str)
    for column_name in REQUIRED_PRICE_COLUMNS:
        prepared[column_name] = pd.to_numeric(prepared[column_name], errors="coerce")
    for column_name in OPTIONAL_CONTEXT_COLUMNS:
        if column_name not in prepared.columns:
            prepared[column_name] = np.nan
    return prepared.sort_values(["ticker", "date"]).reset_index(drop=True)


def _add_price_primitives(data: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(data.copy())
    grouped = prepared.groupby("ticker", sort=False)
    prepared[f"{TEMP_PREFIX}prev_close"] = grouped["stock_close_price"].shift(1)
    prepared[f"{TEMP_PREFIX}prev_high"] = grouped["stock_high_price"].shift(1)
    prepared[f"{TEMP_PREFIX}prev_low"] = grouped["stock_low_price"].shift(1)
    prepared[f"{TEMP_PREFIX}gap_return"] = _safe_log_ratio(
        prepared["stock_open_price"],
        prepared[f"{TEMP_PREFIX}prev_close"],
    )
    prepared[f"{TEMP_PREFIX}intraday_return"] = _safe_log_ratio(
        prepared["stock_close_price"],
        prepared["stock_open_price"],
    )
    prepared[f"{TEMP_PREFIX}true_range_pct"] = _compute_true_range_pct(prepared)
    prepared[f"{TEMP_PREFIX}atr_21d"] = grouped[f"{TEMP_PREFIX}true_range_pct"].transform(
        lambda series: series.rolling(21, min_periods=21).mean(),
    )
    if "quant_realized_vol_21d" not in prepared.columns:
        close_log_return: pd.Series = _safe_log_ratio(
            prepared["stock_close_price"],
            prepared[f"{TEMP_PREFIX}prev_close"],
        )
        prepared[f"{TEMP_PREFIX}close_log_return"] = close_log_return
        clr_grouped = grouped[f"{TEMP_PREFIX}close_log_return"]
        prepared[f"{TEMP_PREFIX}realized_vol_5d"] = clr_grouped.transform(
            lambda s: s.rolling(5, min_periods=5).std(),
        )
        prepared[f"{TEMP_PREFIX}realized_vol_21d"] = clr_grouped.transform(
            lambda s: s.rolling(21, min_periods=21).std(),
        )
        prepared[f"{TEMP_PREFIX}realized_vol_63d"] = clr_grouped.transform(
            lambda s: s.rolling(63, min_periods=63).std(),
        )
    prepared[f"{TEMP_PREFIX}gap_vol_21d"] = grouped[f"{TEMP_PREFIX}gap_return"].transform(
        lambda series: series.rolling(21, min_periods=21).std(),
    )
    prepared[f"{TEMP_PREFIX}gap_vol_63d"] = grouped[f"{TEMP_PREFIX}gap_return"].transform(
        lambda series: series.rolling(63, min_periods=63).std(),
    )
    prepared[f"{TEMP_PREFIX}intraday_vol_21d"] = grouped[f"{TEMP_PREFIX}intraday_return"].transform(
        lambda series: series.rolling(21, min_periods=21).std(),
    )
    prepared[f"{TEMP_PREFIX}intraday_vol_63d"] = grouped[f"{TEMP_PREFIX}intraday_return"].transform(
        lambda series: series.rolling(63, min_periods=63).std(),
    )
    prepared[f"{TEMP_PREFIX}rsi_14"] = grouped["stock_close_price"].transform(_compute_rsi_series)
    return prepared


def _add_xtb_features(data: pd.DataFrame, spec_provider: BrokerSpecProvider) -> pd.DataFrame:
    prepared = pd.DataFrame(data.copy())
    resolved_specs = [
        spec_provider.resolve(ticker, pd.Timestamp(trade_date))
        for ticker, trade_date in zip(
            prepared["ticker"].astype(str),
            pd.to_datetime(prepared["date"]),
            strict=True,
        )
    ]
    spread_bps = np.array([spec.spread_bps for spec in resolved_specs], dtype=np.float64)
    slippage_bps = np.array([spec.slippage_bps for spec in resolved_specs], dtype=np.float64)
    long_swap = np.array([spec.long_swap_bps_daily for spec in resolved_specs], dtype=np.float64)
    short_swap = np.array([spec.short_swap_bps_daily for spec in resolved_specs], dtype=np.float64)
    prepared["xtb_spread_bps"] = spread_bps
    prepared["xtb_slippage_bps"] = slippage_bps
    prepared["xtb_long_swap_bps_daily"] = long_swap
    prepared["xtb_short_swap_bps_daily"] = short_swap
    prepared["xtb_swap_asymmetry"] = short_swap - long_swap
    n_specs: int = len(resolved_specs)
    zero_rates = np.zeros(n_specs, dtype=np.float64)
    prepared["xtb_expected_intraday_cost_rate"] = zero_rates
    prepared["xtb_expected_overnight_cost_rate"] = zero_rates.copy()
    spread_rate = spread_bps / 10_000.0
    if "quant_realized_vol_21d" in prepared.columns:
        vol_21d: pd.Series = _resolve_numeric_feature(prepared, "quant_realized_vol_21d") / np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        vol_21d = _resolve_numeric_feature(prepared, f"{TEMP_PREFIX}realized_vol_21d")
    prepared["xtb_spread_to_realized_vol_21d"] = _safe_divide(spread_rate, vol_21d)
    prepared["xtb_spread_to_gap_abs"] = _safe_divide(
        spread_rate,
        prepared[f"{TEMP_PREFIX}gap_return"].abs(),
    )
    return prepared


def _add_sector_features(data: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(data.copy())
    sector_key = _build_sector_key(prepared)
    group_keys = [pd.to_datetime(prepared["date"]), sector_key]
    prepared["sector_relative_gap_return"] = (
        prepared[f"{TEMP_PREFIX}gap_return"]
        - prepared.groupby(group_keys)[f"{TEMP_PREFIX}gap_return"].transform("mean")
    )
    prepared["sector_relative_intraday_return"] = (
        prepared[f"{TEMP_PREFIX}intraday_return"]
        - prepared.groupby(group_keys)[f"{TEMP_PREFIX}intraday_return"].transform("mean")
    )
    prepared["sector_relative_rsi"] = (
        prepared[f"{TEMP_PREFIX}rsi_14"]
        - prepared.groupby(group_keys)[f"{TEMP_PREFIX}rsi_14"].transform("mean")
    )
    prepared["sector_rsi_rank"] = prepared.groupby(group_keys)[f"{TEMP_PREFIX}rsi_14"].rank(
        method="average",
        pct=True,
    )
    prepared["sector_gap_rank"] = prepared.groupby(group_keys)[f"{TEMP_PREFIX}gap_return"].rank(
        method="average",
        pct=True,
    )
    return prepared


def _add_open_features(data: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(data.copy())
    open_col = prepared["stock_open_price"]
    prev_high = prepared[f"{TEMP_PREFIX}prev_high"]
    prev_low = prepared[f"{TEMP_PREFIX}prev_low"]
    prev_close = prepared[f"{TEMP_PREFIX}prev_close"]
    atr_21d = prepared[f"{TEMP_PREFIX}atr_21d"]
    prepared["open_above_prev_high_flag"] = (open_col > prev_high).astype(np.int8)
    prepared["open_below_prev_low_flag"] = (open_col < prev_low).astype(np.int8)
    prepared["open_in_prev_range_flag"] = ((open_col >= prev_low) & (open_col <= prev_high)).astype(np.int8)
    prepared["open_distance_to_prev_close_over_atr_21d"] = _safe_divide(open_col - prev_close, atr_21d)
    prepared["open_distance_to_prev_high_over_atr_21d"] = _safe_divide(open_col - prev_high, atr_21d)
    prepared["open_distance_to_prev_low_over_atr_21d"] = _safe_divide(open_col - prev_low, atr_21d)
    return prepared


def _add_regime_features(data: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(data.copy())
    has_quant_vol: bool = "quant_realized_vol_21d" in prepared.columns
    vol_5d: pd.Series = (
        _resolve_numeric_feature(prepared, "quant_realized_vol_5d")
        if has_quant_vol
        else _resolve_numeric_feature(prepared, f"{TEMP_PREFIX}realized_vol_5d")
    )
    vol_21d: pd.Series = (
        _resolve_numeric_feature(prepared, "quant_realized_vol_21d")
        if has_quant_vol
        else _resolve_numeric_feature(prepared, f"{TEMP_PREFIX}realized_vol_21d")
    )
    vol_63d: pd.Series = (
        _resolve_numeric_feature(prepared, "quant_realized_vol_63d")
        if has_quant_vol
        else _resolve_numeric_feature(prepared, f"{TEMP_PREFIX}realized_vol_63d")
    )
    prepared["quant_realized_vol_ratio_5d_21d"] = _safe_divide(vol_5d, vol_21d)
    prepared["quant_realized_vol_ratio_21d_63d"] = _safe_divide(vol_21d, vol_63d)
    prepared["quant_gap_vol_ratio_21d_63d"] = _safe_divide(
        prepared[f"{TEMP_PREFIX}gap_vol_21d"],
        prepared[f"{TEMP_PREFIX}gap_vol_63d"],
    )
    prepared["quant_intraday_vol_ratio_21d_63d"] = _safe_divide(
        prepared[f"{TEMP_PREFIX}intraday_vol_21d"],
        prepared[f"{TEMP_PREFIX}intraday_vol_63d"],
    )
    grouped = prepared.groupby("ticker", sort=False)[f"{TEMP_PREFIX}true_range_pct"]
    rolling_mean = grouped.transform(lambda series: series.rolling(21, min_periods=21).mean())
    rolling_std = grouped.transform(lambda series: series.rolling(21, min_periods=21).std())
    prepared["quant_true_range_zscore_21d"] = _safe_divide(
        prepared[f"{TEMP_PREFIX}true_range_pct"] - rolling_mean,
        rolling_std,
    )
    return prepared


def _add_earnings_features(data: pd.DataFrame, earnings_path: Path) -> pd.DataFrame:
    prepared = pd.DataFrame(data.copy())
    session_map = _build_session_index_map(cast_series(prepared["date"]))
    earnings = _load_earnings_reference(earnings_path, cast_series(session_map["date"]))
    if earnings.empty:
        prepared["earnings_days_to_next"] = FAR_EARNINGS_DISTANCE
        prepared["earnings_days_since_last"] = FAR_EARNINGS_DISTANCE
        prepared["earnings_is_week"] = np.int8(0)
        prepared["earnings_proximity_bucket"] = np.int64(3)
        return prepared
    prepared = prepared.merge(session_map, on="date", how="left", sort=False)
    next_days, prev_days = _compute_earnings_distances(prepared, earnings)
    prepared["earnings_days_to_next"] = next_days.fillna(FAR_EARNINGS_DISTANCE)
    prepared["earnings_days_since_last"] = prev_days.fillna(FAR_EARNINGS_DISTANCE)
    prepared["earnings_is_week"] = (prepared["earnings_days_to_next"] <= 5.0).astype(np.int8)
    prepared["earnings_proximity_bucket"] = np.select(
        [
            prepared["earnings_days_to_next"] <= 1.0,
            prepared["earnings_days_to_next"] <= 5.0,
            prepared["earnings_days_to_next"] <= 10.0,
        ],
        [0, 1, 2],
        default=3,
    ).astype(np.int64)
    return prepared


def _add_signal_intersection_features(data: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(data.copy())
    rsi = _resolve_numeric_feature(prepared, "ta_momentum_rsi")
    macd = _resolve_numeric_feature(prepared, "ta_trend_macd")
    momentum_21d = _resolve_numeric_feature(prepared, "quant_momentum_21d")
    sector_relative_gap = _resolve_numeric_feature(prepared, "sector_relative_gap_return")
    sector_relative_intraday = _resolve_numeric_feature(prepared, "sector_relative_intraday_return")
    volatility_ratio = _resolve_numeric_feature(prepared, "quant_realized_vol_ratio_5d_21d")
    earnings_days_to_next = _resolve_numeric_feature(prepared, "earnings_days_to_next")
    relative_cost = _resolve_numeric_feature(prepared, "xtb_spread_to_gap_abs")
    open_stretch = _resolve_numeric_feature(prepared, "open_distance_to_prev_close_over_atr_21d").abs()
    gap_return = _resolve_numeric_feature(prepared, "quant_gap_return")
    open_above_prev_high = _resolve_binary_flag(prepared, "open_above_prev_high_flag")
    open_below_prev_low = _resolve_binary_flag(prepared, "open_below_prev_low_flag")
    open_in_prev_range = _resolve_binary_flag(prepared, "open_in_prev_range_flag")

    prepared["signal_rsi_overbought_macd_positive_flag"] = (
        (rsi > RSI_OVERBOUGHT_THRESHOLD) & (macd > 0.0)
    ).astype(np.int8)
    prepared["signal_rsi_oversold_macd_negative_flag"] = (
        (rsi < RSI_OVERSOLD_THRESHOLD) & (macd < 0.0)
    ).astype(np.int8)
    prepared["signal_gap_up_breakout_flag"] = (
        open_above_prev_high & (gap_return > 0.0)
    ).astype(np.int8)
    prepared["signal_gap_down_breakdown_flag"] = (
        open_below_prev_low & (gap_return < 0.0)
    ).astype(np.int8)
    prepared["signal_sector_strength_confirmed_flag"] = (
        (sector_relative_gap > 0.0) & (sector_relative_intraday > 0.0)
    ).astype(np.int8)
    prepared["signal_sector_weakness_confirmed_flag"] = (
        (sector_relative_gap < 0.0) & (sector_relative_intraday < 0.0)
    ).astype(np.int8)
    prepared["signal_earnings_tension_flag"] = (
        (earnings_days_to_next <= 5.0) & (volatility_ratio > VOLATILITY_TENSION_THRESHOLD)
    ).astype(np.int8)
    prepared["signal_open_stretch_high_cost_flag"] = (
        (open_stretch > OPEN_STRETCH_ATR_THRESHOLD) & (relative_cost > HIGH_RELATIVE_COST_THRESHOLD)
    ).astype(np.int8)
    prepared["signal_trend_follow_through_flag"] = (
        (macd > 0.0) & (momentum_21d > 0.0) & (sector_relative_intraday > 0.0)
    ).astype(np.int8)
    prepared["signal_mean_reversion_exhaustion_flag"] = (
        (rsi < RSI_OVERSOLD_THRESHOLD) & open_in_prev_range & (gap_return < 0.0)
    ).astype(np.int8)
    return prepared


def _load_earnings_reference(earnings_path: Path, trading_dates: pd.Series) -> pd.DataFrame:
    if not earnings_path.exists():
        raise FileNotFoundError(f"Earnings reference CSV not found: {earnings_path}")
    earnings = pd.read_csv(earnings_path)
    missing_columns = [column_name for column_name in EARNINGS_REQUIRED_COLUMNS if column_name not in earnings.columns]
    if missing_columns:
        raise ValueError(f"Missing earnings reference columns: {', '.join(missing_columns)}")
    prepared = pd.DataFrame(earnings.loc[:, list(EARNINGS_REQUIRED_COLUMNS)].copy())
    prepared["ticker"] = prepared["ticker"].astype(str)
    prepared["announcement_date"] = pd.to_datetime(prepared["announcement_date"])
    prepared["announcement_session"] = prepared["announcement_session"].astype(str).str.lower()
    invalid_sessions = prepared.loc[
        ~prepared["announcement_session"].isin(_SUPPORTED_EARNINGS_SESSIONS),
        "announcement_session",
    ]
    if not invalid_sessions.empty:
        invalid_preview = ", ".join(sorted({str(value) for value in invalid_sessions.tolist()}))
        raise ValueError(f"Unsupported earnings announcement_session values: {invalid_preview}")
    unknown_session_mask = prepared["announcement_session"].eq("unknown")
    if bool(unknown_session_mask.any()):
        LOGGER.warning(
            "Ignoring %d earnings events with unknown session timing; earnings alpha features remain disabled until session timing is explicit.",
            int(unknown_session_mask.sum()),
        )
        prepared = pd.DataFrame(prepared.loc[~unknown_session_mask].copy())
    effective_sessions = _resolve_effective_earnings_sessions(prepared, trading_dates)
    return effective_sessions.sort_values(["ticker", f"{TEMP_PREFIX}effective_session_index"]).reset_index(drop=True)


def _resolve_effective_earnings_sessions(earnings: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    trading_index = pd.Index(pd.to_datetime(trading_dates).sort_values().unique())
    trading_values = trading_index.to_numpy(dtype="datetime64[ns]")
    announcement_values = pd.to_datetime(earnings["announcement_date"]).to_numpy(dtype="datetime64[ns]")
    before_indices = np.searchsorted(trading_values, announcement_values, side="left")
    after_indices = np.searchsorted(trading_values, announcement_values, side="right")
    effective_indices = np.where(
        earnings["announcement_session"].to_numpy() == "after_close",
        after_indices,
        before_indices,
    )
    valid_mask = effective_indices < len(trading_index)
    prepared = pd.DataFrame(earnings.loc[valid_mask].copy())
    prepared[f"{TEMP_PREFIX}effective_session_index"] = effective_indices[valid_mask].astype(np.int64)
    return prepared


def _build_session_index_map(trading_dates: pd.Series) -> pd.DataFrame:
    unique_dates = pd.Index(pd.to_datetime(trading_dates).sort_values().unique())
    return pd.DataFrame({
        "date": unique_dates,
        f"{TEMP_PREFIX}session_index": np.arange(len(unique_dates), dtype=np.int64),
    })


def _compute_earnings_distances(data: pd.DataFrame, earnings: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    next_days = pd.Series(np.nan, index=data.index, dtype=np.float64)
    prev_days = pd.Series(np.nan, index=data.index, dtype=np.float64)
    earnings_by_ticker = {
        str(ticker): group[f"{TEMP_PREFIX}effective_session_index"].to_numpy(dtype=np.int64, copy=False)
        for ticker, group in earnings.groupby("ticker", sort=False)
    }
    for ticker, group in data.groupby("ticker", sort=False):
        event_indices = earnings_by_ticker.get(str(ticker), np.array([], dtype=np.int64))
        if event_indices.size == 0:
            continue
        session_indices = group[f"{TEMP_PREFIX}session_index"].to_numpy(dtype=np.int64, copy=False)
        next_pos = np.searchsorted(event_indices, session_indices, side="left")
        prev_pos = np.searchsorted(event_indices, session_indices, side="right") - 1
        next_distance = np.full(session_indices.shape, np.nan, dtype=np.float64)
        prev_distance = np.full(session_indices.shape, np.nan, dtype=np.float64)
        valid_next = next_pos < event_indices.size
        valid_prev = prev_pos >= 0
        next_distance[valid_next] = event_indices[next_pos[valid_next]] - session_indices[valid_next]
        prev_distance[valid_prev] = session_indices[valid_prev] - event_indices[prev_pos[valid_prev]]
        next_days.loc[group.index] = next_distance
        prev_days.loc[group.index] = prev_distance
    return next_days, prev_days


def _finalize_dataset(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [column_name for column_name in data.columns if column_name.startswith(TEMP_PREFIX)]
    finalized = pd.DataFrame(data.drop(columns=columns_to_drop).copy())
    return finalized.sort_values(["date", "ticker"]).reset_index(drop=True)


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid_mask = (numerator > 0.0) & (denominator > 0.0)
    result = pd.Series(np.nan, index=numerator.index, dtype=np.float64)
    result.loc[valid_mask] = np.log(numerator.loc[valid_mask] / denominator.loc[valid_mask])
    return result


def _safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    numerator_series = pd.Series(numerator, copy=False)
    denominator_series = pd.Series(denominator, copy=False)
    valid_mask = denominator_series.abs() > 1e-12
    result = pd.Series(np.nan, index=numerator_series.index, dtype=np.float64)
    result.loc[valid_mask] = numerator_series.loc[valid_mask] / denominator_series.loc[valid_mask]
    return result


def _resolve_numeric_feature(data: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=np.float64)
    return pd.to_numeric(data[column_name], errors="coerce")


def _resolve_binary_flag(data: pd.DataFrame, column_name: str) -> pd.Series:
    return _resolve_numeric_feature(data, column_name).fillna(0.0).astype(bool)


def _compute_true_range_pct(data: pd.DataFrame) -> pd.Series:
    high = data["stock_high_price"]
    low = data["stock_low_price"]
    prev_close = data[f"{TEMP_PREFIX}prev_close"]
    close = data["stock_close_price"]
    components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = cast_series(components.max(axis=1))
    return _safe_divide(true_range, close.abs())


def _build_sector_key(data: pd.DataFrame) -> pd.Series:
    sector = cast_series(data["company_sector"]).astype("string")
    industry = cast_series(data["company_industry"]).astype("string")
    resolved = sector.fillna(industry)
    return resolved.fillna("__missing_sector__").astype(str)


def _compute_rsi_series(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gains = cast_series(delta.clip(lower=0.0))
    losses = cast_series((-delta).clip(lower=0.0))
    avg_gain = gains.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    relative_strength = _safe_divide(avg_gain, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    zero_loss_mask = avg_loss.notna() & avg_gain.notna() & np.isclose(avg_loss, 0.0)
    zero_gain_mask = avg_gain.notna() & avg_loss.notna() & np.isclose(avg_gain, 0.0)
    flat_mask = zero_loss_mask & zero_gain_mask
    rsi = rsi.where(~zero_loss_mask, other=np.float64(100.0))
    rsi = rsi.where(~flat_mask, other=np.float64(50.0))
    rsi = rsi.where(~zero_gain_mask | zero_loss_mask, other=np.float64(0.0))
    return cast_series(rsi)


def cast_series(series: pd.Series | pd.DataFrame) -> pd.Series:
    return cast(pd.Series, series)


__all__ = ["add_high_level_features"]
