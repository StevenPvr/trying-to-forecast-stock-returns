from __future__ import annotations

"""Feature engineering constants: prefixes, windows, column lists, and trading-day calendar."""

REQUIRED_TA_INPUT_COLUMNS: tuple[str, ...] = (
    "stock_open_price",
    "stock_high_price",
    "stock_low_price",
    "stock_close_price",
    "stock_trading_volume",
)
TA_FEATURE_PREFIX: str = "ta_"
QUANT_FEATURE_PREFIX: str = "quant_"
DEEP_FEATURE_PREFIX: str = "deep_"
CALENDAR_FEATURE_PREFIX: str = "calendar_"
SENTIMENT_FEATURE_PREFIX: str = "sentiment_"
MACRO_FEATURE_PREFIX: str = "macro_"
CROSS_ASSET_FEATURE_PREFIX: str = "cross_asset_"
COMPANY_FEATURE_PREFIX: str = "company_"
BROKER_FEATURE_PREFIX: str = "xtb_"
SECTOR_FEATURE_PREFIX: str = "sector_"
OPEN_FEATURE_PREFIX: str = "open_"
EARNINGS_FEATURE_PREFIX: str = "earnings_"
SIGNAL_FEATURE_PREFIX: str = "signal_"
INTERNAL_FEATURE_PREFIX: str = "_quant_internal_"
FEATURE_LAG_WINDOWS: tuple[int, ...] = (1, 5, 21)
SLOW_FEATURE_LAG_WINDOWS: tuple[int, ...] = (1, 21)
COMPANY_FEATURE_LAG_WINDOWS: tuple[int, ...] = (1, 21)
CALENDAR_SINCE_LAG_WINDOWS: tuple[int, ...] = (1, 5)
NON_LAGGABLE_TA_PREFIXES: tuple[str, ...] = (
    f"{TA_FEATURE_PREFIX}volume_",
)
STOCK_LOG_RETURN_COLUMNS: tuple[str, ...] = (
    "stock_open_log_return",
    "stock_high_log_return",
    "stock_low_log_return",
    "stock_close_log_return",
)
NON_LAGGABLE_QUANT_LAG_FEATURES: tuple[str, ...] = (
    f"{QUANT_FEATURE_PREFIX}dollar_volume",
    f"{QUANT_FEATURE_PREFIX}adv_21d",
    f"{QUANT_FEATURE_PREFIX}adv_63d",
    f"{QUANT_FEATURE_PREFIX}adv_252d",
    f"{QUANT_FEATURE_PREFIX}trend_strength_21d",
    f"{QUANT_FEATURE_PREFIX}trend_strength_63d",
)
NON_LAGGABLE_QUANT_FEATURES: tuple[str, ...] = (
    f"{QUANT_FEATURE_PREFIX}day_of_week_sin",
    f"{QUANT_FEATURE_PREFIX}day_of_week_cos",
    f"{QUANT_FEATURE_PREFIX}month_of_year_sin",
    f"{QUANT_FEATURE_PREFIX}month_of_year_cos",
    f"{QUANT_FEATURE_PREFIX}day_of_month_sin",
    f"{QUANT_FEATURE_PREFIX}day_of_month_cos",
    f"{QUANT_FEATURE_PREFIX}day_of_year_sin",
    f"{QUANT_FEATURE_PREFIX}day_of_year_cos",
    f"{QUANT_FEATURE_PREFIX}is_month_start",
    f"{QUANT_FEATURE_PREFIX}is_month_end",
    f"{QUANT_FEATURE_PREFIX}is_quarter_start",
    f"{QUANT_FEATURE_PREFIX}is_quarter_end",
    f"{QUANT_FEATURE_PREFIX}days_in_sample",
    *NON_LAGGABLE_QUANT_LAG_FEATURES,
)

RETURN_WINDOWS: tuple[int, ...] = (5, 10, 21, 63, 126, 252)
VOLATILITY_WINDOWS: tuple[int, ...] = (5, 10, 21, 63, 126)
TREND_WINDOWS: tuple[int, ...] = (21, 63)
LIQUIDITY_WINDOWS: tuple[int, ...] = (21, 63, 252)
DRAWDOWN_WINDOWS: tuple[int, ...] = (63, 252)
MARKET_WINDOWS: tuple[int, ...] = (21, 63)
TRADING_DAYS_PER_YEAR: int = 252
CROSS_SECTIONAL_BASE_FEATURES: tuple[str, ...] = (
    f"{QUANT_FEATURE_PREFIX}momentum_21d",
    f"{QUANT_FEATURE_PREFIX}momentum_63d",
    f"{QUANT_FEATURE_PREFIX}trend_strength_63d",
    f"{QUANT_FEATURE_PREFIX}realized_vol_21d",
    f"{QUANT_FEATURE_PREFIX}amihud_illiquidity_21d",
    f"{QUANT_FEATURE_PREFIX}log_dollar_volume",
    f"{QUANT_FEATURE_PREFIX}drawdown_252d",
    f"{QUANT_FEATURE_PREFIX}relative_strength_63d",
    f"{QUANT_FEATURE_PREFIX}volume_ratio_21d",
)
