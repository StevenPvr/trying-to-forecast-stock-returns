"""Shared column names and the temporal contract for labels vs features.

One row is (ticker, signal day J): decision is made after close(J) when all same-day
market features are available.

- Features on that row must be known at close(J).
- Trading execution default: next regular open (J+1).
- Primary week target (WEEK_HOLD_*): log return from open(J+1) to close(J+6)
    (five sessions after entry).
"""

from __future__ import annotations

DATE_COLUMN: str = "date"
TICKER_COLUMN: str = "ticker"
SPLIT_COLUMN: str = "dataset_split"
TRAIN_SPLIT_NAME: str = "train"
VAL_SPLIT_NAME: str = "val"
TEST_SPLIT_NAME: str = "test"

LEGACY_TARGET_COLUMN: str = "target_main"

INTRADAY_GROSS_RETURN_COLUMN: str = "target_intraday_open_to_close_log_return"
INTRADAY_NET_RETURN_COLUMN: str = "target_intraday_open_to_close_net_log_return"
INTRADAY_BENCHMARK_RETURN_COLUMN: str = "benchmark_intraday_open_to_close_net_log_return"
INTRADAY_EXCESS_RETURN_COLUMN: str = "target_intraday_open_to_close_excess_log_return"
INTRADAY_SECTOR_RESIDUAL_RETURN_COLUMN: str = (
    "target_intraday_open_to_close_sector_residual_log_return"
)
INTRADAY_CS_ZSCORE_TARGET_COLUMN: str = "target_intraday_open_to_close_net_cs_zscore"
INTRADAY_CS_RANK_TARGET_COLUMN: str = "target_intraday_open_to_close_net_cs_rank"

OVERNIGHT_NET_RETURN_COLUMN: str = "target_overnight_close_to_next_open_net_log_return"
SHORT_HOLD_NET_RETURN_COLUMN: str = "target_short_hold_1d_to_2d_net_log_return"
MEDIUM_HOLD_GROSS_RETURN_COLUMN: str = "target_medium_hold_3d_to_5d_log_return"
MEDIUM_HOLD_NET_RETURN_COLUMN: str = "target_medium_hold_3d_to_5d_net_log_return"

# Row date J = signal day; entry = open(J+1); exit = close five sessions after entry.
WEEK_HOLD_GROSS_RETURN_COLUMN: str = "target_week_hold_5sessions_close_log_return"
WEEK_HOLD_NET_RETURN_COLUMN: str = "target_week_hold_5sessions_net_log_return"
WEEK_HOLD_BENCHMARK_RETURN_COLUMN: str = "benchmark_week_hold_net_log_return"
WEEK_HOLD_EXCESS_RETURN_COLUMN: str = "target_week_hold_excess_log_return"
WEEK_HOLD_SECTOR_RESIDUAL_RETURN_COLUMN: str = "target_week_hold_sector_residual_log_return"
WEEK_HOLD_CS_ZSCORE_TARGET_COLUMN: str = "target_week_hold_net_cs_zscore"
WEEK_HOLD_CS_RANK_TARGET_COLUMN: str = "target_week_hold_net_cs_rank"

RAW_FORWARD_RETURN_COLUMN: str = WEEK_HOLD_GROSS_RETURN_COLUMN
BENCHMARK_FORWARD_RETURN_COLUMN: str = WEEK_HOLD_BENCHMARK_RETURN_COLUMN
EXCESS_FORWARD_RETURN_COLUMN: str = WEEK_HOLD_EXCESS_RETURN_COLUMN
SECTOR_RESIDUAL_FORWARD_RETURN_COLUMN: str = WEEK_HOLD_SECTOR_RESIDUAL_RETURN_COLUMN
CS_ZSCORE_TARGET_COLUMN: str = WEEK_HOLD_CS_ZSCORE_TARGET_COLUMN
CS_RANK_TARGET_COLUMN: str = WEEK_HOLD_CS_RANK_TARGET_COLUMN
MODEL_TARGET_COLUMN: str = WEEK_HOLD_CS_ZSCORE_TARGET_COLUMN
REALIZED_RETURN_COLUMN: str = WEEK_HOLD_GROSS_RETURN_COLUMN
PRIMARY_PRODUCTION_LABEL_COLUMN: str = WEEK_HOLD_NET_RETURN_COLUMN
SECONDARY_RESEARCH_LABEL_COLUMNS: tuple[str, ...] = (
    INTRADAY_NET_RETURN_COLUMN,
    OVERNIGHT_NET_RETURN_COLUMN,
    SHORT_HOLD_NET_RETURN_COLUMN,
    MEDIUM_HOLD_NET_RETURN_COLUMN,
)

PREDICTION_COLUMN: str = "prediction"
SIGNAL_DATE_COLUMN: str = "signal_date"

EXECUTION_LAG_DAYS: int = 1
HOLD_PERIOD_DAYS: int = 5
LABEL_EMBARGO_DAYS: int = EXECUTION_LAG_DAYS + HOLD_PERIOD_DAYS
if LABEL_EMBARGO_DAYS < EXECUTION_LAG_DAYS + HOLD_PERIOD_DAYS:
    raise ValueError(
        f"LABEL_EMBARGO_DAYS ({LABEL_EMBARGO_DAYS}) must be >= "
        f"EXECUTION_LAG_DAYS + HOLD_PERIOD_DAYS ({EXECUTION_LAG_DAYS + HOLD_PERIOD_DAYS})"
    )

TARGET_PREFIXES: tuple[str, ...] = ("target_",)
BENCHMARK_PREFIXES: tuple[str, ...] = ("benchmark_",)
HIGH_LEVEL_CONTEXT_PREFIXES: tuple[str, ...] = ("hl_context_",)
TEMPORARILY_DISABLED_ALPHA_FEATURE_PREFIXES: tuple[str, ...] = ("earnings_",)
EXACT_EXCLUDED_FEATURE_COLUMNS: frozenset[str] = frozenset({
    DATE_COLUMN,
    SPLIT_COLUMN,
    LEGACY_TARGET_COLUMN,
    SIGNAL_DATE_COLUMN,
})

# Always-on identity / GICS columns for XGBoost native categoricals (not scored in SFI).
STRUCTURAL_CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    TICKER_COLUMN,
    "company_sector",
    "company_industry",
)


def is_excluded_feature_column(column_name: str) -> bool:
    """Return True if *column_name* must never be used as a model feature."""
    if column_name in EXACT_EXCLUDED_FEATURE_COLUMNS:
        return True
    if column_name.startswith(HIGH_LEVEL_CONTEXT_PREFIXES):
        return True
    if column_name.startswith(TARGET_PREFIXES):
        return True
    return column_name.startswith(BENCHMARK_PREFIXES)


def is_temporarily_disabled_alpha_feature_column(column_name: str) -> bool:
    """Return True if *column_name* belongs to a temporarily disabled alpha group."""
    return column_name.startswith(TEMPORARILY_DISABLED_ALPHA_FEATURE_PREFIXES)


def is_structural_categorical_feature_column(column_name: str) -> bool:
    """Return True if *column_name* is a structural categorical (ticker, sector, industry)."""
    return column_name in STRUCTURAL_CATEGORICAL_FEATURE_COLUMNS


def merge_structural_feature_names_into_selected(
    selected_feature_names: list[str],
    *,
    available_columns: frozenset[str] | set[str],
) -> list[str]:
    """Prepend structural categoricals present in the dataset; dedupe against selected list."""
    available = set(available_columns)
    structural = [name for name in STRUCTURAL_CATEGORICAL_FEATURE_COLUMNS if name in available]
    structural_set = set(structural)
    tail = [name for name in selected_feature_names if name not in structural_set]
    merged = [*structural, *tail]
    if len(merged) != len(set(merged)):
        raise ValueError("merge_structural_feature_names_into_selected produced duplicate feature names.")
    return merged
