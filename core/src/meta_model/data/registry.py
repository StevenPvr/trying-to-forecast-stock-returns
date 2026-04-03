from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pandas as pd

from core.src.meta_model.features_engineering.config import (
    CALENDAR_FEATURE_PREFIX,
    COMPANY_FEATURE_PREFIX,
    CROSS_ASSET_FEATURE_PREFIX,
    DEEP_FEATURE_PREFIX,
    MACRO_FEATURE_PREFIX,
    QUANT_FEATURE_PREFIX,
    SIGNAL_FEATURE_PREFIX,
    SENTIMENT_FEATURE_PREFIX,
    TA_FEATURE_PREFIX,
)
from core.src.meta_model.model_contract import (
    DATE_COLUMN,
    SIGNAL_DATE_COLUMN,
    SPLIT_COLUMN,
    TICKER_COLUMN,
    is_excluded_feature_column,
)

FEATURE_NAME_COLUMN: str = "feature_name"
FEATURE_FAMILY_COLUMN: str = "family"
SOURCE_COLUMN: str = "source"
AVAILABILITY_LAG_COLUMN: str = "availability_lag_sessions"
SAFE_FFILL_MAX_DAYS_COLUMN: str = "safe_ffill_max_days"
MISSING_POLICY_COLUMN: str = "missing_policy"
IS_DATE_LEVEL_COLUMN: str = "is_date_level"
IS_CROSS_SECTIONAL_COLUMN: str = "is_cross_sectional"
ENABLED_FOR_ALPHA_COLUMN: str = "enabled_for_alpha_model"

LEAVE_MISSING_POLICY: str = "leave_missing"
FFILL_LIMITED_POLICY: str = "ffill_limited"
DISALLOW_POLICY: str = "disallow"

FEATURE_REGISTRY_COLUMNS: tuple[str, ...] = (
    FEATURE_NAME_COLUMN,
    FEATURE_FAMILY_COLUMN,
    SOURCE_COLUMN,
    AVAILABILITY_LAG_COLUMN,
    SAFE_FFILL_MAX_DAYS_COLUMN,
    MISSING_POLICY_COLUMN,
    IS_DATE_LEVEL_COLUMN,
    IS_CROSS_SECTIONAL_COLUMN,
    ENABLED_FOR_ALPHA_COLUMN,
)


def _split_lag_suffix(column_name: str) -> tuple[str, int]:
    marker = "_lag_"
    if marker not in column_name:
        return column_name, 0
    base_name, lag_suffix = column_name.rsplit(marker, maxsplit=1)
    lag_days_text = lag_suffix.removesuffix("d")
    if not lag_days_text.isdigit():
        return column_name, 0
    return base_name, int(lag_days_text)


def _base_feature_metadata(column_name: str) -> dict[str, object]:
    if column_name.startswith("pred_"):
        return {
            FEATURE_FAMILY_COLUMN: "stacking",
            SOURCE_COLUMN: "secondary_prediction",
            AVAILABILITY_LAG_COLUMN: 0,
            SAFE_FFILL_MAX_DAYS_COLUMN: 0,
            MISSING_POLICY_COLUMN: DISALLOW_POLICY,
            IS_DATE_LEVEL_COLUMN: False,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: False,
        }
    if column_name.startswith("xtb_"):
        return {
            FEATURE_FAMILY_COLUMN: "broker",
            SOURCE_COLUMN: "broker_xtb",
            AVAILABILITY_LAG_COLUMN: 0,
            SAFE_FFILL_MAX_DAYS_COLUMN: 0,
            MISSING_POLICY_COLUMN: LEAVE_MISSING_POLICY,
            IS_DATE_LEVEL_COLUMN: False,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith("sector_"):
        return {
            FEATURE_FAMILY_COLUMN: "sector",
            SOURCE_COLUMN: "derived_cross_section",
            AVAILABILITY_LAG_COLUMN: 0,
            SAFE_FFILL_MAX_DAYS_COLUMN: 0,
            MISSING_POLICY_COLUMN: LEAVE_MISSING_POLICY,
            IS_DATE_LEVEL_COLUMN: False,
            IS_CROSS_SECTIONAL_COLUMN: True,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith("open_"):
        return {
            FEATURE_FAMILY_COLUMN: "open",
            SOURCE_COLUMN: "derived_price",
            AVAILABILITY_LAG_COLUMN: 0,
            SAFE_FFILL_MAX_DAYS_COLUMN: 0,
            MISSING_POLICY_COLUMN: LEAVE_MISSING_POLICY,
            IS_DATE_LEVEL_COLUMN: False,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith("earnings_"):
        return {
            FEATURE_FAMILY_COLUMN: "earnings",
            SOURCE_COLUMN: "earnings_calendar",
            AVAILABILITY_LAG_COLUMN: 0,
            SAFE_FFILL_MAX_DAYS_COLUMN: 0,
            MISSING_POLICY_COLUMN: LEAVE_MISSING_POLICY,
            IS_DATE_LEVEL_COLUMN: False,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith(SIGNAL_FEATURE_PREFIX):
        return {
            FEATURE_FAMILY_COLUMN: "signal",
            SOURCE_COLUMN: "derived_interaction",
            AVAILABILITY_LAG_COLUMN: 0,
            SAFE_FFILL_MAX_DAYS_COLUMN: 0,
            MISSING_POLICY_COLUMN: LEAVE_MISSING_POLICY,
            IS_DATE_LEVEL_COLUMN: False,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith(MACRO_FEATURE_PREFIX):
        return {
            FEATURE_FAMILY_COLUMN: "macro",
            SOURCE_COLUMN: "macro",
            AVAILABILITY_LAG_COLUMN: 1,
            SAFE_FFILL_MAX_DAYS_COLUMN: 21,
            MISSING_POLICY_COLUMN: FFILL_LIMITED_POLICY,
            IS_DATE_LEVEL_COLUMN: True,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith(SENTIMENT_FEATURE_PREFIX):
        return {
            FEATURE_FAMILY_COLUMN: "sentiment",
            SOURCE_COLUMN: "sentiment",
            AVAILABILITY_LAG_COLUMN: 1,
            SAFE_FFILL_MAX_DAYS_COLUMN: 10,
            MISSING_POLICY_COLUMN: FFILL_LIMITED_POLICY,
            IS_DATE_LEVEL_COLUMN: True,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith(CALENDAR_FEATURE_PREFIX):
        return {
            FEATURE_FAMILY_COLUMN: "calendar",
            SOURCE_COLUMN: "calendar",
            AVAILABILITY_LAG_COLUMN: 1,
            SAFE_FFILL_MAX_DAYS_COLUMN: 1,
            MISSING_POLICY_COLUMN: FFILL_LIMITED_POLICY,
            IS_DATE_LEVEL_COLUMN: True,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith(CROSS_ASSET_FEATURE_PREFIX):
        return {
            FEATURE_FAMILY_COLUMN: "cross_asset",
            SOURCE_COLUMN: "cross_asset",
            AVAILABILITY_LAG_COLUMN: 1,
            SAFE_FFILL_MAX_DAYS_COLUMN: 5,
            MISSING_POLICY_COLUMN: FFILL_LIMITED_POLICY,
            IS_DATE_LEVEL_COLUMN: True,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith(COMPANY_FEATURE_PREFIX):
        return {
            FEATURE_FAMILY_COLUMN: "company",
            SOURCE_COLUMN: "company_history",
            AVAILABILITY_LAG_COLUMN: 1,
            SAFE_FFILL_MAX_DAYS_COLUMN: 63,
            MISSING_POLICY_COLUMN: FFILL_LIMITED_POLICY,
            IS_DATE_LEVEL_COLUMN: False,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith(TA_FEATURE_PREFIX):
        return {
            FEATURE_FAMILY_COLUMN: "ta",
            SOURCE_COLUMN: "derived_price",
            AVAILABILITY_LAG_COLUMN: 0,
            SAFE_FFILL_MAX_DAYS_COLUMN: 5,
            MISSING_POLICY_COLUMN: FFILL_LIMITED_POLICY,
            IS_DATE_LEVEL_COLUMN: False,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith(QUANT_FEATURE_PREFIX):
        return {
            FEATURE_FAMILY_COLUMN: "quant",
            SOURCE_COLUMN: "derived_price",
            AVAILABILITY_LAG_COLUMN: 0,
            SAFE_FFILL_MAX_DAYS_COLUMN: 5,
            MISSING_POLICY_COLUMN: FFILL_LIMITED_POLICY,
            IS_DATE_LEVEL_COLUMN: False,
            IS_CROSS_SECTIONAL_COLUMN: column_name.startswith(f"{QUANT_FEATURE_PREFIX}cs_"),
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith(DEEP_FEATURE_PREFIX):
        return {
            FEATURE_FAMILY_COLUMN: "deep",
            SOURCE_COLUMN: "derived_price",
            AVAILABILITY_LAG_COLUMN: 0,
            SAFE_FFILL_MAX_DAYS_COLUMN: 5,
            MISSING_POLICY_COLUMN: FFILL_LIMITED_POLICY,
            IS_DATE_LEVEL_COLUMN: False,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    if column_name.startswith("stock_"):
        return {
            FEATURE_FAMILY_COLUMN: "stock",
            SOURCE_COLUMN: "market_prices",
            AVAILABILITY_LAG_COLUMN: 0,
            SAFE_FFILL_MAX_DAYS_COLUMN: 0,
            MISSING_POLICY_COLUMN: LEAVE_MISSING_POLICY,
            IS_DATE_LEVEL_COLUMN: False,
            IS_CROSS_SECTIONAL_COLUMN: False,
            ENABLED_FOR_ALPHA_COLUMN: True,
        }
    return {
        FEATURE_FAMILY_COLUMN: "other",
        SOURCE_COLUMN: "unknown",
        AVAILABILITY_LAG_COLUMN: 0,
        SAFE_FFILL_MAX_DAYS_COLUMN: 0,
        MISSING_POLICY_COLUMN: LEAVE_MISSING_POLICY,
        IS_DATE_LEVEL_COLUMN: False,
        IS_CROSS_SECTIONAL_COLUMN: False,
        ENABLED_FOR_ALPHA_COLUMN: True,
    }


def infer_feature_spec(column_name: str) -> dict[str, object]:
    base_name, lag_days = _split_lag_suffix(column_name)
    base_metadata = _base_feature_metadata(base_name)
    availability_lag = cast(int, base_metadata[AVAILABILITY_LAG_COLUMN]) + lag_days
    feature_spec = {
        FEATURE_NAME_COLUMN: column_name,
        **base_metadata,
        AVAILABILITY_LAG_COLUMN: availability_lag,
    }
    return feature_spec


def _select_feature_columns(columns: list[str]) -> list[str]:
    return [
        column_name
        for column_name in columns
        if column_name not in {DATE_COLUMN, TICKER_COLUMN, SPLIT_COLUMN, SIGNAL_DATE_COLUMN}
        and not is_excluded_feature_column(column_name)
    ]


def build_feature_registry_from_columns(columns: list[str]) -> pd.DataFrame:
    feature_columns = _select_feature_columns(columns)
    rows = [infer_feature_spec(column_name) for column_name in sorted(feature_columns)]
    return pd.DataFrame(rows, columns=list(FEATURE_REGISTRY_COLUMNS))


def build_feature_registry(data: pd.DataFrame) -> pd.DataFrame:
    return build_feature_registry_from_columns(list(data.columns))


def load_feature_registry(path: Path) -> pd.DataFrame:
    registry = pd.read_parquet(path)
    return registry.loc[:, list(FEATURE_REGISTRY_COLUMNS)].copy()


def save_feature_registry(registry: pd.DataFrame, parquet_path: Path, json_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = registry.loc[:, list(FEATURE_REGISTRY_COLUMNS)].sort_values(FEATURE_NAME_COLUMN).reset_index(drop=True)
    ordered.to_parquet(parquet_path, index=False)
    json_path.write_text(
        ordered.to_json(orient="records", indent=2),
        encoding="utf-8",
    )


def compute_feature_schema_hash(feature_names: list[str]) -> str:
    ordered = "\n".join(feature_names).encode("utf-8")
    return hashlib.sha256(ordered).hexdigest()


def save_feature_schema_manifest(
    feature_names: list[str],
    manifest_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": feature_names,
        "feature_schema_hash": compute_feature_schema_hash(feature_names),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_feature_schema_manifest(manifest_path: Path) -> dict[str, object]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


__all__ = [
    "AVAILABILITY_LAG_COLUMN",
    "DISALLOW_POLICY",
    "ENABLED_FOR_ALPHA_COLUMN",
    "FEATURE_FAMILY_COLUMN",
    "FEATURE_NAME_COLUMN",
    "FEATURE_REGISTRY_COLUMNS",
    "FFILL_LIMITED_POLICY",
    "IS_CROSS_SECTIONAL_COLUMN",
    "IS_DATE_LEVEL_COLUMN",
    "LEAVE_MISSING_POLICY",
    "MISSING_POLICY_COLUMN",
    "SAFE_FFILL_MAX_DAYS_COLUMN",
    "SOURCE_COLUMN",
    "build_feature_registry",
    "build_feature_registry_from_columns",
    "compute_feature_schema_hash",
    "infer_feature_spec",
    "load_feature_registry",
    "load_feature_schema_manifest",
    "save_feature_registry",
    "save_feature_schema_manifest",
]
