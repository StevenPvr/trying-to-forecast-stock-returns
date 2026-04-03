from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.registry import (
    AVAILABILITY_LAG_COLUMN,
    ENABLED_FOR_ALPHA_COLUMN,
    FEATURE_NAME_COLUMN,
    MISSING_POLICY_COLUMN,
    SAFE_FFILL_MAX_DAYS_COLUMN,
    build_feature_registry,
    build_feature_registry_from_columns,
)


def test_build_feature_registry_declares_contract_fields_for_each_feature() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "ticker": ["AAA", "AAA"],
        "stock_open_price": [100.0, 101.0],
        "ta_trend_macd": [0.1, 0.2],
        "macro_vix_close_level": [15.0, 16.0],
        "company_market_cap_usd": [1_000_000_000.0, 1_010_000_000.0],
        "quant_momentum_21d_lag_5d": [0.03, 0.04],
    })

    registry = build_feature_registry(frame)

    assert {
        FEATURE_NAME_COLUMN,
        AVAILABILITY_LAG_COLUMN,
        SAFE_FFILL_MAX_DAYS_COLUMN,
        MISSING_POLICY_COLUMN,
        ENABLED_FOR_ALPHA_COLUMN,
    } <= set(registry.columns)

    macro_row = registry.loc[registry[FEATURE_NAME_COLUMN] == "macro_vix_close_level"].iloc[0]
    lagged_row = registry.loc[registry[FEATURE_NAME_COLUMN] == "quant_momentum_21d_lag_5d"].iloc[0]

    assert int(macro_row[AVAILABILITY_LAG_COLUMN]) == 1
    assert int(macro_row[SAFE_FFILL_MAX_DAYS_COLUMN]) == 21
    assert str(macro_row[MISSING_POLICY_COLUMN]) == "ffill_limited"
    assert int(lagged_row[AVAILABILITY_LAG_COLUMN]) == 5


def test_build_feature_registry_disables_pred_features_without_explicit_contract() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "ticker": ["AAA"],
        "pred_future_trend_5d": [0.3],
        "stock_open_price": [100.0],
    })

    registry = build_feature_registry(frame)

    pred_row = registry.loc[registry[FEATURE_NAME_COLUMN] == "pred_future_trend_5d"].iloc[0]

    assert bool(pred_row[ENABLED_FOR_ALPHA_COLUMN]) is False
    assert str(pred_row[MISSING_POLICY_COLUMN]) == "disallow"


def test_build_feature_registry_from_columns_matches_dataframe_path() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "ticker": ["AAA"],
        "stock_open_price": [100.0],
        "macro_vix_close_level": [15.0],
        "company_market_cap_usd": [1_000_000_000.0],
    })

    from_frame = build_feature_registry(frame)
    from_columns = build_feature_registry_from_columns(list(frame.columns))

    assert from_columns.equals(from_frame)


def test_build_feature_registry_recognizes_high_level_feature_families() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "ticker": ["AAA"],
        "xtb_expected_intraday_cost_rate": [0.001],
        "sector_relative_gap_return": [0.02],
        "open_above_prev_high_flag": [1],
        "earnings_days_to_next": [3.0],
        "signal_gap_up_breakout_flag": [1],
    })

    registry = build_feature_registry(frame).set_index(FEATURE_NAME_COLUMN)

    assert registry.loc["xtb_expected_intraday_cost_rate", "family"] == "broker"
    assert registry.loc["sector_relative_gap_return", "family"] == "sector"
    assert registry.loc["open_above_prev_high_flag", "family"] == "open"
    assert registry.loc["earnings_days_to_next", "family"] == "earnings"
    assert registry.loc["signal_gap_up_breakout_flag", "family"] == "signal"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
