from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.features_engineering.high_level_features import (
    _compute_rsi_series,
    add_high_level_features,
)
from core.src.meta_model.features_engineering.pipeline import build_feature_dataset


def _make_base_feature_dataset(periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    tickers = ("AAA", "BBB")
    rows: list[dict[str, object]] = []
    for date_index, current_date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            base_price = 100.0 + (1.5 * ticker_index) + (0.2 * date_index)
            open_price = base_price + (0.1 if ticker == "AAA" else -0.05)
            close_price = open_price + np.sin((date_index + 1) / 5.0) + (0.2 * ticker_index)
            high_price = max(open_price, close_price) + 0.8
            low_price = min(open_price, close_price) - 0.8
            rows.append(
                {
                    "date": current_date,
                    "ticker": ticker,
                    "stock_open_price": open_price,
                    "stock_high_price": high_price,
                    "stock_low_price": low_price,
                    "stock_close_price": close_price,
                    "stock_trading_volume": 1_000_000.0 + (10_000.0 * date_index),
                    "company_sector": "Technology" if ticker == "AAA" else "Industrials",
                    "company_industry": "Software" if ticker == "AAA" else "Machinery",
                },
            )
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def _write_earnings_reference(path: Path) -> None:
    pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "announcement_date": ["2024-03-01", "2024-04-15", "2024-03-05"],
            "announcement_session": ["before_open", "after_close", "after_close"],
            "fiscal_year": [2024, 2024, 2024],
            "fiscal_quarter": [1, 2, 1],
        },
    ).to_csv(path, index=False)


def test_add_high_level_features_adds_expected_columns(tmp_path: Path) -> None:
    earnings_path = tmp_path / "earnings.csv"
    _write_earnings_reference(earnings_path)

    enriched = add_high_level_features(_make_base_feature_dataset(), earnings_path=earnings_path)

    expected_columns = {
        "xtb_expected_intraday_cost_rate",
        "sector_relative_gap_return",
        "open_above_prev_high_flag",
        "quant_realized_vol_ratio_21d_63d",
        "earnings_days_to_next",
        "earnings_proximity_bucket",
        "signal_gap_up_breakout_flag",
        "signal_trend_follow_through_flag",
    }
    assert expected_columns <= set(enriched.columns)
    assert not any(column_name.startswith("__hl_") for column_name in enriched.columns)
    valid_tail = enriched.dropna(
        subset=[
            "xtb_spread_to_realized_vol_21d",
            "sector_relative_rsi",
            "quant_realized_vol_ratio_21d_63d",
        ],
    )
    assert not valid_tail.empty


def test_add_high_level_features_aligns_earnings_sessions(tmp_path: Path) -> None:
    earnings_path = tmp_path / "earnings.csv"
    pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "announcement_date": ["2024-01-03", "2024-01-05"],
            "announcement_session": ["after_close", "before_open"],
            "fiscal_year": [2024, 2024],
            "fiscal_quarter": [1, 1],
        },
    ).to_csv(earnings_path, index=False)
    input_dataset = _make_base_feature_dataset(periods=6).loc[lambda frame: frame["ticker"] == "AAA"].reset_index(drop=True)

    enriched = add_high_level_features(input_dataset, earnings_path=earnings_path)
    lookup = enriched.set_index("date")

    assert float(lookup.loc[pd.Timestamp("2024-01-03"), "earnings_days_to_next"]) == 1.0
    assert float(lookup.loc[pd.Timestamp("2024-01-04"), "earnings_days_since_last"]) == 0.0
    assert float(lookup.loc[pd.Timestamp("2024-01-05"), "earnings_days_to_next"]) == 0.0


def test_add_high_level_features_ignores_unknown_earnings_sessions(tmp_path: Path) -> None:
    earnings_path = tmp_path / "earnings.csv"
    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "announcement_date": ["2024-01-03"],
            "announcement_session": ["unknown"],
            "fiscal_year": [2024],
            "fiscal_quarter": [1],
        },
    ).to_csv(earnings_path, index=False)
    input_dataset = _make_base_feature_dataset(periods=6).loc[
        lambda frame: frame["ticker"] == "AAA"
    ].reset_index(drop=True)

    enriched = add_high_level_features(input_dataset, earnings_path=earnings_path)

    assert set(enriched["earnings_days_to_next"].unique()) == {252.0}
    assert set(enriched["earnings_days_since_last"].unique()) == {252.0}


def test_add_high_level_features_requires_earnings_reference(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Earnings reference CSV not found"):
        add_high_level_features(_make_base_feature_dataset(periods=10), earnings_path=missing_path)


def test_compute_rsi_series_handles_zero_loss_and_flat_windows() -> None:
    rising_prices = pd.Series(np.arange(1.0, 25.0, dtype=np.float64))
    flat_prices = pd.Series(np.full(24, 100.0, dtype=np.float64))

    rising_rsi = _compute_rsi_series(rising_prices)
    flat_rsi = _compute_rsi_series(flat_prices)

    assert float(rising_rsi.iloc[-1]) == pytest.approx(100.0)
    assert float(flat_rsi.iloc[-1]) == pytest.approx(50.0)


def test_build_feature_dataset_includes_high_level_columns(tmp_path: Path) -> None:
    earnings_path = tmp_path / "earnings.csv"
    _write_earnings_reference(earnings_path)
    dates = pd.date_range("2024-01-02", periods=120, freq="D")
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(("AAA", "BBB"), start=1):
        base = 100.0 * ticker_index
        for step, date in enumerate(dates):
            price = base + step
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "stock_open_price": price,
                    "stock_high_price": price + 1.0,
                    "stock_low_price": price - 1.0,
                    "stock_close_price": price + 0.5,
                    "stock_trading_volume": 1_000_000.0 + (ticker_index * 1000.0) + step,
                    "stock_open_log_return": np.nan if step == 0 else 0.01,
                    "company_sector": "Technology" if ticker == "AAA" else "Industrials",
                    "company_industry": "Software" if ticker == "AAA" else "Machinery",
                },
            )
    cleaned = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)

    featured = build_feature_dataset(cleaned, earnings_path=earnings_path)

    assert "xtb_spread_to_realized_vol_21d" in featured.columns
    assert "sector_relative_gap_return" in featured.columns
    assert "earnings_days_to_next" in featured.columns
    assert "signal_rsi_overbought_macd_positive_flag" in featured.columns
    assert "signal_open_stretch_high_cost_flag" in featured.columns


def test_add_high_level_features_rebuilds_canonical_earnings_reference(tmp_path: Path) -> None:
    earnings_path = tmp_path / "earnings.csv"
    _write_earnings_reference(earnings_path)
    with patch(
        "core.src.meta_model.features_engineering.high_level_features.ensure_earnings_history_output",
        return_value=earnings_path,
    ) as ensure_earnings:
        add_high_level_features(_make_base_feature_dataset())

    ensure_earnings.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
