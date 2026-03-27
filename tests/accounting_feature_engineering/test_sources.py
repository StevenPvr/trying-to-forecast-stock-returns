from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.accounting_feature_engineering.history import (
    build_accounting_history_for_universe,
    combine_accounting_histories,
)
from core.src.accounting_feature_engineering.sec_source import build_sec_quarterly_history
from core.src.accounting_feature_engineering.yfinance_source import build_yfinance_quarterly_history


class _FakeTicker:
    def __init__(self) -> None:
        first_quarter = pd.Timestamp("2024-03-31")
        second_quarter = pd.Timestamp("2024-06-30")
        self.quarterly_income_stmt = pd.DataFrame(
            {
                first_quarter: [100.0, 52.0, 18.0, 12.0],
                second_quarter: [120.0, 60.0, 21.0, 15.0],
            },
            index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
        )
        self.quarterly_balance_sheet = pd.DataFrame(
            {
                first_quarter: [400.0, 180.0, 220.0, 130.0, 70.0, 30.0, 10.0, 15.0, 90.0],
                second_quarter: [440.0, 200.0, 240.0, 145.0, 75.0, 32.0, 12.0, 16.0, 95.0],
            },
            index=[
                "Total Assets",
                "Total Liabilities Net Minority Interest",
                "Stockholders Equity",
                "Current Assets",
                "Current Liabilities",
                "Cash And Cash Equivalents",
                "Inventory",
                "Accounts Receivable",
                "Total Debt",
            ],
        )
        self.quarterly_cashflow = pd.DataFrame(
            {
                first_quarter: [20.0, -5.0, 15.0],
                second_quarter: [22.0, -6.0, 16.0],
            },
            index=["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"],
        )


def test_build_yfinance_quarterly_history_normalizes_statements() -> None:
    history = build_yfinance_quarterly_history("AAA", ticker_obj=_FakeTicker())

    assert set(["ticker", "period_end", "available_date", "revenue", "total_assets", "operating_cash_flow"]) <= set(history.columns)
    assert history["ticker"].tolist() == ["AAA", "AAA"]
    assert history["available_date"].iloc[0] == pd.Timestamp("2024-05-15")
    assert history["revenue"].tolist() == [100.0, 120.0]


def test_build_sec_quarterly_history_parses_companyfacts_filed_dates() -> None:
    companyfacts_payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 100.0,
                                "fy": 2024,
                                "fp": "Q1",
                                "end": "2024-03-31",
                                "filed": "2024-05-02",
                                "form": "10-Q",
                            },
                        ],
                    },
                },
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "val": 400.0,
                                "fy": 2024,
                                "fp": "Q1",
                                "end": "2024-03-31",
                                "filed": "2024-05-02",
                                "form": "10-Q",
                            },
                        ],
                    },
                },
            },
        },
    }

    history = build_sec_quarterly_history("AAA", companyfacts_payload)

    assert history["ticker"].tolist() == ["AAA"]
    assert history["available_date"].iloc[0] == pd.Timestamp("2024-05-02")
    assert history["revenue"].iloc[0] == 100.0
    assert history["total_assets"].iloc[0] == 400.0


def test_combine_accounting_histories_prefers_yfinance_values_and_sec_dates() -> None:
    yfinance_history = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "period_end": [pd.Timestamp("2024-03-31")],
            "available_date": [pd.Timestamp("2024-05-15")],
            "revenue": [120.0],
            "total_assets": [410.0],
            "availability_source": ["yfinance"],
            "value_source": ["yfinance"],
            "used_fallback_source": [False],
        },
    )
    sec_history = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "period_end": [pd.Timestamp("2024-03-31")],
            "available_date": [pd.Timestamp("2024-05-02")],
            "revenue": [100.0],
            "total_assets": [400.0],
            "availability_source": ["sec"],
            "value_source": ["sec"],
            "used_fallback_source": [True],
        },
    )

    combined = combine_accounting_histories(yfinance_history, sec_history)

    assert len(combined) == 1
    assert combined["available_date"].iloc[0] == pd.Timestamp("2024-05-02")
    assert combined["revenue"].iloc[0] == 120.0
    assert combined["total_assets"].iloc[0] == 410.0


def test_history_builder_survives_sec_mapping_failure() -> None:
    sample_history = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "period_end": [pd.Timestamp("2024-03-31")],
            "available_date": [pd.Timestamp("2024-05-15")],
        },
    )

    with (
        patch(
            "core.src.accounting_feature_engineering.history.fetch_sec_ticker_to_cik_mapping",
            side_effect=requests.HTTPError("403 forbidden"),
        ),
        patch(
            "core.src.accounting_feature_engineering.history.fetch_accounting_history_for_ticker",
            return_value=sample_history,
        ),
    ):
        history = build_accounting_history_for_universe(["AAA"], max_workers=1)

    assert history["ticker"].tolist() == ["AAA"]
