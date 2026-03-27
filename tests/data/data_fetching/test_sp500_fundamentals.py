# pyright: reportPrivateUsage=false
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.src.meta_model.data.constants import FUNDAMENTAL_FIELDS
from core.src.meta_model.data.data_fetching.sp500_pipeline import (
    PipelineConfig,
    _camel_to_snake,
    _extract_fundamentals,
    _fetch_all_fundamentals,
    _fetch_ticker_info,
)


# ---------------------------------------------------------------------------
# _camel_to_snake
# ---------------------------------------------------------------------------


class TestCamelToSnake:
    def test_market_cap(self) -> None:
        assert _camel_to_snake("marketCap") == "market_cap"

    def test_trailing_pe(self) -> None:
        assert _camel_to_snake("trailingPE") == "trailing_p_e"

    def test_single_word(self) -> None:
        assert _camel_to_snake("sector") == "sector"

    def test_already_snake(self) -> None:
        assert _camel_to_snake("profit_margins") == "profit_margins"

    def test_multiple_capitals(self) -> None:
        assert _camel_to_snake("returnOnEquity") == "return_on_equity"


# ---------------------------------------------------------------------------
# _extract_fundamentals
# ---------------------------------------------------------------------------


class TestExtractFundamentals:
    def test_full_fields(self) -> None:
        info: dict[str, object] = {
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 3_000_000_000_000,
            "trailingPE": 30.5,
            "forwardPE": 28.0,
            "priceToBook": 45.0,
            "dividendYield": 0.005,
            "beta": 1.2,
            "profitMargins": 0.25,
            "returnOnEquity": 1.5,
            "enterpriseValue": 2_900_000_000_000,
            "revenueGrowth": 0.08,
            "earningsGrowth": 0.10,
            "debtToEquity": 150.0,
            "currentRatio": 1.1,
            "bookValue": 4.0,
            "trailingEps": 6.5,
            "forwardEps": 7.0,
        }
        result: dict[str, object] = _extract_fundamentals(info, FUNDAMENTAL_FIELDS)
        assert result["sector"] == "Technology"
        assert result["market_cap"] == 3_000_000_000_000
        assert len(result) == len(FUNDAMENTAL_FIELDS)

    def test_partial_fields(self) -> None:
        info: dict[str, object] = {"sector": "Technology", "marketCap": 1_000_000}
        result: dict[str, object] = _extract_fundamentals(info, FUNDAMENTAL_FIELDS)
        assert result["sector"] == "Technology"
        assert result["market_cap"] == 1_000_000
        assert result["trailing_p_e"] is None

    def test_empty_info(self) -> None:
        result: dict[str, object] = _extract_fundamentals({}, FUNDAMENTAL_FIELDS)
        assert all(v is None for v in result.values())
        assert len(result) == len(FUNDAMENTAL_FIELDS)


# ---------------------------------------------------------------------------
# _fetch_ticker_info
# ---------------------------------------------------------------------------


class TestFetchTickerInfo:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.yf")
    def test_success(self, mock_yf: MagicMock) -> None:
        mock_ticker: MagicMock = MagicMock()
        mock_ticker.info = {"sector": "Technology", "marketCap": 1_000_000}
        mock_yf.Ticker.return_value = mock_ticker

        result: dict[str, object] = _fetch_ticker_info("AAPL", max_retries=3, retry_sleep=0.0)
        assert result["sector"] == "Technology"
        mock_yf.Ticker.assert_called_once_with("AAPL")

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.yf")
    def test_retry_then_success(self, mock_yf: MagicMock) -> None:
        mock_ticker_ok: MagicMock = MagicMock()
        mock_ticker_ok.info = {"sector": "Tech"}

        call_count: int = 0

        def side_effect(symbol: str) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("network error")
            return mock_ticker_ok

        mock_yf.Ticker.side_effect = side_effect
        result: dict[str, object] = _fetch_ticker_info("AAPL", max_retries=3, retry_sleep=0.0)
        assert result["sector"] == "Tech"

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.yf")
    def test_all_retries_fail(self, mock_yf: MagicMock) -> None:
        mock_yf.Ticker.side_effect = RuntimeError("fail")
        result: dict[str, object] = _fetch_ticker_info("AAPL", max_retries=2, retry_sleep=0.0)
        assert result == {}


# ---------------------------------------------------------------------------
# _fetch_all_fundamentals
# ---------------------------------------------------------------------------


class TestFetchAllFundamentals:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_ticker_info")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.time.sleep")
    def test_basic_flow(
        self,
        mock_sleep: MagicMock,
        mock_fetch_info: MagicMock,
    ) -> None:
        mock_fetch_info.return_value = {
            "sector": "Technology",
            "marketCap": 3_000_000_000_000,
        }

        config: PipelineConfig = PipelineConfig()
        result: pd.DataFrame = _fetch_all_fundamentals(["AAPL", "MSFT"], config)

        assert "ticker" in result.columns
        assert "sector" in result.columns
        assert "market_cap" in result.columns
        assert len(result) == 2
        assert set(result["ticker"]) == {"AAPL", "MSFT"}


# --- Execution directe (clic dans l'IDE) ---
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
