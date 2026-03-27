from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import datetime as dt
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from core.src.meta_model.data.data_fetching.sp500_pipeline import (
    OHLCV_OUTPUT_COLS,
    PipelineConfig,
    _fetch_prices,
    _fetch_stooq_symbol,
    _fetch_tiingo_symbol,
    _fetch_yfinance_batch,
    _flatten_columns,
    _load_aliases,
    _load_constituents_from_csv,
    _parse_date,
    _rename_ohlcv_columns,
    _resolve_tiingo_api_key,
    _stooq_symbol,
    _yfinance_symbol,
    load_constituents_table_from_wikipedia,
    build_dataset,
    load_constituents,
    run_pipeline,
    save_outputs,
)
from core.src.meta_model.data.paths import DATA_FETCHING_DIR


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_string_input(self) -> None:
        result: dt.date = _parse_date("2020-06-15")
        assert result == dt.date(2020, 6, 15)

    def test_date_input(self) -> None:
        d: dt.date = dt.date(2023, 1, 1)
        assert _parse_date(d) is d

    def test_datetime_input(self) -> None:
        result: dt.date = _parse_date(dt.datetime(2023, 3, 10, 14, 30))
        assert result == dt.date(2023, 3, 10)

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_date("15/06/2020")


# ---------------------------------------------------------------------------
# _flatten_columns
# ---------------------------------------------------------------------------


class TestFlattenColumns:
    def test_multiindex_flattened(self) -> None:
        arrays: list[list[str]] = [["Close", "Close"], ["AAPL", "MSFT"]]
        tuples: list[tuple[str, str]] = list(zip(*arrays))
        index: pd.MultiIndex = pd.MultiIndex.from_tuples(tuples)
        df: pd.DataFrame = pd.DataFrame([[1, 2]], columns=index)
        result: pd.DataFrame = _flatten_columns(df)
        assert list(result.columns) == ["Close AAPL", "Close MSFT"]

    def test_regular_columns_unchanged(self) -> None:
        df: pd.DataFrame = pd.DataFrame({"a": [1], "b": [2]})
        result: pd.DataFrame = _flatten_columns(df)
        assert list(result.columns) == ["a", "b"]



# ---------------------------------------------------------------------------
# _rename_ohlcv_columns
# ---------------------------------------------------------------------------


class TestRenameOhlcvColumns:
    def test_renames_all_columns(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Adj Close": [1.4],
            "Volume": [1000],
        })
        result: pd.DataFrame = _rename_ohlcv_columns(df)
        assert list(result.columns) == OHLCV_OUTPUT_COLS

    def test_drops_extra_columns(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Adj Close": [1.4],
            "Volume": [1000],
            "Extra": [999],
        })
        result: pd.DataFrame = _rename_ohlcv_columns(df)
        assert "Extra" not in result.columns
        assert "extra" not in result.columns

    def test_partial_columns(self) -> None:
        df: pd.DataFrame = pd.DataFrame({"Close": [1.5], "Volume": [1000]})
        result: pd.DataFrame = _rename_ohlcv_columns(df)
        assert list(result.columns) == ["close", "volume"]

    def test_drops_all_nan_rows(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "Open": [1.0, np.nan],
            "High": [2.0, np.nan],
            "Low": [0.5, np.nan],
            "Close": [1.5, np.nan],
            "Adj Close": [1.4, np.nan],
            "Volume": [1000, np.nan],
        })
        result: pd.DataFrame = _rename_ohlcv_columns(df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _load_aliases
# ---------------------------------------------------------------------------


class TestLoadAliases:
    def test_none_path_returns_empty(self) -> None:
        assert _load_aliases(None) == {}

    def test_missing_file_returns_empty(self) -> None:
        assert _load_aliases(Path("/nonexistent/file.csv")) == {}

    def test_valid_csv(self, tmp_path: Path) -> None:
        csv_path: Path = tmp_path / "aliases.csv"
        csv_path.write_text("symbol,provider_symbol\nBRK.B,BRK-B\nBF.B,BF-B\n")
        result: dict[str, str] = _load_aliases(csv_path)
        assert result == {"BRK.B": "BRK-B", "BF.B": "BF-B"}

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        csv_path: Path = tmp_path / "bad.csv"
        csv_path.write_text("col1,col2\na,b\n")
        with pytest.raises(ValueError, match="symbol"):
            _load_aliases(csv_path)


# ---------------------------------------------------------------------------
# _load_constituents_from_csv
# ---------------------------------------------------------------------------


class TestLoadConstituentsFromCsv:
    def test_valid_csv(self, tmp_path: Path) -> None:
        csv_path: Path = tmp_path / "constituents.csv"
        csv_path.write_text("symbol\nAAPL\nMSFT\nGOOG\n")
        result: set[str] = _load_constituents_from_csv(csv_path)
        assert result == {"AAPL", "MSFT", "GOOG"}

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        csv_path: Path = tmp_path / "constituents.csv"
        csv_path.write_text("symbol\n AAPL \n MSFT \n")
        result: set[str] = _load_constituents_from_csv(csv_path)
        assert result == {"AAPL", "MSFT"}

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        csv_path: Path = tmp_path / "bad.csv"
        csv_path.write_text("ticker\nAAPL\n")
        with pytest.raises(ValueError, match="symbol"):
            _load_constituents_from_csv(csv_path)


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------


class TestSymbolHelpers:
    def test_yfinance_symbol_dot_to_dash(self) -> None:
        assert _yfinance_symbol("BRK.B", {}) == "BRK-B"

    def test_yfinance_symbol_with_alias(self) -> None:
        assert _yfinance_symbol("BRK.B", {"BRK.B": "BRK-B"}) == "BRK-B"

    def test_yfinance_symbol_no_dot(self) -> None:
        assert _yfinance_symbol("AAPL", {}) == "AAPL"

    def test_stooq_symbol_default(self) -> None:
        assert _stooq_symbol("AAPL", {}) == "aapl.us"

    def test_stooq_symbol_with_alias(self) -> None:
        assert _stooq_symbol("BRK.B", {"BRK.B": "BRKB"}) == "brkb.us"


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_default_values(self) -> None:
        config: PipelineConfig = PipelineConfig()
        assert config.start_date == "2004-01-01"
        assert config.end_date == "2025-12-31"
        assert config.output_dir == DATA_FETCHING_DIR
        assert config.sample_frac == 0.05
        assert config.random_seed == 7
        assert config.chunk_size == 50
        assert config.max_retries == 3
        assert config.retry_sleep == 2.0
        assert config.use_stooq_fallback is True
        assert config.use_tiingo_fallback is True
        assert config.constituents_csv is None
        assert config.ticker_aliases_csv is None
        assert config.membership_history_csv is None
        assert config.fundamentals_history_csv is None
        assert config.allow_current_constituents_snapshot is True

    def test_frozen(self) -> None:
        config: PipelineConfig = PipelineConfig()
        with pytest.raises(AttributeError):
            config.start_date = "2010-01-01"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_constituents (via CSV, mocked)
# ---------------------------------------------------------------------------


class TestLoadConstituents:
    def test_uses_csv_when_provided(self, tmp_path: Path) -> None:
        csv_path: Path = tmp_path / "constituents.csv"
        csv_path.write_text("symbol\nAAPL\nMSFT\n")
        config: PipelineConfig = PipelineConfig(constituents_csv=csv_path)
        result: set[str] = load_constituents(config)
        assert result == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# save_outputs
# ---------------------------------------------------------------------------


class TestSaveOutputs:
    def _make_ohlcv_data(self, periods: int = 10) -> pd.DataFrame:
        rng: np.random.Generator = np.random.default_rng(42)
        return pd.DataFrame({
            "date": pd.bdate_range("2020-01-06", periods=periods),
            "ticker": ["AAPL"] * periods,
            "open": rng.random(periods) * 100,
            "high": rng.random(periods) * 100,
            "low": rng.random(periods) * 100,
            "close": rng.random(periods) * 100,
            "adj_close": rng.random(periods) * 100,
            "volume": rng.integers(1000, 10000, size=periods),
        })

    def test_creates_files(self, tmp_path: Path) -> None:
        data: pd.DataFrame = self._make_ohlcv_data()
        config: PipelineConfig = PipelineConfig(output_dir=tmp_path)
        paths: dict[str, Path] = save_outputs(data, config)

        assert paths["parquet"].exists()
        assert paths["sample_csv"].exists()

    def test_sample_fraction(self, tmp_path: Path) -> None:
        data: pd.DataFrame = self._make_ohlcv_data(periods=100)
        config: PipelineConfig = PipelineConfig(output_dir=tmp_path, sample_frac=0.1)
        paths: dict[str, Path] = save_outputs(data, config)

        sample: pd.DataFrame = pd.read_csv(paths["sample_csv"])
        assert len(sample) == 10

    def test_parquet_roundtrip_columns(self, tmp_path: Path) -> None:
        data: pd.DataFrame = self._make_ohlcv_data()
        config: PipelineConfig = PipelineConfig(output_dir=tmp_path)
        paths: dict[str, Path] = save_outputs(data, config)

        loaded: pd.DataFrame = pd.read_parquet(paths["parquet"])
        expected_cols: list[str] = [
            "date", "ticker", "open", "high", "low", "close", "adj_close", "volume",
        ]
        assert list(loaded.columns) == expected_cols


# ---------------------------------------------------------------------------
# build_dataset (mocked)
# ---------------------------------------------------------------------------


def _make_mock_ohlcv_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    n: int = len(index)
    rng: np.random.Generator = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "open": rng.random(n) * 100 + 100,
            "high": rng.random(n) * 100 + 110,
            "low": rng.random(n) * 100 + 90,
            "close": rng.random(n) * 100 + 100,
            "adj_close": rng.random(n) * 100 + 100,
            "volume": rng.integers(1000, 10000, size=n),
        },
        index=index,
    )


class TestBuildDataset:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_all_fundamentals")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.load_constituents")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_prices")
    def test_basic_flow(
        self,
        mock_fetch: MagicMock,
        mock_constituents: MagicMock,
        mock_fundamentals: MagicMock,
    ) -> None:
        mock_constituents.return_value = {"AAPL", "MSFT"}

        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        mock_fetch.return_value = {
            "AAPL": _make_mock_ohlcv_df(index),
            "MSFT": _make_mock_ohlcv_df(index),
        }
        mock_fundamentals.return_value = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "sector": ["Technology", "Technology"],
        })

        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            allow_current_constituents_snapshot=True,
        )
        result: pd.DataFrame = build_dataset(config)

        assert "date" in result.columns
        assert "ticker" in result.columns
        assert "adj_close" in result.columns
        assert "sector" not in result.columns
        assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}
        assert len(result) == 10
        assert list(result["ticker"]) == ["AAPL", "MSFT"] * 5

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_all_fundamentals")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.load_constituents")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_prices")
    def test_no_data_raises(
        self,
        mock_fetch: MagicMock,
        mock_constituents: MagicMock,
        mock_fundamentals: MagicMock,
    ) -> None:
        mock_constituents.return_value = {"AAPL"}
        mock_fetch.return_value = {}

        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            allow_current_constituents_snapshot=True,
        )
        with pytest.raises(RuntimeError, match="No price data"):
            build_dataset(config)

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.load_constituents")
    def test_no_constituents_raises(self, mock_constituents: MagicMock) -> None:
        mock_constituents.return_value = set()
        config: PipelineConfig = PipelineConfig(
            allow_current_constituents_snapshot=True,
        )
        with pytest.raises(RuntimeError, match="No constituents"):
            build_dataset(config)


# ---------------------------------------------------------------------------
# _fetch_ticker_info -- yfinance not installed
# ---------------------------------------------------------------------------


class TestFetchTickerInfoYfNone:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.yf", None)
    def test_yf_none_raises(self) -> None:
        from core.src.meta_model.data.data_fetching.sp500_pipeline import _fetch_ticker_info
        with pytest.raises(ImportError, match="yfinance is not installed"):
            _fetch_ticker_info("AAPL", max_retries=1, retry_sleep=0.0)


# ---------------------------------------------------------------------------
# load_constituents_table_from_wikipedia (mocked)
# ---------------------------------------------------------------------------


class TestLoadConstituentsFromWikipedia:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.pd.read_html")
    def test_basic_flow(
        self, mock_read_html: MagicMock, mock_get: MagicMock,
    ) -> None:
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        current_table: pd.DataFrame = pd.DataFrame({
            "Symbol": ["AAPL", "MSFT", "GOOG"],
            "Security": ["Apple", "Microsoft", "Alphabet"],
        })
        mock_read_html.return_value = [current_table]

        result: pd.DataFrame = load_constituents_table_from_wikipedia()
        assert list(result["ticker"]) == ["AAPL", "MSFT", "GOOG"]
        assert list(result["company_name"]) == ["Apple", "Microsoft", "Alphabet"]

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.pd.read_html")
    def test_no_symbol_table_raises(
        self, mock_read_html: MagicMock, mock_get: MagicMock,
    ) -> None:
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = "<html></html>"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        mock_read_html.return_value = [
            pd.DataFrame({"col_a": [1]}),
        ]

        with pytest.raises(RuntimeError, match="Could not locate"):
            load_constituents_table_from_wikipedia()


# ---------------------------------------------------------------------------
# load_constituents -- Wikipedia path
# ---------------------------------------------------------------------------


class TestLoadConstituentsWikipedia:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.load_constituents_table_from_wikipedia")
    def test_calls_wikipedia_when_no_csv(self, mock_wiki: MagicMock) -> None:
        mock_wiki.return_value = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "company_name": ["Apple", "Microsoft"],
        })
        config: PipelineConfig = PipelineConfig()
        result: set[str] = load_constituents(config)
        assert result == {"AAPL", "MSFT"}
        mock_wiki.assert_called_once()


# ---------------------------------------------------------------------------
# _fetch_yfinance_batch
# ---------------------------------------------------------------------------


class TestFetchYfinanceBatch:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.yf")
    def test_multiindex_result(self, mock_yf: MagicMock) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        n: int = len(index)
        arrays: dict[tuple[str, str], list[float]] = {
            ("Open", "AAPL"): [1.0] * n,
            ("High", "AAPL"): [2.0] * n,
            ("Low", "AAPL"): [0.5] * n,
            ("Close", "AAPL"): [1.5] * n,
            ("Adj Close", "AAPL"): [1.4] * n,
            ("Volume", "AAPL"): [1000.0] * n,
        }
        columns: pd.MultiIndex = pd.MultiIndex.from_tuples(list(arrays.keys()))
        data: np.ndarray = np.column_stack(list(arrays.values()))
        raw: pd.DataFrame = pd.DataFrame(data, index=index, columns=columns)
        mock_yf.download.return_value = raw

        result: dict[str, pd.DataFrame] = _fetch_yfinance_batch(
            ["AAPL"], "2020-01-06", "2020-01-10",
        )
        assert "AAPL" in result
        assert "adj_close" in result["AAPL"].columns

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.yf")
    def test_single_symbol_flat_columns(self, mock_yf: MagicMock) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        n: int = len(index)
        raw: pd.DataFrame = pd.DataFrame({
            "Open": [1.0] * n,
            "High": [2.0] * n,
            "Low": [0.5] * n,
            "Close": [1.5] * n,
            "Adj Close": [1.4] * n,
            "Volume": [1000.0] * n,
        }, index=index)
        mock_yf.download.return_value = raw

        result: dict[str, pd.DataFrame] = _fetch_yfinance_batch(
            ["AAPL"], "2020-01-06", "2020-01-10",
        )
        assert "AAPL" in result

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.yf")
    def test_empty_result(self, mock_yf: MagicMock) -> None:
        mock_yf.download.return_value = pd.DataFrame()
        result: dict[str, pd.DataFrame] = _fetch_yfinance_batch(
            ["AAPL"], "2020-01-06", "2020-01-10",
        )
        assert result == {}

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.yf")
    def test_none_result(self, mock_yf: MagicMock) -> None:
        mock_yf.download.return_value = None
        result: dict[str, pd.DataFrame] = _fetch_yfinance_batch(
            ["AAPL"], "2020-01-06", "2020-01-10",
        )
        assert result == {}

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.yf", None)
    def test_yf_not_installed(self) -> None:
        with pytest.raises(ImportError, match="yfinance is not installed"):
            _fetch_yfinance_batch(["AAPL"], "2020-01-06", "2020-01-10")

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.yf")
    def test_missing_symbol_in_multiindex(self, mock_yf: MagicMock) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        n: int = len(index)
        arrays: dict[tuple[str, str], list[float]] = {
            ("Adj Close", "AAPL"): [1.4] * n,
            ("Volume", "AAPL"): [1000.0] * n,
        }
        columns: pd.MultiIndex = pd.MultiIndex.from_tuples(list(arrays.keys()))
        data: np.ndarray = np.column_stack(list(arrays.values()))
        raw: pd.DataFrame = pd.DataFrame(data, index=index, columns=columns)
        mock_yf.download.return_value = raw

        result: dict[str, pd.DataFrame] = _fetch_yfinance_batch(
            ["AAPL", "MSFT"], "2020-01-06", "2020-01-10",
        )
        assert "AAPL" in result
        assert "MSFT" not in result

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.yf")
    def test_empty_single_symbol(self, mock_yf: MagicMock) -> None:
        """Flat columns with all NaN should produce empty result."""
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        n: int = len(index)
        raw: pd.DataFrame = pd.DataFrame({
            "Open": [np.nan] * n,
            "High": [np.nan] * n,
            "Low": [np.nan] * n,
            "Close": [np.nan] * n,
            "Adj Close": [np.nan] * n,
            "Volume": [np.nan] * n,
        }, index=index)
        mock_yf.download.return_value = raw

        result: dict[str, pd.DataFrame] = _fetch_yfinance_batch(
            ["AAPL"], "2020-01-06", "2020-01-10",
        )
        assert result == {}


# ---------------------------------------------------------------------------
# _fetch_stooq_symbol
# ---------------------------------------------------------------------------


class TestFetchStooqSymbol:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    def test_success(self, mock_get: MagicMock) -> None:
        csv_text: str = (
            "Date,Open,High,Low,Close,Volume\n"
            "2020-01-06,100,105,95,102,1000\n"
            "2020-01-07,101,106,96,103,1100\n"
        )
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = csv_text
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result: pd.DataFrame | None = _fetch_stooq_symbol(
            "aapl.us", "2020-01-06", "2020-01-10",
        )
        assert result is not None
        assert "close" in result.columns
        # adj_close should be created from close when missing
        assert "adj_close" in result.columns

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    def test_no_data_text(self, mock_get: MagicMock) -> None:
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = "No data"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result: pd.DataFrame | None = _fetch_stooq_symbol(
            "aapl.us", "2020-01-06", "2020-01-10",
        )
        assert result is None

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    def test_single_line_response(self, mock_get: MagicMock) -> None:
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = "Date,Close\n"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result: pd.DataFrame | None = _fetch_stooq_symbol(
            "aapl.us", "2020-01-06", "2020-01-10",
        )
        assert result is None

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    def test_empty_csv(self, mock_get: MagicMock) -> None:
        csv_text: str = "Date,Open,High,Low,Volume\n"
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = csv_text
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result: pd.DataFrame | None = _fetch_stooq_symbol(
            "aapl.us", "2020-01-06", "2020-01-10",
        )
        assert result is None

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    def test_all_nan_returns_none(self, mock_get: MagicMock) -> None:
        csv_text: str = (
            "Date,Open,High,Low,Close,Volume\n"
            "2020-01-06,,,,,\n"
        )
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = csv_text
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result: pd.DataFrame | None = _fetch_stooq_symbol(
            "aapl.us", "2020-01-06", "2020-01-10",
        )
        assert result is None

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    def test_no_close_column_returns_none(self, mock_get: MagicMock) -> None:
        """CSV has data but no 'Close' column."""
        csv_text: str = (
            "Date,Open,High,Low,Volume\n"
            "2020-01-06,100,105,95,1000\n"
        )
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = csv_text
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result: pd.DataFrame | None = _fetch_stooq_symbol(
            "aapl.us", "2020-01-06", "2020-01-10",
        )
        assert result is None

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    def test_with_adj_close_column(self, mock_get: MagicMock) -> None:
        csv_text: str = (
            "Date,Open,High,Low,Close,Adj Close,Volume\n"
            "2020-01-06,100,105,95,102,101,1000\n"
        )
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = csv_text
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result: pd.DataFrame | None = _fetch_stooq_symbol(
            "aapl.us", "2020-01-06", "2020-01-10",
        )
        assert result is not None
        assert "adj_close" in result.columns


# ---------------------------------------------------------------------------
# _resolve_tiingo_api_key
# ---------------------------------------------------------------------------


class TestResolveTiingoApiKey:
    def test_returns_key_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIINGO_API_KEY", "test-key-123")
        assert _resolve_tiingo_api_key() == "test-key-123"

    def test_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TIINGO_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TIINGO_API_KEY"):
            _resolve_tiingo_api_key()


# ---------------------------------------------------------------------------
# _fetch_tiingo_symbol
# ---------------------------------------------------------------------------


class TestFetchTiingoSymbol:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    def test_success(self, mock_get: MagicMock) -> None:
        csv_text: str = (
            "date,close,high,low,open,volume,adjClose,adjHigh,adjLow,adjOpen,adjVolume,divCash,splitFactor\n"
            "2020-01-06,102,105,95,100,1000,101,104,94,99,1000,0,1\n"
            "2020-01-07,103,106,96,101,1100,102,105,95,100,1100,0,1\n"
        )
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = csv_text
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result: pd.DataFrame | None = _fetch_tiingo_symbol(
            "AAPL", "2020-01-06", "2020-01-10", "fake-key",
        )
        assert result is not None
        assert not result.columns.duplicated().any()
        assert "adj_close" in result.columns
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert len(result) == 2
        # Values should come from adjusted columns, not raw
        assert result["adj_close"].iloc[0] == 101
        assert result["open"].iloc[0] == 99

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    def test_empty_response_returns_none(self, mock_get: MagicMock) -> None:
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = ""
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result: pd.DataFrame | None = _fetch_tiingo_symbol(
            "AAPL", "2020-01-06", "2020-01-10", "fake-key",
        )
        assert result is None

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    def test_http_error_raises(self, mock_get: MagicMock) -> None:
        mock_resp: MagicMock = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get.return_value = mock_resp

        with pytest.raises(requests.HTTPError):
            _fetch_tiingo_symbol("AAPL", "2020-01-06", "2020-01-10", "fake-key")

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.requests.get")
    def test_header_only_returns_none(self, mock_get: MagicMock) -> None:
        mock_resp: MagicMock = MagicMock()
        mock_resp.text = "date,close,adjClose\n"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result: pd.DataFrame | None = _fetch_tiingo_symbol(
            "AAPL", "2020-01-06", "2020-01-10", "fake-key",
        )
        assert result is None


# ---------------------------------------------------------------------------
# _fetch_prices
# ---------------------------------------------------------------------------


class TestFetchPrices:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_yfinance_batch")
    def test_yfinance_success_no_fallback(self, mock_batch: MagicMock) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        mock_batch.return_value = {
            "AAPL": _make_mock_ohlcv_df(index),
        }
        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            use_stooq_fallback=False,
        )
        result: dict[str, pd.DataFrame] = _fetch_prices(
            ["AAPL"], "2020-01-06", "2020-01-10", config,
        )
        assert "AAPL" in result

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_stooq_symbol")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_yfinance_batch")
    def test_stooq_fallback(
        self, mock_batch: MagicMock, mock_stooq: MagicMock,
    ) -> None:
        mock_batch.return_value = {}
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        mock_stooq.return_value = _make_mock_ohlcv_df(index)

        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            use_stooq_fallback=True,
            retry_sleep=0.0,
        )
        result: dict[str, pd.DataFrame] = _fetch_prices(
            ["AAPL"], "2020-01-06", "2020-01-10", config,
        )
        assert "AAPL" in result
        mock_stooq.assert_called_once()

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_stooq_symbol")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_yfinance_batch")
    def test_stooq_fallback_returns_none(
        self, mock_batch: MagicMock, mock_stooq: MagicMock,
    ) -> None:
        mock_batch.return_value = {}
        mock_stooq.return_value = None

        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            use_stooq_fallback=True,
            use_tiingo_fallback=False,
            retry_sleep=0.0,
        )
        result: dict[str, pd.DataFrame] = _fetch_prices(
            ["AAPL"], "2020-01-06", "2020-01-10", config,
        )
        assert "AAPL" not in result

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.time.sleep")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_yfinance_batch")
    def test_yfinance_retry_on_failure(
        self, mock_batch: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        mock_batch.side_effect = [
            RuntimeError("network error"),
            {"AAPL": _make_mock_ohlcv_df(index)},
        ]
        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            use_stooq_fallback=False,
            retry_sleep=0.01,
        )
        result: dict[str, pd.DataFrame] = _fetch_prices(
            ["AAPL"], "2020-01-06", "2020-01-10", config,
        )
        assert "AAPL" in result
        assert mock_batch.call_count == 2

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.time.sleep")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_yfinance_batch")
    def test_yfinance_all_retries_fail(
        self, mock_batch: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_batch.side_effect = RuntimeError("network error")
        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            use_stooq_fallback=False,
            max_retries=2,
            retry_sleep=0.01,
        )
        result: dict[str, pd.DataFrame] = _fetch_prices(
            ["AAPL"], "2020-01-06", "2020-01-10", config,
        )
        assert result == {}

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_yfinance_batch")
    def test_no_missing_skips_stooq(self, mock_batch: MagicMock) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        mock_batch.return_value = {
            "AAPL": _make_mock_ohlcv_df(index),
        }
        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            use_stooq_fallback=True,
        )
        result: dict[str, pd.DataFrame] = _fetch_prices(
            ["AAPL"], "2020-01-06", "2020-01-10", config,
        )
        assert "AAPL" in result

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.time.sleep")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_stooq_symbol")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_yfinance_batch")
    def test_stooq_retry_on_failure(
        self, mock_batch: MagicMock, mock_stooq: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_batch.return_value = {}
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        mock_stooq.side_effect = [
            RuntimeError("network error"),
            _make_mock_ohlcv_df(index),
        ]
        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            use_stooq_fallback=True,
            retry_sleep=0.01,
        )
        result: dict[str, pd.DataFrame] = _fetch_prices(
            ["AAPL"], "2020-01-06", "2020-01-10", config,
        )
        assert "AAPL" in result
        assert mock_stooq.call_count == 2

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_yfinance_batch")
    def test_chunked_fetch(self, mock_batch: MagicMock) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        mock_batch.return_value = {
            "SYM": _make_mock_ohlcv_df(index),
        }
        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            chunk_size=1,
            use_stooq_fallback=False,
        )
        symbols: list[str] = ["SYM1", "SYM2"]
        # Mock returns one symbol per call
        def _side_effect(
            batch: list[str], start: str, end: str,
        ) -> dict[str, pd.DataFrame]:
            return {batch[0]: _make_mock_ohlcv_df(index)}

        mock_batch.side_effect = _side_effect
        _result: dict[str, pd.DataFrame] = _fetch_prices(
            symbols, "2020-01-06", "2020-01-10", config,
        )
        assert mock_batch.call_count == 2

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._resolve_tiingo_api_key")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_tiingo_symbol")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_stooq_symbol")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_yfinance_batch")
    def test_tiingo_fallback_recovers_ticker(
        self,
        mock_batch: MagicMock,
        mock_stooq: MagicMock,
        mock_tiingo: MagicMock,
        mock_key: MagicMock,
    ) -> None:
        mock_batch.return_value = {}
        mock_stooq.return_value = None
        mock_key.return_value = "fake-key"
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        mock_tiingo.return_value = _make_mock_ohlcv_df(index)

        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            use_stooq_fallback=True,
            use_tiingo_fallback=True,
            retry_sleep=0.0,
        )
        result: dict[str, pd.DataFrame] = _fetch_prices(
            ["AAPL"], "2020-01-06", "2020-01-10", config,
        )
        assert "AAPL" in result
        mock_tiingo.assert_called_once()

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_stooq_symbol")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_yfinance_batch")
    def test_tiingo_disabled_skips_fallback(
        self,
        mock_batch: MagicMock,
        mock_stooq: MagicMock,
    ) -> None:
        mock_batch.return_value = {}
        mock_stooq.return_value = None

        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            use_stooq_fallback=True,
            use_tiingo_fallback=False,
            retry_sleep=0.0,
        )
        result: dict[str, pd.DataFrame] = _fetch_prices(
            ["AAPL"], "2020-01-06", "2020-01-10", config,
        )
        assert "AAPL" not in result

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_yfinance_batch")
    def test_alias_lookup(self, mock_batch: MagicMock, tmp_path: Path) -> None:
        csv_path: Path = tmp_path / "aliases.csv"
        csv_path.write_text("symbol,provider_symbol\nBRK.B,BRK-B\n")
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")

        def _side_effect(
            batch: list[str], start: str, end: str,
        ) -> dict[str, pd.DataFrame]:
            return {batch[0]: _make_mock_ohlcv_df(index)}

        mock_batch.side_effect = _side_effect
        config: PipelineConfig = PipelineConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            use_stooq_fallback=False,
            ticker_aliases_csv=csv_path,
        )
        result: dict[str, pd.DataFrame] = _fetch_prices(
            ["BRK.B"], "2020-01-06", "2020-01-10", config,
        )
        # The original symbol should be in the result, not the alias
        assert "BRK.B" in result


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.save_outputs")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.build_dataset")
    def test_calls_build_and_save(
        self, mock_build: MagicMock, mock_save: MagicMock, tmp_path: Path,
    ) -> None:
        mock_build.return_value = pd.DataFrame({
            "date": pd.bdate_range("2020-01-06", periods=5),
            "ticker": ["AAPL"] * 5,
            "adj_close": [100.0] * 5,
        })
        mock_save.return_value = {"parquet": tmp_path / "out.parquet", "sample_csv": tmp_path / "out.csv"}

        config: PipelineConfig = PipelineConfig()
        result: dict[str, Path] = run_pipeline(config)

        mock_build.assert_called_once_with(config)
        mock_save.assert_called_once()
        assert "parquet" in result


# --- Execution directe (clic dans l'IDE) ---
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
