# pyright: reportPrivateUsage=false
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.src.meta_model.data.data_fetching.cross_asset_pipeline import (
    ALL_CROSS_ASSET_SYMBOLS,
    CrossAssetConfig,
    _extract_symbol_df,
    _fetch_all_cross_assets,
    _fetch_cross_asset_batch,
    _fetch_cross_asset_chunk_with_retry,
    build_cross_asset_dataset,
)
from core.src.meta_model.data.paths import DATA_FETCHING_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_cross_asset_df(
    index: pd.DatetimeIndex, symbols: list[str],
) -> pd.DataFrame:
    rng: np.random.Generator = np.random.default_rng(42)
    n: int = len(index)
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for symbol in symbols:
        arrays[("Adj Close", symbol)] = rng.random(n) * 100 + 50
        arrays[("Volume", symbol)] = rng.integers(1000, 100000, size=n).astype(float)
    columns: pd.MultiIndex = pd.MultiIndex.from_tuples(list(arrays.keys()))
    data: np.ndarray = np.column_stack(list(arrays.values()))
    return pd.DataFrame(data, index=index, columns=columns)


def _make_single_symbol_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    rng: np.random.Generator = np.random.default_rng(42)
    n: int = len(index)
    return pd.DataFrame(
        {
            "Adj Close": rng.random(n) * 100 + 50,
            "Volume": rng.integers(1000, 100000, size=n).astype(float),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# CrossAssetConfig
# ---------------------------------------------------------------------------


class TestCrossAssetConfig:
    def test_default_values(self) -> None:
        config: CrossAssetConfig = CrossAssetConfig()
        assert config.start_date == "2004-01-01"
        assert config.end_date == "2025-12-31"
        assert config.output_dir == DATA_FETCHING_DIR
        assert config.sample_frac == 0.05
        assert config.random_seed == 7
        assert config.chunk_size == 50
        assert config.max_retries == 3
        assert config.retry_sleep == 2.0

    def test_frozen(self) -> None:
        config: CrossAssetConfig = CrossAssetConfig()
        with pytest.raises(AttributeError):
            config.start_date = "2010-01-01"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _fetch_cross_asset_batch
# ---------------------------------------------------------------------------


class TestFetchCrossAssetBatch:
    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline.yf")
    def test_success_multi_symbol(self, mock_yf: MagicMock) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        symbols: list[str] = ["^N225", "XLK"]
        mock_yf.download.return_value = _make_mock_cross_asset_df(index, symbols)

        result: dict[str, pd.DataFrame] = _fetch_cross_asset_batch(
            symbols, "2020-01-06", "2020-01-10",
        )

        assert set(result.keys()) == {"^N225", "XLK"}
        for symbol in symbols:
            assert "adj_close" in result[symbol].columns
            assert "volume" in result[symbol].columns
            assert len(result[symbol]) == 5

    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline.yf")
    def test_empty_result(self, mock_yf: MagicMock) -> None:
        mock_yf.download.return_value = pd.DataFrame()

        result: dict[str, pd.DataFrame] = _fetch_cross_asset_batch(
            ["^N225"], "2020-01-06", "2020-01-10",
        )
        assert result == {}

    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline.yf")
    def test_none_result(self, mock_yf: MagicMock) -> None:
        mock_yf.download.return_value = None

        result: dict[str, pd.DataFrame] = _fetch_cross_asset_batch(
            ["^N225"], "2020-01-06", "2020-01-10",
        )
        assert result == {}


# ---------------------------------------------------------------------------
# _fetch_all_cross_assets
# ---------------------------------------------------------------------------


class TestFetchAllCrossAssets:
    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline._fetch_cross_asset_batch")
    def test_chunked_fetch(self, mock_batch: MagicMock) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        rng: np.random.Generator = np.random.default_rng(42)
        n: int = len(index)

        def _side_effect(
            symbols: list[str], start: str, end: str,
        ) -> dict[str, pd.DataFrame]:
            result: dict[str, pd.DataFrame] = {}
            for sym in symbols:
                result[sym] = pd.DataFrame(
                    {
                        "adj_close": rng.random(n) * 100,
                        "volume": rng.integers(1000, 10000, size=n).astype(float),
                    },
                    index=index,
                )
            return result

        mock_batch.side_effect = _side_effect

        symbols: tuple[str, ...] = ("SYM1", "SYM2", "SYM3")
        config: CrossAssetConfig = CrossAssetConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            chunk_size=2,
        )
        result: dict[str, pd.DataFrame] = _fetch_all_cross_assets(symbols, config)

        assert mock_batch.call_count == 2
        assert set(result.keys()) == {"SYM1", "SYM2", "SYM3"}

    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline._fetch_cross_asset_batch")
    def test_retry_on_failure(self, mock_batch: MagicMock) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        rng: np.random.Generator = np.random.default_rng(42)
        n: int = len(index)

        call_count: list[int] = [0]

        def _side_effect(
            symbols: list[str], start: str, end: str,
        ) -> dict[str, pd.DataFrame]:
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Network error")
            return {
                symbols[0]: pd.DataFrame(
                    {
                        "adj_close": rng.random(n) * 100,
                        "volume": rng.integers(1000, 10000, size=n).astype(float),
                    },
                    index=index,
                ),
            }

        mock_batch.side_effect = _side_effect

        config: CrossAssetConfig = CrossAssetConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            retry_sleep=0.0,
        )
        result: dict[str, pd.DataFrame] = _fetch_all_cross_assets(
            ("^N225",), config,
        )

        assert mock_batch.call_count == 2
        assert "^N225" in result


# ---------------------------------------------------------------------------
# build_cross_asset_dataset
# ---------------------------------------------------------------------------


class TestBuildCrossAssetDataset:
    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline._fetch_all_cross_assets")
    def test_full_flow(self, mock_fetch: MagicMock) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        rng: np.random.Generator = np.random.default_rng(42)
        n: int = len(index)

        price_map: dict[str, pd.DataFrame] = {}
        for symbol in ALL_CROSS_ASSET_SYMBOLS[:3]:
            price_map[symbol] = pd.DataFrame(
                {
                    "adj_close": rng.random(n) * 100 + 50,
                    "volume": rng.integers(1000, 100000, size=n).astype(float),
                },
                index=index,
            )
        mock_fetch.return_value = price_map

        config: CrossAssetConfig = CrossAssetConfig(
            start_date="2020-01-06", end_date="2020-01-10",
        )
        result: pd.DataFrame = build_cross_asset_dataset(config)

        assert "date" in result.columns
        assert "ticker" in result.columns
        assert "adj_close" in result.columns
        assert "volume" in result.columns
        assert set(result["ticker"].unique()) == set(ALL_CROSS_ASSET_SYMBOLS[:3])
        assert len(result) == 15

    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline._fetch_all_cross_assets")
    def test_no_data_raises(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {}

        config: CrossAssetConfig = CrossAssetConfig(
            start_date="2020-01-06", end_date="2020-01-10",
        )
        with pytest.raises(RuntimeError, match="No cross-asset data"):
            build_cross_asset_dataset(config)

    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline._fetch_all_cross_assets")
    def test_all_empty_after_reindex_raises(self, mock_fetch: MagicMock) -> None:
        """When price_map has entries but all are empty DataFrames, raises."""
        mock_fetch.return_value = {
            ALL_CROSS_ASSET_SYMBOLS[0]: pd.DataFrame(),
        }
        config: CrossAssetConfig = CrossAssetConfig(
            start_date="2020-01-06", end_date="2020-01-10",
        )
        with pytest.raises(RuntimeError, match="No cross-asset data"):
            build_cross_asset_dataset(config)

    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline._fetch_all_cross_assets")
    def test_ffill_after_reindex(self, mock_fetch: MagicMock) -> None:
        """Gaps from reindex should be forward-filled, not left as NaN."""
        full_index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        # Only provide data for first 3 of 5 business days
        partial_index: pd.DatetimeIndex = full_index[:3]
        symbol: str = ALL_CROSS_ASSET_SYMBOLS[0]
        price_map: dict[str, pd.DataFrame] = {
            symbol: pd.DataFrame(
                {"adj_close": [100.0, 101.0, 102.0], "volume": [1000.0, 2000.0, 3000.0]},
                index=partial_index,
            ),
        }
        mock_fetch.return_value = price_map

        config: CrossAssetConfig = CrossAssetConfig(
            start_date="2020-01-06", end_date="2020-01-10",
        )
        result: pd.DataFrame = build_cross_asset_dataset(config)

        symbol_data: pd.DataFrame = cast(
            pd.DataFrame,
            result.loc[result["ticker"] == symbol],
        )
        assert len(symbol_data) == 5
        # Forward-filled: no NaN in adj_close
        assert symbol_data["adj_close"].isna().sum() == 0
        # Last two rows should carry forward the value 102.0
        assert symbol_data["adj_close"].iloc[-1] == 102.0
        assert symbol_data["adj_close"].iloc[-2] == 102.0

    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline._fetch_all_cross_assets")
    def test_sorted_by_date_ticker(self, mock_fetch: MagicMock) -> None:
        """Result must be sorted by [date, ticker]."""
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        rng: np.random.Generator = np.random.default_rng(42)
        n: int = len(index)
        symbols: list[str] = list(ALL_CROSS_ASSET_SYMBOLS[:3])
        price_map: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            price_map[sym] = pd.DataFrame(
                {"adj_close": rng.random(n) * 100, "volume": rng.random(n) * 1000},
                index=index,
            )
        mock_fetch.return_value = price_map

        config: CrossAssetConfig = CrossAssetConfig(
            start_date="2020-01-06", end_date="2020-01-10",
        )
        result: pd.DataFrame = build_cross_asset_dataset(config)
        assert result["date"].is_monotonic_increasing or (
            result["date"].diff().dropna() >= pd.Timedelta(0)
        ).all()



# ---------------------------------------------------------------------------
# _extract_symbol_df -- additional edge cases
# ---------------------------------------------------------------------------


class TestExtractSymbolDf:
    def test_single_symbol_flat_columns(self) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        df: pd.DataFrame = _make_single_symbol_df(index)
        result: pd.DataFrame | None = _extract_symbol_df(df, "XLK", is_multi=False)
        assert result is not None
        assert "adj_close" in result.columns
        assert "volume" in result.columns

    def test_no_matching_columns_returns_none(self) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        df: pd.DataFrame = pd.DataFrame({"Other": [1.0] * 5}, index=index)
        result: pd.DataFrame | None = _extract_symbol_df(df, "XLK", is_multi=False)
        assert result is None

    def test_all_nan_returns_none(self) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        df: pd.DataFrame = pd.DataFrame({
            "Adj Close": [np.nan] * 5,
            "Volume": [np.nan] * 5,
        }, index=index)
        result: pd.DataFrame | None = _extract_symbol_df(df, "XLK", is_multi=False)
        assert result is None

    def test_multiindex_missing_symbol(self) -> None:
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        symbols: list[str] = ["^N225"]
        raw: pd.DataFrame = _make_mock_cross_asset_df(index, symbols)
        result: pd.DataFrame | None = _extract_symbol_df(raw, "XLK", is_multi=True)
        assert result is None


# ---------------------------------------------------------------------------
# _fetch_cross_asset_batch -- yfinance not installed
# ---------------------------------------------------------------------------


class TestFetchCrossAssetBatchYfNone:
    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline.yf", None)
    def test_yf_none_raises(self) -> None:
        with pytest.raises(ImportError, match="yfinance is not installed"):
            _fetch_cross_asset_batch(["^N225"], "2020-01-06", "2020-01-10")


# ---------------------------------------------------------------------------
# _fetch_cross_asset_chunk_with_retry -- all retries exhausted
# ---------------------------------------------------------------------------


class TestFetchCrossAssetChunkWithRetry:
    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline.time.sleep")
    @patch("core.src.meta_model.data.data_fetching.cross_asset_pipeline._fetch_cross_asset_batch")
    def test_all_retries_fail_returns_empty(
        self, mock_batch: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_batch.side_effect = ConnectionError("Network error")
        config: CrossAssetConfig = CrossAssetConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            max_retries=2,
            retry_sleep=0.01,
        )
        result: dict[str, pd.DataFrame] = _fetch_cross_asset_chunk_with_retry(
            ["^N225"], config,
        )
        assert result == {}
        assert mock_batch.call_count == 2


# --- Execution directe (clic dans l'IDE) ---
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
