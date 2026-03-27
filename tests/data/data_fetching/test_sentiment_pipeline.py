# pyright: reportPrivateUsage=false
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_fetching.sentiment_pipeline import (
    SentimentConfig,
    _build_sentiment_dataframe,
    _download_bytes,
    _fetch_aaii_sentiment,
    _fetch_gpr_index,
    _parse_aaii_dataframe,
    _parse_gpr_dataframe,
    build_sentiment_dataset,
)
from core.src.meta_model.data.paths import DATA_FETCHING_DIR
from core.src.meta_model.data.trading_calendar import get_nyse_sessions


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


# ---------------------------------------------------------------------------
# Helpers -- mock Excel bytes
# ---------------------------------------------------------------------------


def _make_aaii_excel_bytes(
    start: str = "2020-01-02",
    periods: int = 5,
    freq: str = "W-THU",
) -> bytes:
    """Create realistic AAII survey Excel bytes for testing."""
    dates: pd.DatetimeIndex = pd.date_range(start, periods=periods, freq=freq)
    rng: np.random.Generator = np.random.default_rng(42)
    bullish: np.ndarray = rng.uniform(0.2, 0.5, size=periods)
    bearish: np.ndarray = rng.uniform(0.2, 0.5, size=periods)
    neutral: np.ndarray = 1.0 - bullish - bearish
    df: pd.DataFrame = pd.DataFrame({
        "Date": dates,
        "Bullish": bullish,
        "Neutral": neutral,
        "Bearish": bearish,
    })
    buf: io.BytesIO = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _make_gpr_excel_bytes(
    start: str = "2020-01-01",
    periods: int = 6,
    freq: str = "MS",
) -> bytes:
    """Create realistic GPR index Excel bytes for testing."""
    dates: pd.DatetimeIndex = pd.date_range(start, periods=periods, freq=freq)
    rng: np.random.Generator = np.random.default_rng(99)
    df: pd.DataFrame = pd.DataFrame({
        "month": dates,
        "GPRD": rng.uniform(50, 200, size=periods),
        "GPR_ACT": rng.uniform(10, 80, size=periods),
        "GPR_THREAT": rng.uniform(30, 150, size=periods),
    })
    buf: io.BytesIO = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _make_aaii_dataframe(
    start: str = "2020-01-02",
    periods: int = 5,
    freq: str = "W-THU",
) -> pd.DataFrame:
    """Return a parsed AAII DataFrame with DatetimeIndex."""
    dates: pd.DatetimeIndex = pd.date_range(start, periods=periods, freq=freq)
    rng: np.random.Generator = np.random.default_rng(42)
    bullish: np.ndarray = rng.uniform(0.2, 0.5, size=periods)
    bearish: np.ndarray = rng.uniform(0.2, 0.5, size=periods)
    neutral: np.ndarray = 1.0 - bullish - bearish
    df: pd.DataFrame = pd.DataFrame({
        "aaii_bullish": bullish,
        "aaii_neutral": neutral,
        "aaii_bearish": bearish,
    }, index=dates)
    df.index.name = "date"
    return df


def _make_gpr_dataframe(
    start: str = "2020-01-01",
    periods: int = 6,
    freq: str = "MS",
) -> pd.DataFrame:
    """Return a parsed GPR DataFrame with DatetimeIndex."""
    dates: pd.DatetimeIndex = pd.date_range(start, periods=periods, freq=freq)
    rng: np.random.Generator = np.random.default_rng(99)
    df: pd.DataFrame = pd.DataFrame({
        "gpr_index": rng.uniform(50, 200, size=periods),
    }, index=dates)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# SentimentConfig
# ---------------------------------------------------------------------------


class TestSentimentConfig:
    def test_default_values(self) -> None:
        config: SentimentConfig = SentimentConfig()
        assert config.start_date == "2004-01-01"
        assert config.end_date == "2025-12-31"
        assert config.output_dir == DATA_FETCHING_DIR
        assert config.sample_frac == 0.05
        assert config.random_seed == 7
        assert config.max_retries == 3
        assert config.retry_sleep == 2.0

    def test_frozen(self) -> None:
        config: SentimentConfig = SentimentConfig()
        with pytest.raises(AttributeError):
            config.start_date = "2010-01-01"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config: SentimentConfig = SentimentConfig(
            start_date="2010-01-01",
            end_date="2023-12-31",
            sample_frac=0.1,
            random_seed=42,
        )
        assert config.start_date == "2010-01-01"
        assert config.end_date == "2023-12-31"
        assert config.sample_frac == 0.1
        assert config.random_seed == 42


# ---------------------------------------------------------------------------
# _download_bytes
# ---------------------------------------------------------------------------


class TestDownloadBytes:
    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline.requests.get")
    def test_success(self, mock_get: MagicMock) -> None:
        mock_response: MagicMock = MagicMock()
        mock_response.content = b"fake-data"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        result: bytes = _download_bytes("https://example.com/file.xls", 3, 0.01)
        assert result == b"fake-data"
        mock_get.assert_called_once()

    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline.time.sleep")
    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline.requests.get")
    def test_retry_then_success(
        self, mock_get: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_ok: MagicMock = MagicMock()
        mock_ok.content = b"ok-data"
        mock_ok.raise_for_status.return_value = None
        mock_get.side_effect = [RuntimeError("network error"), mock_ok]
        result: bytes = _download_bytes("https://example.com/file.xls", 3, 0.01)
        assert result == b"ok-data"
        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline.time.sleep")
    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline.requests.get")
    def test_total_failure_raises(
        self, mock_get: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_get.side_effect = RuntimeError("network error")
        with pytest.raises(RuntimeError, match="Failed to download"):
            _download_bytes("https://example.com/file.xls", 2, 0.01)
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# _fetch_aaii_sentiment
# ---------------------------------------------------------------------------


class TestFetchAaiiSentiment:
    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._download_bytes")
    def test_success(self, mock_download: MagicMock) -> None:
        mock_download.return_value = _make_aaii_excel_bytes(
            start="2020-01-02", periods=5,
        )
        config: SentimentConfig = SentimentConfig(
            start_date="2020-01-01", end_date="2020-12-31",
        )
        result: pd.DataFrame = _fetch_aaii_sentiment(config)
        assert not result.empty
        assert result.index.name == "date"
        assert "aaii_bullish" in result.columns
        assert "aaii_neutral" in result.columns
        assert "aaii_bearish" in result.columns

    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._download_bytes")
    def test_parse_failure_returns_empty(self, mock_download: MagicMock) -> None:
        mock_download.return_value = b"not-a-valid-excel-file"
        config: SentimentConfig = SentimentConfig()
        result: pd.DataFrame = _fetch_aaii_sentiment(config)
        assert result.empty

    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._download_bytes")
    def test_download_failure_returns_empty(self, mock_download: MagicMock) -> None:
        mock_download.side_effect = RuntimeError("download failed")
        config: SentimentConfig = SentimentConfig()
        result: pd.DataFrame = _fetch_aaii_sentiment(config)
        assert result.empty

    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._download_bytes")
    def test_date_filtering(self, mock_download: MagicMock) -> None:
        mock_download.return_value = _make_aaii_excel_bytes(
            start="2020-01-02", periods=10, freq="W-THU",
        )
        config: SentimentConfig = SentimentConfig(
            start_date="2020-01-01", end_date="2020-02-01",
        )
        result: pd.DataFrame = _fetch_aaii_sentiment(config)
        assert not result.empty
        assert cast(pd.Timestamp, result.index.min()) >= _ts("2020-01-01")
        assert cast(pd.Timestamp, result.index.max()) <= _ts("2020-02-01")


# ---------------------------------------------------------------------------
# _fetch_gpr_index
# ---------------------------------------------------------------------------


class TestFetchGprIndex:
    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._download_bytes")
    def test_success(self, mock_download: MagicMock) -> None:
        mock_download.return_value = _make_gpr_excel_bytes(
            start="2020-01-01", periods=6,
        )
        config: SentimentConfig = SentimentConfig(
            start_date="2020-01-01", end_date="2020-12-31",
        )
        result: pd.DataFrame = _fetch_gpr_index(config)
        assert not result.empty
        assert result.index.name == "date"
        assert "gpr_index" in result.columns

    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._download_bytes")
    def test_optional_columns(self, mock_download: MagicMock) -> None:
        mock_download.return_value = _make_gpr_excel_bytes(
            start="2020-01-01", periods=6,
        )
        config: SentimentConfig = SentimentConfig(
            start_date="2020-01-01", end_date="2020-12-31",
        )
        result: pd.DataFrame = _fetch_gpr_index(config)
        assert "gpr_act" in result.columns
        assert "gpr_threat" in result.columns

    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._download_bytes")
    def test_parse_failure_returns_empty(self, mock_download: MagicMock) -> None:
        mock_download.return_value = b"not-a-valid-excel-file"
        config: SentimentConfig = SentimentConfig()
        result: pd.DataFrame = _fetch_gpr_index(config)
        assert result.empty

    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._download_bytes")
    def test_download_failure_returns_empty(self, mock_download: MagicMock) -> None:
        mock_download.side_effect = RuntimeError("download failed")
        config: SentimentConfig = SentimentConfig()
        result: pd.DataFrame = _fetch_gpr_index(config)
        assert result.empty


# ---------------------------------------------------------------------------
# _build_sentiment_dataframe
# ---------------------------------------------------------------------------


class TestBuildSentimentDataframe:
    def test_combines_both_sources(self) -> None:
        aaii: pd.DataFrame = _make_aaii_dataframe(
            start="2020-01-02", periods=3, freq="W-THU",
        )
        gpr: pd.DataFrame = _make_gpr_dataframe(
            start="2020-01-01", periods=2, freq="MS",
        )
        result: pd.DataFrame = _build_sentiment_dataframe(
            aaii, gpr, "2020-01-01", "2020-03-31",
        )
        assert "date" in result.columns
        assert "aaii_bullish" in result.columns
        assert "gpr_index" in result.columns
        sessions: pd.DatetimeIndex = get_nyse_sessions("2020-01-01", "2020-03-31")
        assert len(result) == len(sessions)

    def test_one_source_empty(self) -> None:
        aaii: pd.DataFrame = _make_aaii_dataframe(
            start="2020-01-02", periods=3, freq="W-THU",
        )
        result: pd.DataFrame = _build_sentiment_dataframe(
            aaii, pd.DataFrame(), "2020-01-01", "2020-01-31",
        )
        assert "date" in result.columns
        assert "aaii_bullish" in result.columns
        assert "gpr_index" not in result.columns

    def test_aligns_to_nyse_sessions(self) -> None:
        aaii: pd.DataFrame = _make_aaii_dataframe(
            start="2020-01-02", periods=4, freq="W-THU",
        )
        result: pd.DataFrame = _build_sentiment_dataframe(
            aaii, pd.DataFrame(), "2020-01-01", "2020-01-31",
        )
        sessions: pd.DatetimeIndex = get_nyse_sessions("2020-01-01", "2020-01-31")
        assert len(result) == len(sessions)

    def test_nan_preserved(self) -> None:
        aaii: pd.DataFrame = _make_aaii_dataframe(
            start="2020-01-02", periods=2, freq="W-THU",
        )
        result: pd.DataFrame = _build_sentiment_dataframe(
            aaii, pd.DataFrame(), "2020-01-01", "2020-01-31",
        )
        nan_count: int = int(cast(pd.Series, result["aaii_bullish"]).isna().sum())
        total: int = len(result)
        assert nan_count > 0
        assert nan_count < total

    def test_both_empty(self) -> None:
        result: pd.DataFrame = _build_sentiment_dataframe(
            pd.DataFrame(), pd.DataFrame(), "2020-01-06", "2020-01-10",
        )
        assert "date" in result.columns
        sessions: pd.DatetimeIndex = get_nyse_sessions("2020-01-06", "2020-01-10")
        assert len(result) == len(sessions)


# ---------------------------------------------------------------------------
# build_sentiment_dataset
# ---------------------------------------------------------------------------


class TestBuildSentimentDataset:
    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._fetch_gpr_index")
    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._fetch_aaii_sentiment")
    def test_full_flow(
        self, mock_aaii: MagicMock, mock_gpr: MagicMock,
    ) -> None:
        mock_aaii.return_value = _make_aaii_dataframe(
            start="2020-01-02", periods=3, freq="W-THU",
        )
        mock_gpr.return_value = _make_gpr_dataframe(
            start="2020-01-01", periods=2, freq="MS",
        )
        config: SentimentConfig = SentimentConfig(
            start_date="2020-01-01", end_date="2020-03-31",
        )
        result: pd.DataFrame = build_sentiment_dataset(config)
        assert "date" in result.columns
        assert "aaii_bullish" in result.columns
        assert "gpr_index" in result.columns
        sessions: pd.DatetimeIndex = get_nyse_sessions("2020-01-01", "2020-03-31")
        assert len(result) == len(sessions)

    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._fetch_gpr_index")
    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._fetch_aaii_sentiment")
    def test_one_source_empty_ok(
        self, mock_aaii: MagicMock, mock_gpr: MagicMock,
    ) -> None:
        mock_aaii.return_value = _make_aaii_dataframe(
            start="2020-01-02", periods=3, freq="W-THU",
        )
        mock_gpr.return_value = pd.DataFrame()
        config: SentimentConfig = SentimentConfig(
            start_date="2020-01-01", end_date="2020-03-31",
        )
        result: pd.DataFrame = build_sentiment_dataset(config)
        assert not result.empty
        assert "aaii_bullish" in result.columns

    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._fetch_gpr_index")
    @patch("core.src.meta_model.data.data_fetching.sentiment_pipeline._fetch_aaii_sentiment")
    def test_both_empty_raises(
        self, mock_aaii: MagicMock, mock_gpr: MagicMock,
    ) -> None:
        mock_aaii.return_value = pd.DataFrame()
        mock_gpr.return_value = pd.DataFrame()
        config: SentimentConfig = SentimentConfig()
        with pytest.raises(RuntimeError, match="Both sentiment sources"):
            build_sentiment_dataset(config)


# ---------------------------------------------------------------------------
# _parse_aaii_dataframe -- missing columns
# ---------------------------------------------------------------------------


class TestParseAaiiDataframeMissingColumns:
    def test_missing_required_columns_raises(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=3),
            "Bullish": [0.3, 0.4, 0.5],
            # Missing "Neutral" and "Bearish"
        })
        with pytest.raises(ValueError, match="AAII Excel missing columns"):
            _parse_aaii_dataframe(df, "2020-01-01", "2020-12-31")

    def test_non_date_summary_rows_dropped(self) -> None:
        """Summary rows like 'Observations over life of survey' must be dropped."""
        df: pd.DataFrame = pd.DataFrame({
            "Date": ["2020-01-02", "2020-01-09", "Observations over life of survey"],
            "Bullish": [0.35, 0.40, 0.38],
            "Neutral": [0.30, 0.30, 0.31],
            "Bearish": [0.35, 0.30, 0.31],
        })
        result: pd.DataFrame = _parse_aaii_dataframe(df, "2020-01-01", "2020-12-31")
        assert len(result) == 2
        assert result.index.min() == pd.Timestamp("2020-01-02")
        assert result.index.max() == pd.Timestamp("2020-01-09")


# ---------------------------------------------------------------------------
# _parse_gpr_dataframe -- missing columns
# ---------------------------------------------------------------------------


class TestParseGprDataframeMissingColumns:
    def test_missing_gpr_index_raises(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
            "other_col": [1, 2, 3],
        })
        with pytest.raises(ValueError, match="GPR Excel missing"):
            _parse_gpr_dataframe(df, "2020-01-01", "2020-12-31")

    def test_missing_date_raises(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "other": pd.date_range("2020-01-01", periods=3, freq="MS"),
            "GPRD": [1, 2, 3],
        })
        with pytest.raises(ValueError, match="GPR Excel missing"):
            _parse_gpr_dataframe(df, "2020-01-01", "2020-12-31")


# --- Execution directe (clic dans l'IDE) ---
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
