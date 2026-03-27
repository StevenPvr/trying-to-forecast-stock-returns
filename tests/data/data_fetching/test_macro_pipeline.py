# pyright: reportPrivateUsage=false
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_fetching.macro_pipeline import (
    MacroConfig,
    _build_macro_dataframe,
    _create_fred_client,
    _fetch_all_series,
    _fetch_series_with_semaphore,
    _fetch_single_series,
    _resolve_api_key,
    build_macro_dataset,
)
from core.src.meta_model.data.paths import DATA_FETCHING_DIR
from core.src.meta_model.data.trading_calendar import get_nyse_sessions


# ---------------------------------------------------------------------------
# MacroConfig
# ---------------------------------------------------------------------------


class TestMacroConfig:
    def test_default_values(self) -> None:
        config: MacroConfig = MacroConfig()
        assert config.start_date == "2004-01-01"
        assert config.end_date == "2025-12-31"
        assert config.output_dir == DATA_FETCHING_DIR
        assert config.sample_frac == 0.05
        assert config.random_seed == 7
        assert config.max_retries == 3
        assert config.retry_sleep == 2.0
        assert config.fred_api_key is None

    def test_frozen(self) -> None:
        config: MacroConfig = MacroConfig()
        with pytest.raises(AttributeError):
            config.start_date = "2010-01-01"  # type: ignore[misc]

    def test_custom_api_key(self) -> None:
        config: MacroConfig = MacroConfig(fred_api_key="test_key_123")
        assert config.fred_api_key == "test_key_123"


# ---------------------------------------------------------------------------
# _resolve_api_key
# ---------------------------------------------------------------------------


class TestResolveApiKey:
    def test_from_config(self) -> None:
        config: MacroConfig = MacroConfig(fred_api_key="config_key")
        assert _resolve_api_key(config) == "config_key"

    def test_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "env_key")
        config: MacroConfig = MacroConfig()
        assert _resolve_api_key(config) == "env_key"

    def test_config_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "env_key")
        config: MacroConfig = MacroConfig(fred_api_key="config_key")
        assert _resolve_api_key(config) == "config_key"

    def test_absent_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        config: MacroConfig = MacroConfig()
        with pytest.raises(ValueError, match="FRED API key not found"):
            _resolve_api_key(config)

    def test_empty_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "  ")
        config: MacroConfig = MacroConfig()
        with pytest.raises(ValueError, match="FRED API key not found"):
            _resolve_api_key(config)

    def test_wrong_length_warns(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setenv("FRED_API_KEY", "short_key")
        config: MacroConfig = MacroConfig()
        import logging
        with caplog.at_level(logging.WARNING):
            result: str = _resolve_api_key(config)
        assert result == "short_key"

    def test_strips_whitespace(self) -> None:
        config: MacroConfig = MacroConfig(fred_api_key="  my_key  ")
        assert _resolve_api_key(config) == "my_key"


# ---------------------------------------------------------------------------
# _create_fred_client
# ---------------------------------------------------------------------------


class TestCreateFredClient:
    @patch("core.src.meta_model.data.data_fetching.macro_pipeline.Fred", create=True)
    def test_success(self, mock_fred_cls: MagicMock) -> None:
        mock_instance: MagicMock = MagicMock()
        mock_fred_cls.return_value = mock_instance
        with patch.dict("sys.modules", {"fredapi": MagicMock(Fred=mock_fred_cls)}):
            result: object = _create_fred_client("test_key")
        assert result is not None

    def test_import_error(self) -> None:
        with patch.dict("sys.modules", {"fredapi": None}):
            with pytest.raises(ImportError, match="fredapi is not installed"):
                _create_fred_client("test_key")


# ---------------------------------------------------------------------------
# _fetch_single_series
# ---------------------------------------------------------------------------


class TestFetchSingleSeries:
    def test_success(self) -> None:
        mock_client: MagicMock = MagicMock()
        expected: pd.Series = pd.Series(  # type: ignore[type-arg]
            [1.0, 2.0, 3.0],
            index=pd.date_range("2020-01-01", periods=3),
        )
        mock_client.get_series.return_value = expected
        result: pd.Series = _fetch_single_series(  # type: ignore[type-arg]
            mock_client, "DGS10", "2020-01-01", "2020-12-31", 3, 0.01,
        )
        pd.testing.assert_series_equal(result, expected)

    def test_retry_then_success(self) -> None:
        mock_client: MagicMock = MagicMock()
        expected: pd.Series = pd.Series(  # type: ignore[type-arg]
            [1.0], index=pd.date_range("2020-01-01", periods=1),
        )
        mock_client.get_series.side_effect = [
            RuntimeError("API error"),
            expected,
        ]
        result: pd.Series = _fetch_single_series(  # type: ignore[type-arg]
            mock_client, "DGS10", "2020-01-01", "2020-12-31", 3, 0.01,
        )
        pd.testing.assert_series_equal(result, expected)
        assert mock_client.get_series.call_count == 2

    def test_total_failure_raises(self) -> None:
        mock_client: MagicMock = MagicMock()
        mock_client.get_series.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError, match="Failed to fetch DGS10"):
            _fetch_single_series(
                mock_client, "DGS10", "2020-01-01", "2020-12-31", 2, 0.01,
            )
        assert mock_client.get_series.call_count == 2


# ---------------------------------------------------------------------------
# _fetch_series_with_semaphore
# ---------------------------------------------------------------------------


class TestFetchSeriesWithSemaphore:

    @patch("core.src.meta_model.data.data_fetching.macro_pipeline.time.sleep")
    def test_success(self, mock_sleep: MagicMock) -> None:
        import threading
        mock_client: MagicMock = MagicMock()
        expected: pd.Series = pd.Series(  # type: ignore[type-arg]
            [1.0], index=pd.date_range("2020-01-01", periods=1),
        )
        mock_client.get_series.return_value = expected
        sem: threading.Semaphore = threading.Semaphore(1)
        series_id: str
        result: pd.Series | None  # type: ignore[type-arg]
        series_id, result = _fetch_series_with_semaphore(
            mock_client, "DGS10", "2020-01-01", "2020-12-31", 3, 0.01, 0.5, sem,
        )
        assert series_id == "DGS10"
        assert result is not None
        mock_sleep.assert_called_with(0.5)

    @patch("core.src.meta_model.data.data_fetching.macro_pipeline.time.sleep")
    def test_failure_returns_none(self, mock_sleep: MagicMock) -> None:
        import threading
        mock_client: MagicMock = MagicMock()
        mock_client.get_series.side_effect = RuntimeError("fail")
        sem: threading.Semaphore = threading.Semaphore(1)
        series_id: str
        result: pd.Series | None  # type: ignore[type-arg]
        series_id, result = _fetch_series_with_semaphore(
            mock_client, "BAD", "2020-01-01", "2020-12-31", 2, 0.01, 0.5, sem,
        )
        assert series_id == "BAD"
        assert result is None
        # Still calls rate_limit sleep in finally block
        mock_sleep.assert_called_with(0.5)


# ---------------------------------------------------------------------------
# _fetch_all_series
# ---------------------------------------------------------------------------


class TestFetchAllSeries:
    def test_multiple_series(self) -> None:
        mock_client: MagicMock = MagicMock()
        s1: pd.Series = pd.Series(  # type: ignore[type-arg]
            [1.0], index=pd.date_range("2020-01-01", periods=1),
        )
        s2: pd.Series = pd.Series(  # type: ignore[type-arg]
            [2.0], index=pd.date_range("2020-01-01", periods=1),
        )
        mock_client.get_series.side_effect = [s1, s2]
        result: dict[str, pd.Series] = _fetch_all_series(  # type: ignore[type-arg]
            mock_client, ("DGS10", "DGS2"), "2020-01-01", "2020-12-31", 3, 0.01, 0.01,
        )
        assert set(result.keys()) == {"DGS10", "DGS2"}

    @patch("core.src.meta_model.data.data_fetching.macro_pipeline.time.sleep")
    def test_rate_limit_sleep(self, mock_sleep: MagicMock) -> None:
        mock_client: MagicMock = MagicMock()
        s1: pd.Series = pd.Series(  # type: ignore[type-arg]
            [1.0], index=pd.date_range("2020-01-01", periods=1),
        )
        mock_client.get_series.return_value = s1
        _fetch_all_series(
            mock_client, ("A", "B", "C"), "2020-01-01", "2020-12-31", 3, 0.01, 0.5,
        )
        sleep_calls: list[float] = [
            call.args[0] for call in mock_sleep.call_args_list
        ]
        rate_limit_calls: list[float] = [s for s in sleep_calls if s == 0.5]
        # Each of the 3 series gets a rate-limit sleep in the finally block
        assert len(rate_limit_calls) == 3

    def test_skip_failed_series(self) -> None:
        mock_client: MagicMock = MagicMock()
        s1: pd.Series = pd.Series(  # type: ignore[type-arg]
            [1.0], index=pd.date_range("2020-01-01", periods=1),
        )
        mock_client.get_series.side_effect = [
            s1,
            RuntimeError("fail"),
            RuntimeError("fail"),
        ]
        result: dict[str, pd.Series] = _fetch_all_series(  # type: ignore[type-arg]
            mock_client, ("OK", "FAIL"), "2020-01-01", "2020-12-31", 2, 0.01, 0.01,
        )
        assert "OK" in result
        assert "FAIL" not in result


# ---------------------------------------------------------------------------
# _build_macro_dataframe
# ---------------------------------------------------------------------------


class TestBuildMacroDataframe:
    def test_aligns_on_nyse_sessions(self) -> None:
        index: pd.DatetimeIndex = pd.date_range("2020-01-01", periods=5)
        series: pd.Series = pd.Series(  # type: ignore[type-arg]
            [1.0, 2.0, 3.0, 4.0, 5.0], index=index,
        )
        result: pd.DataFrame = _build_macro_dataframe(
            {"DGS10": series}, "2020-01-01", "2020-01-10",
        )
        assert "date" in result.columns
        assert "dgs10" in result.columns
        sessions: pd.DatetimeIndex = get_nyse_sessions("2020-01-01", "2020-01-10")
        assert len(result) == len(sessions)

    def test_applies_conservative_availability_lag_before_ffill(self) -> None:
        bdays: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        series: pd.Series = pd.Series(  # type: ignore[type-arg]
            [1.0, np.nan, 3.0], index=bdays[:3],
        )
        result: pd.DataFrame = _build_macro_dataframe(
            {"DGS10": series}, "2020-01-06", "2020-01-10",
        )
        assert pd.isna(result["dgs10"].iloc[0])
        assert result["dgs10"].iloc[1:].tolist() == [1.0, 1.0, 3.0, 3.0]

    def test_columns_lowercase(self) -> None:
        series: pd.Series = pd.Series(  # type: ignore[type-arg]
            [1.0], index=pd.date_range("2020-01-01", periods=1),
        )
        result: pd.DataFrame = _build_macro_dataframe(
            {"VIXCLS": series, "DFF": series}, "2020-01-01", "2020-01-03",
        )
        assert "vixcls" in result.columns
        assert "dff" in result.columns
        assert "VIXCLS" not in result.columns

    def test_empty_series_map(self) -> None:
        result: pd.DataFrame = _build_macro_dataframe(
            {}, "2020-01-06", "2020-01-10",
        )
        assert "date" in result.columns
        sessions: pd.DatetimeIndex = get_nyse_sessions("2020-01-06", "2020-01-10")
        assert len(result) == len(sessions)


# ---------------------------------------------------------------------------
# build_macro_dataset
# ---------------------------------------------------------------------------


class TestBuildMacroDataset:
    @patch("core.src.meta_model.data.data_fetching.macro_pipeline._fetch_all_series")
    @patch("core.src.meta_model.data.data_fetching.macro_pipeline._create_fred_client")
    @patch("core.src.meta_model.data.data_fetching.macro_pipeline._resolve_api_key")
    def test_full_flow(
        self,
        mock_resolve: MagicMock,
        mock_create: MagicMock,
        mock_fetch_all: MagicMock,
    ) -> None:
        mock_resolve.return_value = "test_key"
        mock_create.return_value = MagicMock()
        index: pd.DatetimeIndex = pd.bdate_range("2020-01-06", "2020-01-10")
        mock_fetch_all.return_value = {
            "DGS10": pd.Series([1.0] * len(index), index=index),
            "VIXCLS": pd.Series([20.0] * len(index), index=index),
        }
        config: MacroConfig = MacroConfig(
            start_date="2020-01-06",
            end_date="2020-01-10",
            fred_api_key="test_key",
        )
        result: pd.DataFrame = build_macro_dataset(config)
        assert "date" in result.columns
        assert "dgs10" in result.columns
        assert "vixcls" in result.columns
        assert len(result) == len(index)

    @patch("core.src.meta_model.data.data_fetching.macro_pipeline._fetch_all_series")
    @patch("core.src.meta_model.data.data_fetching.macro_pipeline._create_fred_client")
    @patch("core.src.meta_model.data.data_fetching.macro_pipeline._resolve_api_key")
    def test_no_data_raises(
        self,
        mock_resolve: MagicMock,
        mock_create: MagicMock,
        mock_fetch_all: MagicMock,
    ) -> None:
        mock_resolve.return_value = "test_key"
        mock_create.return_value = MagicMock()
        mock_fetch_all.return_value = {}
        config: MacroConfig = MacroConfig(fred_api_key="test_key")
        with pytest.raises(RuntimeError, match="No macro series"):
            build_macro_dataset(config)

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        config: MacroConfig = MacroConfig()
        with pytest.raises(ValueError, match="FRED API key not found"):
            build_macro_dataset(config)


# --- Execution directe (clic dans l'IDE) ---
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
