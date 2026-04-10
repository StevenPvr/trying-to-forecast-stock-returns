# pyright: reportPrivateUsage=false
from __future__ import annotations

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

from core.src.meta_model.data.data_fetching.main import (
    _adjust_ohlc_prices,
    _resolve_pipeline_config,
    _build_date_features,
    _build_universe_company_listing,
    _clean_cross_asset_symbol,
    _drop_unneeded_raw_price_columns,
    _fill_missing_beta_from_market_returns,
    _drop_high_nan_tickers,
    _drop_leading_nan_rows,
    _merge_date_features,
    _pivot_cross_asset_wide,
    _rename_columns_explicitly,
    _save_merged,
    _save_universe_company_listing,
    _transform_price_columns_to_log_returns,
    main,
)


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


# ---------------------------------------------------------------------------
# _clean_cross_asset_symbol
# ---------------------------------------------------------------------------


class TestCleanCrossAssetSymbol:
    def test_nikkei_mapping(self) -> None:
        assert _clean_cross_asset_symbol("^N225") == "nikkei"

    def test_dax_mapping(self) -> None:
        assert _clean_cross_asset_symbol("^GDAXI") == "dax"

    def test_ftse_mapping(self) -> None:
        assert _clean_cross_asset_symbol("^FTSE") == "ftse"

    def test_hangseng_mapping(self) -> None:
        assert _clean_cross_asset_symbol("^HSI") == "hangseng"

    def test_shanghai_mapping(self) -> None:
        assert _clean_cross_asset_symbol("000001.SS") == "shanghai"



    def test_gold_mapping(self) -> None:
        assert _clean_cross_asset_symbol("GC=F") == "gold"

    def test_fallback_simple(self) -> None:
        assert _clean_cross_asset_symbol("XLK") == "xlk"

    def test_fallback_dash(self) -> None:
        assert _clean_cross_asset_symbol("HYG") == "hyg"

    def test_fallback_caret(self) -> None:
        assert _clean_cross_asset_symbol("^UNKNOWN") == "unknown"

    def test_fallback_equals(self) -> None:
        assert _clean_cross_asset_symbol("SI=F") == "sif"

    def test_fallback_combined(self) -> None:
        assert _clean_cross_asset_symbol("A-B=C") == "a_bc"


# ---------------------------------------------------------------------------
# _pivot_cross_asset_wide
# ---------------------------------------------------------------------------


class TestPivotCrossAssetWide:
    def test_basic_pivot(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-06", "2020-01-07", "2020-01-07"]),
            "ticker": ["^N225", "XLK", "^N225", "XLK"],
            "adj_close": [100.0, 200.0, 101.0, 201.0],
        })
        result: pd.DataFrame = _pivot_cross_asset_wide(df)
        assert "date" in result.columns
        assert "xa_nikkei" in result.columns
        assert "xa_xlk" in result.columns
        assert len(result) == 2

    def test_column_names_prefixed(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06"]),
            "ticker": ["GC=F"],
            "adj_close": [1800.0],
        })
        result: pd.DataFrame = _pivot_cross_asset_wide(df)
        assert "xa_gold" in result.columns

    def test_does_not_modify_original(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06"]),
            "ticker": ["^N225"],
            "adj_close": [100.0],
        })
        original_ticker: str = df["ticker"].iloc[0]
        _pivot_cross_asset_wide(df)
        assert df["ticker"].iloc[0] == original_ticker



# ---------------------------------------------------------------------------
# _merge_date_features
# ---------------------------------------------------------------------------


class TestMergeDateFeatures:
    def test_empty_list(self) -> None:
        result: pd.DataFrame = _merge_date_features([])
        assert "date" in result.columns
        assert len(result) == 0

    def test_single_frame(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
            "value": [1.0, 2.0],
        })
        result: pd.DataFrame = _merge_date_features([df])
        assert len(result) == 2
        assert "value" in result.columns

    def test_multiple_frames(self) -> None:
        df1: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
            "value_a": [1.0, 2.0],
        })
        df2: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
            "value_b": [3.0, 4.0],
        })
        result: pd.DataFrame = _merge_date_features([df1, df2])
        assert "value_a" in result.columns
        assert "value_b" in result.columns
        assert len(result) == 2

    def test_outer_join_fills_nans(self) -> None:
        df1: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06"]),
            "value_a": [1.0],
        })
        df2: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-07"]),
            "value_b": [2.0],
        })
        result: pd.DataFrame = _merge_date_features([df1, df2])
        assert len(result) == 2
        assert result["value_a"].isna().sum() == 1
        assert result["value_b"].isna().sum() == 1


# ---------------------------------------------------------------------------
# _build_date_features
# ---------------------------------------------------------------------------


class TestBuildDateFeatures:
    @patch("core.src.meta_model.data.data_fetching.main.build_cross_asset_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_sentiment_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_calendar_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_macro_dataset")
    def test_all_succeed(
        self,
        mock_macro: MagicMock,
        mock_calendar: MagicMock,
        mock_sentiment: MagicMock,
        mock_cross: MagicMock,
    ) -> None:
        dates: list[pd.Timestamp] = [_ts("2020-01-06"), _ts("2020-01-07")]
        mock_macro.return_value = pd.DataFrame({"date": dates, "dgs10": [1.5, 1.6]})
        mock_calendar.return_value = pd.DataFrame({"date": dates, "is_fomc_day": [0, 1]})
        mock_sentiment.return_value = pd.DataFrame({"date": dates, "aaii_bullish": [0.3, 0.4]})
        mock_cross.return_value = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-06"]),
            "ticker": ["^N225", "XLK"],
            "adj_close": [100.0, 200.0],
        })
        result: list[pd.DataFrame] = _build_date_features()
        assert len(result) == 4

    @patch("core.src.meta_model.data.data_fetching.main.build_cross_asset_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_sentiment_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_calendar_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_macro_dataset")
    def test_partial_failures(
        self,
        mock_macro: MagicMock,
        mock_calendar: MagicMock,
        mock_sentiment: MagicMock,
        mock_cross: MagicMock,
    ) -> None:
        mock_macro.side_effect = RuntimeError("FRED failed")
        mock_calendar.return_value = pd.DataFrame({
            "date": [pd.Timestamp("2020-01-06")],
            "is_fomc_day": [0],
        })
        mock_sentiment.side_effect = RuntimeError("Sentiment failed")
        mock_cross.side_effect = RuntimeError("Cross-asset failed")
        result: list[pd.DataFrame] = _build_date_features(strict=False)
        assert len(result) == 1

    @patch("core.src.meta_model.data.data_fetching.main.build_cross_asset_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_sentiment_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_calendar_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_macro_dataset")
    def test_all_fail(
        self,
        mock_macro: MagicMock,
        mock_calendar: MagicMock,
        mock_sentiment: MagicMock,
        mock_cross: MagicMock,
    ) -> None:
        mock_macro.side_effect = RuntimeError("fail")
        mock_calendar.side_effect = RuntimeError("fail")
        mock_sentiment.side_effect = RuntimeError("fail")
        mock_cross.side_effect = RuntimeError("fail")
        result: list[pd.DataFrame] = _build_date_features(strict=False)
        assert len(result) == 0


class TestFillMissingBetaFromMarketReturns:
    def test_fills_beta_from_rolling_market_returns(self) -> None:
        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-02", "2024-01-02",
                "2024-01-03", "2024-01-03",
                "2024-01-04", "2024-01-04",
            ]),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "adj_close_log_return": [0.01, 0.02, 0.02, 0.01, 0.03, 0.01],
            "beta": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        result = _fill_missing_beta_from_market_returns(df, window=3, min_periods=2)

        assert int(result["beta"].notna().sum()) > 0
        assert result["beta"].iloc[-1] == pytest.approx(result["beta"].iloc[-1])


class TestResolvePipelineConfig:
    def test_raises_when_point_in_time_membership_history_is_missing(
        self,
        tmp_path: Path,
    ) -> None:
        missing_membership: Path = tmp_path / "missing_membership.csv"
        missing_fundamentals: Path = tmp_path / "missing_fundamentals.csv"

        with (
            patch(
                "core.src.meta_model.data.data_fetching.main.MEMBERSHIP_HISTORY_CSV",
                missing_membership,
            ),
            patch(
                "core.src.meta_model.data.data_fetching.main.FUNDAMENTALS_HISTORY_CSV",
                missing_fundamentals,
            ),
        ):
            with pytest.raises(FileNotFoundError, match="membership history"):
                _resolve_pipeline_config()

    def test_raises_when_point_in_time_fundamentals_history_is_missing(
        self,
        tmp_path: Path,
    ) -> None:
        membership: Path = tmp_path / "membership.csv"
        missing_fundamentals: Path = tmp_path / "missing_fundamentals.csv"
        membership.write_text("ticker,start_date,end_date\nAAPL,2024-01-01,2024-12-31\n")

        with (
            patch(
                "core.src.meta_model.data.data_fetching.main.MEMBERSHIP_HISTORY_CSV",
                membership,
            ),
            patch(
                "core.src.meta_model.data.data_fetching.main.FUNDAMENTALS_HISTORY_CSV",
                missing_fundamentals,
            ),
        ):
            with pytest.raises(FileNotFoundError, match="fundamentals history"):
                _resolve_pipeline_config()

    def test_raises_when_xtb_instrument_specs_are_missing(
        self,
        tmp_path: Path,
    ) -> None:
        membership: Path = tmp_path / "membership.csv"
        fundamentals: Path = tmp_path / "fundamentals.csv"
        missing_xtb_specs: Path = tmp_path / "missing_xtb_specs.json"
        membership.write_text("ticker,start_date,end_date\nAAPL,2024-01-01,2024-12-31\n")
        fundamentals.write_text("date,ticker,company_market_cap_usd\n2024-01-02,AAPL,100\n")

        with (
            patch(
                "core.src.meta_model.data.data_fetching.main.MEMBERSHIP_HISTORY_CSV",
                membership,
            ),
            patch(
                "core.src.meta_model.data.data_fetching.main.FUNDAMENTALS_HISTORY_CSV",
                fundamentals,
            ),
            patch(
                "core.src.meta_model.data.data_fetching.main.XTB_INSTRUMENT_SPECS_REFERENCE_JSON",
                missing_xtb_specs,
            ),
        ):
            with pytest.raises(FileNotFoundError, match="XTB instrument specification"):
                _resolve_pipeline_config()

    def test_reads_standard_paths_when_present(
        self,
        tmp_path: Path,
    ) -> None:
        membership: Path = tmp_path / "membership.csv"
        fundamentals: Path = tmp_path / "fundamentals.csv"
        xtb_specs: Path = tmp_path / "xtb_instrument_specs.json"
        membership.write_text("ticker,start_date,end_date\nAAPL,2024-01-01,2024-12-31\n")
        fundamentals.write_text("date,ticker,company_market_cap_usd\n2024-01-02,AAPL,100\n")
        xtb_specs.write_text("[]")

        with (
            patch(
                "core.src.meta_model.data.data_fetching.main.MEMBERSHIP_HISTORY_CSV",
                membership,
            ),
            patch(
                "core.src.meta_model.data.data_fetching.main.FUNDAMENTALS_HISTORY_CSV",
                fundamentals,
            ),
            patch(
                "core.src.meta_model.data.data_fetching.main.XTB_INSTRUMENT_SPECS_REFERENCE_JSON",
                xtb_specs,
            ),
        ):
            config = _resolve_pipeline_config()

        assert config.membership_history_csv == membership
        assert config.fundamentals_history_csv == fundamentals
        assert config.xtb_instrument_specs_json == xtb_specs
        assert config.xtb_only is True
        assert config.require_xtb_snapshot is True
        assert config.allow_current_constituents_snapshot is False


# ---------------------------------------------------------------------------
# _save_merged
# ---------------------------------------------------------------------------


class TestSaveMerged:
    @patch("core.src.meta_model.data.data_fetching.main.DATA_FETCHING_DIR")
    def test_creates_files(self, mock_dir: MagicMock, tmp_path: Path) -> None:
        mock_dir.__truediv__ = lambda self, name: tmp_path / name  # type: ignore[assignment]
        mock_dir.mkdir = MagicMock()
        mock_dir.name = tmp_path.name

        with (
            patch("core.src.meta_model.data.data_fetching.main.DATA_FETCHING_DIR", tmp_path),
            patch("core.src.meta_model.data.data_fetching.main.MERGED_OUTPUT_PARQUET", tmp_path / "dataset.parquet"),
            patch("core.src.meta_model.data.data_fetching.main.MERGED_OUTPUT_SAMPLE_CSV", tmp_path / "dataset_sample.csv"),
        ):
            data: pd.DataFrame = pd.DataFrame({
                "date": pd.bdate_range("2020-01-06", periods=100),
                "ticker": ["AAPL"] * 100,
                "adj_close": list(range(100)),
            })
            result: dict[str, Path] = _save_merged(data)
            assert result["parquet"].exists()
            assert result["sample_csv"].exists()

    @patch("core.src.meta_model.data.data_fetching.main.DATA_FETCHING_DIR")
    def test_parquet_roundtrip(self, mock_dir: MagicMock, tmp_path: Path) -> None:
        with (
            patch("core.src.meta_model.data.data_fetching.main.DATA_FETCHING_DIR", tmp_path),
            patch("core.src.meta_model.data.data_fetching.main.MERGED_OUTPUT_PARQUET", tmp_path / "dataset.parquet"),
            patch("core.src.meta_model.data.data_fetching.main.MERGED_OUTPUT_SAMPLE_CSV", tmp_path / "dataset_sample.csv"),
        ):
            data: pd.DataFrame = pd.DataFrame({
                "date": pd.bdate_range("2020-01-06", periods=50),
                "ticker": ["AAPL"] * 50,
                "adj_close": list(range(50)),
            })
            result: dict[str, Path] = _save_merged(data)
            loaded: pd.DataFrame = pd.read_parquet(result["parquet"])
            assert len(loaded) == 50
            assert list(loaded.columns) == list(data.columns)

    @patch("core.src.meta_model.data.data_fetching.main.DATA_FETCHING_DIR")
    def test_sample_size(self, mock_dir: MagicMock, tmp_path: Path) -> None:
        with (
            patch("core.src.meta_model.data.data_fetching.main.DATA_FETCHING_DIR", tmp_path),
            patch("core.src.meta_model.data.data_fetching.main.MERGED_OUTPUT_PARQUET", tmp_path / "dataset.parquet"),
            patch("core.src.meta_model.data.data_fetching.main.MERGED_OUTPUT_SAMPLE_CSV", tmp_path / "dataset_sample.csv"),
        ):
            data: pd.DataFrame = pd.DataFrame({
                "date": pd.bdate_range("2020-01-06", periods=100),
                "ticker": ["AAPL"] * 100,
                "adj_close": list(range(100)),
            })
            result: dict[str, Path] = _save_merged(data)
            sample: pd.DataFrame = pd.read_csv(result["sample_csv"])
            assert len(sample) == 5  # 5% of 100


class TestUniverseCompanyListing:
    def test_builds_listing_from_dataset_company_name(self, tmp_path: Path) -> None:
        data = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-06"]),
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "company_name": ["Apple Inc.", "Apple Inc.", "Microsoft Corporation"],
        })
        config = MagicMock(membership_history_csv=None, allow_current_constituents_snapshot=False)

        result = _build_universe_company_listing(data, config)

        assert result.to_dict(orient="records") == [
            {"ticker": "AAPL", "company_name": "Apple Inc."},
            {"ticker": "MSFT", "company_name": "Microsoft Corporation"},
        ]

    def test_falls_back_to_membership_history_name_column(self, tmp_path: Path) -> None:
        membership_path = tmp_path / "membership.csv"
        membership_path.write_text(
            "ticker,security,start_date,end_date\n"
            "AAPL,Apple Inc.,2024-01-01,2024-12-31\n"
            "MSFT,Microsoft Corporation,2024-01-01,2024-12-31\n",
        )
        data = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-06"]),
            "ticker": ["AAPL", "MSFT"],
        })
        config = MagicMock(
            membership_history_csv=membership_path,
            allow_current_constituents_snapshot=False,
        )

        result = _build_universe_company_listing(data, config)

        assert result.to_dict(orient="records") == [
            {"ticker": "AAPL", "company_name": "Apple Inc."},
            {"ticker": "MSFT", "company_name": "Microsoft Corporation"},
        ]

    def test_save_listing_writes_excel(self, tmp_path: Path) -> None:
        listing = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "company_name": ["Apple Inc.", "Microsoft Corporation"],
        })

        result = _save_universe_company_listing(
            listing,
            tmp_path / "universe.xlsx",
        )

        loaded = pd.read_excel(result)
        assert result.exists()
        assert loaded.to_dict(orient="records") == listing.to_dict(orient="records")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    @patch("core.src.meta_model.data.data_fetching.main._resolve_pipeline_config")
    @patch("core.src.meta_model.data.data_fetching.main._save_merged")
    @patch("core.src.meta_model.data.data_fetching.main._save_universe_company_listing")
    @patch("core.src.meta_model.data.data_fetching.main._merge_date_features")
    @patch("core.src.meta_model.data.data_fetching.main._build_date_features")
    @patch("core.src.meta_model.data.data_fetching.main.build_dataset")
    def test_full_flow(
        self,
        mock_build: MagicMock,
        mock_date_features: MagicMock,
        mock_merge: MagicMock,
        mock_save_universe: MagicMock,
        mock_save: MagicMock,
        mock_resolve_config: MagicMock,
    ) -> None:
        dates: list[pd.Timestamp] = [_ts("2020-01-06"), _ts("2020-01-07")]
        mock_build.return_value = pd.DataFrame({
            "date": dates,
            "ticker": ["AAPL", "AAPL"],
            "adj_close": [100.0, 101.0],
        })
        mock_date_features.return_value = [
            pd.DataFrame({"date": dates, "feature": [1.0, 2.0]}),
        ]
        mock_merge.return_value = pd.DataFrame({
            "date": dates,
            "feature": [1.0, 2.0],
        })
        mock_save_universe.return_value = Path("/fake_universe.xlsx")
        mock_save.return_value = {"parquet": Path("/fake.parquet"), "sample_csv": Path("/fake.csv")}
        mock_resolve_config.return_value = MagicMock()

        main()

        mock_build.assert_called_once()
        mock_date_features.assert_called_once()
        mock_merge.assert_called_once()
        mock_save_universe.assert_called_once()
        mock_save.assert_called_once()

    @patch("core.src.meta_model.data.data_fetching.main._resolve_pipeline_config")
    @patch("core.src.meta_model.data.data_fetching.main._save_merged")
    @patch("core.src.meta_model.data.data_fetching.main._build_date_features")
    @patch("core.src.meta_model.data.data_fetching.main.build_dataset")
    def test_empty_date_features(
        self,
        mock_build: MagicMock,
        mock_date_features: MagicMock,
        mock_save: MagicMock,
        mock_resolve_config: MagicMock,
    ) -> None:
        dates: list[pd.Timestamp] = [_ts("2020-01-06")]
        mock_build.return_value = pd.DataFrame({
            "date": dates,
            "ticker": ["AAPL"],
            "adj_close": [100.0],
        })
        mock_date_features.return_value = []
        mock_save.return_value = {"parquet": Path("/fake.parquet"), "sample_csv": Path("/fake.csv")}
        mock_resolve_config.return_value = MagicMock()

        main()

        mock_save.assert_called_once()

    @patch("core.src.meta_model.data.data_fetching.main.DATA_FETCHING_DIR")
    @patch("core.src.meta_model.data.data_fetching.main._resolve_pipeline_config")
    @patch("core.src.meta_model.data.data_fetching.main._save_merged")
    @patch("core.src.meta_model.data.data_fetching.main._build_date_features")
    @patch("core.src.meta_model.data.data_fetching.main.build_dataset")
    def test_mkdir_called_early(
        self,
        mock_build: MagicMock,
        mock_date_features: MagicMock,
        mock_save: MagicMock,
        mock_resolve_config: MagicMock,
        mock_dir: MagicMock,
    ) -> None:
        """Output directory must be created at the very start of main()."""
        dates: list[pd.Timestamp] = [_ts("2020-01-06")]
        mock_build.return_value = pd.DataFrame({
            "date": dates,
            "ticker": ["AAPL"],
            "adj_close": [100.0],
        })
        mock_date_features.return_value = []
        mock_save.return_value = {"parquet": Path("/fake.parquet"), "sample_csv": Path("/fake.csv")}
        mock_resolve_config.return_value = MagicMock()

        main()

        mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# _drop_high_nan_tickers
# ---------------------------------------------------------------------------


class TestDropHighNanTickers:
    def test_drops_ticker_above_threshold(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-06"]),
            "ticker": ["GOOD", "BAD"],
            "val": [1.0, np.nan],
        })
        result: pd.DataFrame = _drop_high_nan_tickers(df, threshold=0.0)
        assert "GOOD" in result["ticker"].values
        assert "BAD" not in result["ticker"].values

    def test_keeps_ticker_below_threshold(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-06", "2020-01-07"]),
            "ticker": ["A", "A", "B", "B"],
            "val": [1.0, 2.0, 3.0, 4.0],
        })
        result: pd.DataFrame = _drop_high_nan_tickers(df, threshold=1.0)
        assert result["ticker"].nunique() == 2

    def test_ignores_date_level_nan_when_price_columns_are_complete(self) -> None:
        df: pd.DataFrame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2020-01-06", "2020-01-07", "2020-01-06", "2020-01-07"],
                ),
                "ticker": ["A", "A", "B", "B"],
                "open": [1.0, 2.0, 3.0, 4.0],
                "macro_signal": [np.nan, 1.0, np.nan, 1.0],
            }
        )

        result: pd.DataFrame = _drop_high_nan_tickers(df, threshold=1.0)

        assert result["ticker"].nunique() == 2


# ---------------------------------------------------------------------------
# _drop_leading_nan_rows
# ---------------------------------------------------------------------------


class TestDropLeadingNanRows:
    def test_drops_leading_dates_with_nan(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"]),
            "ticker": ["A", "A", "A"],
            "val": [np.nan, np.nan, 3.0],
        })
        result: pd.DataFrame = _drop_leading_nan_rows(df)
        assert len(result) == 1
        assert result["date"].iloc[0] == pd.Timestamp("2020-01-08")

    def test_keeps_all_when_no_leading_nan(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
            "ticker": ["A", "A"],
            "val": [1.0, 2.0],
        })
        result: pd.DataFrame = _drop_leading_nan_rows(df)
        assert len(result) == 2

    def test_stops_at_first_complete_date(self) -> None:
        """If date 3 has NaN but date 2 is clean, only date 1 is trimmed."""
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"]),
            "ticker": ["A", "A", "A"],
            "val": [np.nan, 2.0, np.nan],
        })
        result: pd.DataFrame = _drop_leading_nan_rows(df)
        assert len(result) == 2
        assert result["date"].min() == pd.Timestamp("2020-01-07")

    def test_ignores_sparse_company_columns_when_trimming(self) -> None:
        df: pd.DataFrame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
                "ticker": ["A", "A"],
                "open": [100.0, 101.0],
                "macro_signal": [np.nan, 1.0],
                "enterprise_value": [np.nan, np.nan],
            }
        )

        result: pd.DataFrame = _drop_leading_nan_rows(df)

        assert len(result) == 1
        assert result["date"].iloc[0] == pd.Timestamp("2020-01-07")


# ---------------------------------------------------------------------------
# _transform_price_columns_to_log_returns
# ---------------------------------------------------------------------------


class TestTransformPriceColumnsToLogReturns:
    def test_keeps_raw_prices_and_adds_log_return_columns(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime([
                "2020-01-06", "2020-01-06",
                "2020-01-07", "2020-01-07",
            ]),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "open": [100.0, 200.0, 110.0, 220.0],
            "adj_close": [100.0, 200.0, 121.0, 242.0],
            "xa_dax": [1000.0, 1000.0, 1100.0, 1100.0],
            "volume": [1_000_000, 2_000_000, 1_100_000, 2_200_000],
        }).sort_values(["date", "ticker"]).reset_index(drop=True)

        result: pd.DataFrame = _transform_price_columns_to_log_returns(df)

        aapl_rows: pd.DataFrame = cast(
            pd.DataFrame,
            result.loc[result["ticker"] == "AAPL"].reset_index(drop=True),
        )
        msft_rows: pd.DataFrame = cast(
            pd.DataFrame,
            result.loc[result["ticker"] == "MSFT"].reset_index(drop=True),
        )

        assert aapl_rows.loc[0, "open"] == pytest.approx(100.0)
        assert msft_rows.loc[1, "adj_close"] == pytest.approx(242.0)
        assert aapl_rows.loc[1, "xa_dax"] == pytest.approx(1100.0)
        assert pd.isna(aapl_rows.loc[0, "open_log_return"])
        assert pd.isna(msft_rows.loc[0, "adj_close_log_return"])
        assert aapl_rows.loc[1, "open_log_return"] == pytest.approx(np.log(110.0 / 100.0))
        assert msft_rows.loc[1, "adj_close_log_return"] == pytest.approx(np.log(242.0 / 200.0))
        assert aapl_rows.loc[1, "xa_dax_log_return"] == pytest.approx(np.log(1100.0 / 1000.0))

    def test_non_positive_prices_become_nan(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
            "ticker": ["AAPL", "AAPL"],
            "adj_close": [100.0, 0.0],
        })
        result: pd.DataFrame = _transform_price_columns_to_log_returns(df)
        assert result.loc[1, "adj_close"] == pytest.approx(0.0)
        assert pd.isna(result.loc[1, "adj_close_log_return"])

    def test_no_ticker_column_returns_input(self) -> None:
        df: pd.DataFrame = pd.DataFrame({"open": [100.0, 101.0]})
        result: pd.DataFrame = _transform_price_columns_to_log_returns(df)
        pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# _adjust_ohlc_prices
# ---------------------------------------------------------------------------


class TestAdjustOhlcPrices:
    def test_adjusts_ohlc_by_ratio(self) -> None:
        """A 2:1 split (adj_close = close / 2) halves all OHLC columns."""
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
            "ticker": ["AAPL", "AAPL"],
            "open": [200.0, 100.0],
            "high": [210.0, 105.0],
            "low": [190.0, 95.0],
            "close": [200.0, 100.0],
            "adj_close": [100.0, 100.0],  # split happened between day 1 and day 2
        })
        result: pd.DataFrame = _adjust_ohlc_prices(df)
        # Day 1: ratio = 100/200 = 0.5 → open becomes 100
        assert result["open"].iloc[0] == pytest.approx(100.0)
        assert result["high"].iloc[0] == pytest.approx(105.0)
        assert result["low"].iloc[0] == pytest.approx(95.0)
        assert result["close"].iloc[0] == pytest.approx(100.0)
        # Day 2: ratio = 100/100 = 1.0 → unchanged
        assert result["open"].iloc[1] == pytest.approx(100.0)

    def test_close_zero_leaves_unchanged(self) -> None:
        """Rows with close=0 should not crash; ratio defaults to 1.0."""
        df: pd.DataFrame = pd.DataFrame({
            "open": [50.0], "high": [55.0], "low": [45.0],
            "close": [0.0], "adj_close": [np.nan],
        })
        result: pd.DataFrame = _adjust_ohlc_prices(df)
        assert result["open"].iloc[0] == pytest.approx(50.0)

    def test_no_adj_close_returns_unchanged(self) -> None:
        """Missing adj_close column should return df unmodified."""
        df: pd.DataFrame = pd.DataFrame({
            "open": [100.0], "high": [110.0], "low": [90.0], "close": [105.0],
        })
        result: pd.DataFrame = _adjust_ohlc_prices(df)
        pd.testing.assert_frame_equal(result, df)

    def test_already_adjusted_is_idempotent(self) -> None:
        """Tiingo-style data where adj_close == close → no change."""
        df: pd.DataFrame = pd.DataFrame({
            "open": [100.0], "high": [110.0], "low": [90.0],
            "close": [105.0], "adj_close": [105.0],
        })
        result: pd.DataFrame = _adjust_ohlc_prices(df)
        assert result["open"].iloc[0] == pytest.approx(100.0)
        assert result["close"].iloc[0] == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# _drop_unneeded_raw_price_columns
# ---------------------------------------------------------------------------


class TestDropUnneededRawPriceColumns:
    def test_keeps_only_raw_ohlc_prices(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06"]),
            "ticker": ["AAPL"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "adj_close": [100.4],
            "xa_dax": [1000.0],
            "open_log_return": [0.01],
            "adj_close_log_return": [0.02],
            "xa_dax_log_return": [0.03],
            "volume": [1_000_000],
        })

        result: pd.DataFrame = _drop_unneeded_raw_price_columns(df)

        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "adj_close" not in result.columns
        assert "xa_dax" not in result.columns
        assert "open_log_return" in result.columns
        assert "adj_close_log_return" in result.columns
        assert "xa_dax_log_return" in result.columns
        assert "volume" in result.columns


# ---------------------------------------------------------------------------
# _rename_columns_explicitly
# ---------------------------------------------------------------------------


class TestRenameColumnsExplicitly:
    def test_renames_known_columns(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06"]),
            "ticker": ["AAPL"],
            "open": [100.0],
            "open_log_return": [0.01],
            "xa_dax_log_return": [0.02],
            "dgs10": [4.0],
            "is_fomc_day": [0],
        })

        result: pd.DataFrame = _rename_columns_explicitly(df)

        assert "stock_open_price" in result.columns
        assert "stock_open_log_return" in result.columns
        assert "cross_asset_dax_log_return" in result.columns
        assert "macro_us_treasury_10y_yield_pct" in result.columns
        assert "calendar_is_fomc_announcement_day" in result.columns
        assert "open" not in result.columns
        assert "open_log_return" not in result.columns
        assert "xa_dax_log_return" not in result.columns
        assert "dgs10" not in result.columns
        assert "is_fomc_day" not in result.columns

    def test_keeps_unknown_columns_unchanged(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-06"]),
            "ticker": ["AAPL"],
            "custom_feature": [123.0],
        })
        result: pd.DataFrame = _rename_columns_explicitly(df)
        assert "custom_feature" in result.columns


# --- Execution directe (clic dans l'IDE) ---
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
