from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.features_engineering.main import (
    REQUIRED_TA_INPUT_COLUMNS,
    TA_FEATURE_PREFIX,
    add_feature_lags,
    build_ta_feature_dataset,
    build_feature_dataset,
    load_cleaned_dataset,
    main,
    save_feature_dataset,
    save_lagged_feature_dataset,
)
from core.src.meta_model.features_engineering.pipeline import resolve_max_workers
from core.src.meta_model.features_engineering.lag_features import (
    build_lagged_feature_group,
    get_laggable_feature_columns,
)


def _make_cleaned_price_df(tickers: tuple[str, ...] = ("AAPL", "MSFT")) -> pd.DataFrame:
    dates: pd.DatetimeIndex = pd.date_range("2020-01-01", periods=120, freq="D")
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(tickers, start=1):
        base: float = 100.0 * ticker_index
        for step, date in enumerate(dates):
            price: float = base + step
            rows.append({
                "date": date,
                "ticker": ticker,
                "stock_open_price": price,
                "stock_high_price": price + 1.0,
                "stock_low_price": price - 1.0,
                "stock_close_price": price + 0.5,
                "stock_trading_volume": 1_000_000.0 + (ticker_index * 1000.0) + step,
                "stock_open_log_return": np.nan if step == 0 else 0.01,
            })
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def _make_cleaned_price_df_with_exogenous_features() -> pd.DataFrame:
    data = _make_cleaned_price_df(("AAPL", "MSFT"))
    row_count = len(data)
    data["macro_us_treasury_10y_yield_pct"] = np.linspace(3.0, 4.0, row_count)
    data["sentiment_aaii_bullish_share"] = np.linspace(0.2, 0.4, row_count)
    data["calendar_is_fomc_announcement_day"] = 0
    data.loc[data.index[::21], "calendar_is_fomc_announcement_day"] = 1
    data["calendar_days_since_previous_fomc"] = np.tile(np.arange(len(data) // 2), 2)[:row_count]
    data["cross_asset_dax_log_return"] = np.linspace(-0.02, 0.02, row_count)
    data["company_market_cap_usd"] = np.where(
        data["ticker"] == "AAPL",
        2_500_000_000_000.0,
        2_000_000_000_000.0,
    )
    return data


class TestLoadCleanedDataset:
    def test_loads_parquet(self, tmp_path: Path) -> None:
        path: Path = tmp_path / "cleaned.parquet"
        df: pd.DataFrame = _make_cleaned_price_df(("AAPL",))
        df.to_parquet(path, index=False)

        result: pd.DataFrame = load_cleaned_dataset(path)

        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Input parquet not found"):
            load_cleaned_dataset(tmp_path / "missing.parquet")


class TestBuildTaFeatureDataset:
    def test_resolve_max_workers_defaults_to_all_detected_cores(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "core.src.meta_model.runtime_parallelism.os.cpu_count",
            lambda: 8,
        )

        assert resolve_max_workers() == 8

    def test_rejects_duplicate_date_ticker_rows(self) -> None:
        df: pd.DataFrame = _make_cleaned_price_df(("AAPL",))
        duplicated: pd.DataFrame = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        with pytest.raises(ValueError, match="duplicate \\(date, ticker\\) rows"):
            build_feature_dataset(duplicated)

    def test_requires_ohlcv_input_columns(self) -> None:
        df: pd.DataFrame = _make_cleaned_price_df(("AAPL",)).drop(
            columns=["stock_high_price"],
        )

        with pytest.raises(ValueError, match="Missing required TA input columns"):
            build_ta_feature_dataset(df)

    def test_adds_ta_indicators_per_ticker(self) -> None:
        df: pd.DataFrame = _make_cleaned_price_df()

        result: pd.DataFrame = build_ta_feature_dataset(df)

        assert len(result) == len(df)
        assert list(result[["date", "ticker"]].head(4).itertuples(index=False, name=None)) == [
            (pd.Timestamp("2020-01-01"), "AAPL"),
            (pd.Timestamp("2020-01-01"), "MSFT"),
            (pd.Timestamp("2020-01-02"), "AAPL"),
            (pd.Timestamp("2020-01-02"), "MSFT"),
        ]
        for column in REQUIRED_TA_INPUT_COLUMNS:
            assert column in result.columns
        for column in (
            f"{TA_FEATURE_PREFIX}volume_adi",
            f"{TA_FEATURE_PREFIX}volatility_bbm",
            f"{TA_FEATURE_PREFIX}trend_macd",
            f"{TA_FEATURE_PREFIX}momentum_rsi",
            f"{TA_FEATURE_PREFIX}others_dr",
        ):
            assert column in result.columns

    def test_does_not_mix_tickers_when_computing_returns(self) -> None:
        df: pd.DataFrame = _make_cleaned_price_df()

        result: pd.DataFrame = build_ta_feature_dataset(df)

        aapl_first = result.loc[result["ticker"] == "AAPL", f"{TA_FEATURE_PREFIX}others_dr"].iloc[0]
        msft_first = result.loc[result["ticker"] == "MSFT", f"{TA_FEATURE_PREFIX}others_dr"].iloc[0]
        assert pd.isna(aapl_first)
        assert pd.isna(msft_first)

    def test_future_shock_does_not_change_past_features(self) -> None:
        baseline: pd.DataFrame = _make_cleaned_price_df()
        shocked: pd.DataFrame = baseline.copy()
        shock_mask = (
            (shocked["ticker"] == "AAPL")
            & (shocked["date"] == shocked["date"].max())
        )
        shocked.loc[shock_mask, "stock_close_price"] *= 10.0
        shocked.loc[shock_mask, "stock_high_price"] *= 10.0
        shocked.loc[shock_mask, "stock_low_price"] *= 10.0
        shocked.loc[shock_mask, "stock_open_price"] *= 10.0

        baseline_result: pd.DataFrame = build_feature_dataset(baseline)
        shocked_result: pd.DataFrame = build_feature_dataset(shocked)

        compare_date = baseline["date"].max() - pd.Timedelta(days=1)
        baseline_row = baseline_result[
            (baseline_result["ticker"] == "AAPL") & (baseline_result["date"] == compare_date)
        ].iloc[0]
        shocked_row = shocked_result[
            (shocked_result["ticker"] == "AAPL") & (shocked_result["date"] == compare_date)
        ].iloc[0]

        for column in (
            f"{TA_FEATURE_PREFIX}trend_macd",
            f"{TA_FEATURE_PREFIX}momentum_rsi",
            "quant_momentum_21d",
            "quant_universe_return_1d_ex_self",
        ):
            assert shocked_row[column] == pytest.approx(baseline_row[column], nan_ok=True)

    def test_uses_only_cyclical_calendar_time_encodings(self) -> None:
        df: pd.DataFrame = _make_cleaned_price_df(("AAPL",))

        result: pd.DataFrame = build_feature_dataset(df)

        for column in (
            "quant_day_of_week_sin",
            "quant_day_of_week_cos",
            "quant_month_of_year_sin",
            "quant_month_of_year_cos",
            "quant_day_of_month_sin",
            "quant_day_of_month_cos",
            "quant_day_of_year_sin",
            "quant_day_of_year_cos",
        ):
            assert column in result.columns

        for column in (
            "quant_day_of_week",
            "quant_month_of_year",
            "quant_day_of_month",
            "quant_day_of_year",
        ):
            assert column not in result.columns

    def test_adds_deep_features_in_the_main_feature_pipeline(self) -> None:
        df: pd.DataFrame = _make_cleaned_price_df(("AAPL",))

        result: pd.DataFrame = build_feature_dataset(df)

        for column in (
            "deep_event_inside_day_flag",
            "deep_path_sign_flip_rate_21d",
            "deep_state_gap_fill_rate_21d",
        ):
            assert column in result.columns

    def test_builds_features_with_process_pool_executor_available(self) -> None:
        df: pd.DataFrame = _make_cleaned_price_df(
            ("AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM"),
        )
        result: pd.DataFrame = build_feature_dataset(df)

        from core.src.meta_model.features_engineering import pipeline as feature_pipeline
        assert hasattr(feature_pipeline, "ProcessPoolExecutor")
        assert len(result) == len(df)

    def test_uses_process_pool_executor_when_multiple_workers_requested(self) -> None:
        df: pd.DataFrame = _make_cleaned_price_df(("AAPL", "MSFT", "NVDA"))

        class FakeExecutor:
            def __init__(self, max_workers: int) -> None:
                self.max_workers = max_workers

            def __enter__(self) -> "FakeExecutor":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def map(self, func, groups):
                return [func(group) for group in groups]

        with patch("core.src.meta_model.features_engineering.pipeline.ProcessPoolExecutor", FakeExecutor):
            result: pd.DataFrame = build_feature_dataset(df, max_workers=2)

        assert len(result) == len(df)


class TestSaveFeatureDataset:
    def test_saves_parquet_and_sample(self, tmp_path: Path) -> None:
        df: pd.DataFrame = _make_cleaned_price_df(("AAPL",))
        parquet_path: Path = tmp_path / "features.parquet"
        sample_path: Path = tmp_path / "features_sample.csv"

        paths = save_feature_dataset(df, parquet_path, sample_path)

        assert parquet_path.exists()
        assert sample_path.exists()
        assert paths["parquet"] == parquet_path
        assert paths["sample_csv"] == sample_path

    def test_saves_lagged_dataset_without_materializing_global_lag_frame(self, tmp_path: Path) -> None:
        featured: pd.DataFrame = build_feature_dataset(_make_cleaned_price_df(("AAPL", "MSFT")))
        parquet_path: Path = tmp_path / "features_lagged.parquet"
        sample_path: Path = tmp_path / "features_lagged_sample.csv"

        paths = save_lagged_feature_dataset(featured, parquet_path, sample_path)

        saved = pd.read_parquet(parquet_path)
        assert parquet_path.exists()
        assert sample_path.exists()
        assert paths["parquet"] == parquet_path
        assert paths["sample_csv"] == sample_path
        assert "ta_trend_macd_lag_1d" in saved.columns
        assert "quant_momentum_21d_lag_5d" in saved.columns
        assert "deep_event_inside_day_flag_lag_1d" in saved.columns

    def test_saves_adaptive_lags_for_prefixed_exogenous_features(self, tmp_path: Path) -> None:
        featured = build_feature_dataset(_make_cleaned_price_df_with_exogenous_features())
        parquet_path: Path = tmp_path / "features_lagged_exogenous.parquet"
        sample_path: Path = tmp_path / "features_lagged_exogenous_sample.csv"

        save_lagged_feature_dataset(featured, parquet_path, sample_path)

        saved = pd.read_parquet(parquet_path)
        assert "macro_us_treasury_10y_yield_pct_lag_21d" in saved.columns
        assert "sentiment_aaii_bullish_share_lag_10d" in saved.columns
        assert "cross_asset_dax_log_return_lag_63d" in saved.columns
        assert "company_market_cap_usd_lag_21d" in saved.columns
        assert "calendar_days_since_previous_fomc_lag_21d" in saved.columns
        assert "calendar_is_fomc_announcement_day_lag_1d" not in saved.columns

    def test_saves_lagged_dataset_without_pandas_schema_metadata(self, tmp_path: Path) -> None:
        featured: pd.DataFrame = build_feature_dataset(_make_cleaned_price_df(("AAPL", "MSFT")))
        parquet_path: Path = tmp_path / "features_lagged_no_metadata.parquet"
        sample_path: Path = tmp_path / "features_lagged_no_metadata_sample.csv"

        save_lagged_feature_dataset(featured, parquet_path, sample_path)

        schema = pq.read_schema(parquet_path)
        metadata = schema.metadata or {}
        assert b"pandas" not in metadata

    def test_streams_lagged_sample_csv_during_save(self, tmp_path: Path) -> None:
        featured: pd.DataFrame = build_feature_dataset(_make_cleaned_price_df(("AAPL", "MSFT")))
        parquet_path: Path = tmp_path / "features_lagged_streamed.parquet"
        sample_path: Path = tmp_path / "features_lagged_streamed_sample.csv"
        original_to_csv = pd.DataFrame.to_csv
        captured_modes: list[str] = []

        def spy_to_csv(self, *args, **kwargs):
            captured_modes.append(kwargs.get("mode", "w"))
            return original_to_csv(self, *args, **kwargs)

        with patch.object(pd.DataFrame, "to_csv", autospec=True, side_effect=spy_to_csv):
            save_lagged_feature_dataset(featured, parquet_path, sample_path)

        assert captured_modes[0] == "w"
        assert "a" in captured_modes[1:]

    def test_does_not_publish_partial_outputs_when_streaming_save_fails(self, tmp_path: Path) -> None:
        featured: pd.DataFrame = build_feature_dataset(_make_cleaned_price_df(("AAPL", "MSFT")))
        parquet_path: Path = tmp_path / "features_lagged_atomic.parquet"
        sample_path: Path = tmp_path / "features_lagged_atomic_sample.csv"

        original_builder = (
            __import__("core.src.meta_model.features_engineering.io", fromlist=["build_lagged_feature_group"])
            .build_lagged_feature_group
        )
        call_count = {"value": 0}

        def flaky_builder(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 2:
                raise RuntimeError("boom")
            return original_builder(*args, **kwargs)

        with patch("core.src.meta_model.features_engineering.io.build_lagged_feature_group", side_effect=flaky_builder):
            with pytest.raises(RuntimeError, match="boom"):
                save_lagged_feature_dataset(featured, parquet_path, sample_path)

        assert not parquet_path.exists()
        assert not sample_path.exists()


class TestFeatureLags:
    def test_excludes_non_numeric_company_columns_from_laggable_features(self) -> None:
        featured = build_feature_dataset(_make_cleaned_price_df_with_exogenous_features())
        featured["company_name"] = np.where(
            featured["ticker"] == "AAPL",
            "Apple Inc.",
            "Microsoft Corporation",
        )

        laggable_columns = get_laggable_feature_columns(list(featured.columns), featured)

        assert "company_name" not in laggable_columns
        assert "company_market_cap_usd" in laggable_columns

    def test_adds_lags_only_for_dynamic_feature_families(self) -> None:
        featured: pd.DataFrame = build_feature_dataset(_make_cleaned_price_df(("AAPL",)))

        lagged: pd.DataFrame = add_feature_lags(featured)

        for column in (
            "ta_trend_macd_lag_1d",
            "quant_momentum_21d_lag_5d",
            "quant_universe_return_1d_ex_self_lag_10d",
            "stock_open_log_return_lag_1d",
            "xtb_spread_to_realized_vol_21d_lag_1d",
            "sector_relative_gap_return_lag_5d",
            "open_above_prev_high_flag_lag_1d",
            "earnings_days_to_next_lag_1d",
        ):
            assert column in lagged.columns

        for column in (
            "stock_open_price_lag_1d",
            "stock_close_price_lag_5d",
            "stock_trading_volume_lag_1d",
            "ta_volume_adi_lag_1d",
            "ta_volume_obv_lag_5d",
            "quant_day_of_week_sin_lag_1d",
            "quant_month_of_year_cos_lag_5d",
            "quant_is_month_end_lag_1d",
            "quant_days_in_sample_lag_1d",
            "quant_dollar_volume_lag_1d",
            "quant_adv_21d_lag_1d",
            "quant_trend_strength_63d_lag_5d",
        ):
            assert column not in lagged.columns

    def test_stores_all_lag_columns_as_float32(self) -> None:
        featured: pd.DataFrame = build_feature_dataset(_make_cleaned_price_df(("AAPL",)))

        lagged: pd.DataFrame = add_feature_lags(featured)

        for column in (
            "ta_trend_macd_lag_1d",
            "quant_momentum_21d_lag_5d",
            "quant_universe_return_1d_ex_self_lag_10d",
            "stock_open_log_return_lag_1d",
        ):
            assert str(lagged[column].dtype) == "float32"

    def test_computes_lags_within_each_ticker_only(self) -> None:
        featured: pd.DataFrame = build_feature_dataset(_make_cleaned_price_df(("AAPL", "MSFT")))

        lagged: pd.DataFrame = add_feature_lags(featured)

        aapl = lagged.loc[lagged["ticker"] == "AAPL"].sort_values("date").reset_index(drop=True)
        msft = lagged.loc[lagged["ticker"] == "MSFT"].sort_values("date").reset_index(drop=True)

        assert pd.isna(aapl.loc[0, "stock_open_log_return_lag_1d"])
        assert pd.isna(msft.loc[0, "stock_open_log_return_lag_1d"])
        assert aapl.loc[1, "stock_open_log_return_lag_1d"] == pytest.approx(
            aapl.loc[0, "stock_open_log_return"],
            nan_ok=True,
        )
        assert msft.loc[1, "stock_open_log_return_lag_1d"] == pytest.approx(
            msft.loc[0, "stock_open_log_return"],
            nan_ok=True,
        )

    def test_preserves_stable_schema_for_non_lagged_columns_across_groups(self) -> None:
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        group_small = pd.DataFrame({
            "date": dates,
            "ticker": ["AAA"] * 4,
            "stock_trading_volume": [1_000.0, 1_100.0, 1_200.0, 1_300.0],
            "ta_volume_em": [0.1, 0.2, 0.3, 0.4],
            "ta_volume_sma_em": [0.1, 0.2, 0.3, 0.4],
            "ta_volume_nvi": [100.0, 101.0, 102.0, 103.0],
            "quant_trend_slope_21d": [0.01, 0.02, 0.03, 0.04],
            "quant_trend_strength_21d": [0.01, 0.02, 0.03, 0.04],
            "stock_open_log_return": [np.nan, 0.01, 0.01, 0.01],
        })
        group_large = pd.DataFrame({
            "date": dates,
            "ticker": ["BBB"] * 4,
            "stock_trading_volume": [1e12, 1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0],
            "ta_volume_em": [1e10, 1e10 + 1.0, 1e10 + 2.0, 1e10 + 3.0],
            "ta_volume_sma_em": [1e10, 1e10 + 1.0, 1e10 + 2.0, 1e10 + 3.0],
            "ta_volume_nvi": [1e12, 1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0],
            "quant_trend_slope_21d": [1e10, 1e10 + 1.0, 1e10 + 2.0, 1e10 + 3.0],
            "quant_trend_strength_21d": [1e10, 1e10 + 1.0, 1e10 + 2.0, 1e10 + 3.0],
            "stock_open_log_return": [np.nan, 0.01, 0.01, 0.01],
        })

        laggable_columns = ["quant_trend_slope_21d", "stock_open_log_return"]
        lagged_small = build_lagged_feature_group(group_small, laggable_columns=laggable_columns)
        lagged_large = build_lagged_feature_group(group_large, laggable_columns=laggable_columns)

        for column in (
            "stock_trading_volume",
            "ta_volume_em",
            "ta_volume_sma_em",
            "ta_volume_nvi",
            "quant_trend_slope_21d",
            "quant_trend_strength_21d",
        ):
            assert str(lagged_small[column].dtype) == str(lagged_large[column].dtype)


class TestMain:
    def test_full_flow(self, tmp_path: Path) -> None:
        input_path: Path = tmp_path / "cleaned.parquet"
        output_path: Path = tmp_path / "features.parquet"
        sample_path: Path = tmp_path / "features_sample.csv"
        _make_cleaned_price_df().to_parquet(input_path, index=False)

        with (
            patch("core.src.meta_model.features_engineering.main.CLEANED_OUTPUT_PARQUET", input_path),
            patch("core.src.meta_model.features_engineering.main.FEATURES_OUTPUT_PARQUET", output_path),
            patch("core.src.meta_model.features_engineering.main.FEATURES_OUTPUT_SAMPLE_CSV", sample_path),
        ):
            main()

        result: pd.DataFrame = pd.read_parquet(output_path)
        assert f"{TA_FEATURE_PREFIX}volume_adi" in result.columns
        assert f"{TA_FEATURE_PREFIX}trend_macd" in result.columns
        assert "ta_trend_macd_lag_1d" in result.columns
        assert "quant_momentum_21d_lag_5d" in result.columns
        assert "stock_open_log_return_lag_1d" in result.columns
        assert "quant_day_of_week_sin_lag_1d" not in result.columns
        assert result["ticker"].nunique() == 2
        assert len(result) == 240

    def test_full_flow_calls_build_feature_dataset(self, tmp_path: Path) -> None:
        input_path: Path = tmp_path / "cleaned.parquet"
        output_path: Path = tmp_path / "features.parquet"
        sample_path: Path = tmp_path / "features_sample.csv"
        _make_cleaned_price_df(("AAPL", "MSFT", "NVDA")).to_parquet(input_path, index=False)

        featured = build_feature_dataset(_make_cleaned_price_df(("AAPL", "MSFT", "NVDA")), max_workers=1)
        with (
            patch("core.src.meta_model.features_engineering.main.CLEANED_OUTPUT_PARQUET", input_path),
            patch("core.src.meta_model.features_engineering.main.FEATURES_OUTPUT_PARQUET", output_path),
            patch("core.src.meta_model.features_engineering.main.FEATURES_OUTPUT_SAMPLE_CSV", sample_path),
            patch(
                "core.src.meta_model.features_engineering.main.build_feature_dataset",
                return_value=featured,
            ) as build_feature_dataset_mock,
        ):
            main()

        build_feature_dataset_mock.assert_called_once()
        result: pd.DataFrame = pd.read_parquet(output_path, columns=["ticker"])
        assert result["ticker"].nunique() == 3
