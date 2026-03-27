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

from core.src.accounting_feature_engineering.main import (
    resolve_max_workers,
    run_accounting_feature_engineering,
)


def _make_feature_dataset() -> pd.DataFrame:
    dates = pd.bdate_range("2014-01-02", periods=260)
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(("AAA", "BBB"), start=1):
        close_price = 100.0 + np.cumsum(np.full(len(dates), 0.2 + (0.01 * ticker_index)))
        target_main = np.roll(np.diff(np.log(close_price), prepend=np.log(close_price[0])), -5)
        ticker_frame = pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "stock_close_price": close_price,
            "target_main": target_main,
            "quant_signal_a": np.sin(np.linspace(0.0, 8.0, len(dates))) * ticker_index,
        })
        rows.extend(ticker_frame.to_dict(orient="records"))
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def _make_accounting_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(("AAA", "BBB"), start=1):
        base_revenue = 100.0 * ticker_index
        quarter_dates = pd.to_datetime(
            [
                "2012-09-30",
                "2012-12-31",
                "2013-03-31",
                "2013-06-30",
                "2013-09-30",
                "2013-12-31",
                "2014-03-31",
                "2014-06-30",
            ],
        )
        available_dates = quarter_dates + pd.Timedelta(days=45)
        for quarter_index, (period_end, available_date) in enumerate(zip(quarter_dates, available_dates), start=1):
            revenue = base_revenue + (quarter_index * 10.0)
            net_income = revenue * 0.15
            operating_cash_flow = revenue * 0.18
            capex = -(revenue * 0.04)
            rows.append({
                "ticker": ticker,
                "period_end": period_end,
                "available_date": available_date,
                "revenue": revenue,
                "gross_profit": revenue * 0.52,
                "operating_income": revenue * 0.21,
                "net_income": net_income,
                "total_assets": revenue * 4.0,
                "total_liabilities": revenue * 1.8,
                "stockholders_equity": revenue * 2.2,
                "current_assets": revenue * 1.3,
                "current_liabilities": revenue * 0.7,
                "cash_and_equivalents": revenue * 0.3,
                "inventory": revenue * 0.1,
                "accounts_receivable": revenue * 0.15,
                "total_debt": revenue * 0.9,
                "operating_cash_flow": operating_cash_flow,
                "capital_expenditure": capex,
                "free_cash_flow": operating_cash_flow + capex,
                "availability_source": "sec",
                "value_source": "yfinance",
                "used_fallback_source": False,
            })
    return pd.DataFrame(rows).sort_values(["ticker", "available_date"]).reset_index(drop=True)


class TestRunAccountingFeatureEngineering:
    def test_resolve_max_workers_defaults_to_n_minus_one(self) -> None:
        with patch("core.src.accounting_feature_engineering.main.os.cpu_count", return_value=8):
            assert resolve_max_workers(None) == 7

    def test_rejects_non_positive_worker_counts(self) -> None:
        with pytest.raises(ValueError, match="max_workers must be strictly positive"):
            resolve_max_workers(0)

    def test_merges_accounting_features_and_saves_outputs(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        history_parquet = tmp_path / "history.parquet"
        history_csv = tmp_path / "history.csv"
        output_parquet = tmp_path / "accounting_features.parquet"
        output_csv = tmp_path / "accounting_features_sample.csv"
        _make_feature_dataset().to_parquet(feature_path, index=False)
        accounting_history = _make_accounting_history()

        with (
            patch(
                "core.src.accounting_feature_engineering.main.build_accounting_history_for_universe",
                return_value=accounting_history,
            ),
            patch(
                "core.src.accounting_feature_engineering.main.ACCOUNTING_FEATURES_OUTPUT_PARQUET",
                output_parquet,
            ),
            patch(
                "core.src.accounting_feature_engineering.main.ACCOUNTING_FEATURES_OUTPUT_SAMPLE_CSV",
                output_csv,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_HISTORY_PARQUET",
                history_parquet,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_HISTORY_SAMPLE_CSV",
                history_csv,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_FEATURES_OUTPUT_PARQUET",
                output_parquet,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_FEATURES_OUTPUT_SAMPLE_CSV",
                output_csv,
            ),
        ):
            enriched = run_accounting_feature_engineering(
                feature_parquet_path=feature_path,
                max_workers=1,
            )

        assert output_parquet.exists()
        assert output_csv.exists()
        assert history_parquet.exists()
        assert history_csv.exists()
        expected_columns = {
            "accounting_revenue_ttm_usd",
            "accounting_net_income_ttm_usd",
            "accounting_operating_cash_flow_ttm_usd",
            "accounting_free_cash_flow_ttm_usd",
            "accounting_current_ratio",
            "accounting_quick_ratio",
            "accounting_debt_to_equity_ratio",
            "accounting_gross_margin_ttm",
            "accounting_net_margin_ttm",
            "accounting_revenue_yoy_growth",
            "accounting_leverage_qoq_change",
            "accounting_available_date",
            "accounting_period_end",
            "accounting_availability_source",
            "accounting_value_source",
            "accounting_revenue_ttm_usd_lag_1report",
            "accounting_revenue_ttm_usd_lag_2report",
            "accounting_revenue_ttm_usd_lag_4report",
            "accounting_current_ratio_lag_1report",
            "accounting_debt_to_equity_ratio_lag_4report",
            "accounting_revenue_qoq_growth",
            "accounting_net_income_qoq_growth",
            "accounting_operating_cash_flow_qoq_growth",
            "accounting_revenue_growth_acceleration",
            "accounting_net_income_growth_acceleration",
            "accounting_revenue_vs_trailing_mean_ratio",
            "accounting_revenue_historical_percentile",
            "accounting_days_since_available_report",
            "accounting_is_recent_report_5d",
            "accounting_is_recent_report_20d",
        }
        assert expected_columns <= set(enriched.columns)
        warm_enriched = enriched.loc[enriched["date"] >= pd.Timestamp("2014-08-20")].reset_index(drop=True)
        assert warm_enriched.loc[:, sorted(expected_columns - {"accounting_availability_source", "accounting_value_source"})].notna().any().all()

    def test_logs_history_build_and_merge_progress(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        feature_path = tmp_path / "features.parquet"
        output_parquet = tmp_path / "accounting_features.parquet"
        output_csv = tmp_path / "accounting_features_sample.csv"
        _make_feature_dataset().to_parquet(feature_path, index=False)

        with (
            caplog.at_level("INFO"),
            patch(
                "core.src.accounting_feature_engineering.main.build_accounting_history_for_universe",
                return_value=_make_accounting_history(),
            ),
            patch(
                "core.src.accounting_feature_engineering.main.ACCOUNTING_FEATURES_OUTPUT_PARQUET",
                output_parquet,
            ),
            patch(
                "core.src.accounting_feature_engineering.main.ACCOUNTING_FEATURES_OUTPUT_SAMPLE_CSV",
                output_csv,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_FEATURES_OUTPUT_PARQUET",
                output_parquet,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_FEATURES_OUTPUT_SAMPLE_CSV",
                output_csv,
            ),
        ):
            run_accounting_feature_engineering(
                feature_parquet_path=feature_path,
                max_workers=1,
            )

        messages = [record.getMessage() for record in caplog.records]
        assert any("Loaded accounting feature base dataset" in message for message in messages)
        assert any("Built accounting feature snapshot dataset" in message for message in messages)
        assert any("Merged accounting features into base dataset" in message for message in messages)

    def test_handles_object_dtype_numeric_columns_with_pd_na(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        output_parquet = tmp_path / "accounting_features.parquet"
        output_csv = tmp_path / "accounting_features_sample.csv"
        history_parquet = tmp_path / "history.parquet"
        history_csv = tmp_path / "history.csv"
        _make_feature_dataset().to_parquet(feature_path, index=False)
        accounting_history = _make_accounting_history()
        accounting_history["gross_profit"] = accounting_history["gross_profit"].astype(object)
        accounting_history.loc[0, "gross_profit"] = pd.NA
        accounting_history["operating_income"] = accounting_history["operating_income"].astype(object)
        accounting_history.loc[1, "operating_income"] = pd.NA

        with (
            patch(
                "core.src.accounting_feature_engineering.main.build_accounting_history_for_universe",
                return_value=accounting_history,
            ),
            patch(
                "core.src.accounting_feature_engineering.main.ACCOUNTING_FEATURES_OUTPUT_PARQUET",
                output_parquet,
            ),
            patch(
                "core.src.accounting_feature_engineering.main.ACCOUNTING_FEATURES_OUTPUT_SAMPLE_CSV",
                output_csv,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_HISTORY_PARQUET",
                history_parquet,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_HISTORY_SAMPLE_CSV",
                history_csv,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_FEATURES_OUTPUT_PARQUET",
                output_parquet,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_FEATURES_OUTPUT_SAMPLE_CSV",
                output_csv,
            ),
        ):
            enriched = run_accounting_feature_engineering(
                feature_parquet_path=feature_path,
                max_workers=1,
            )

        assert "accounting_gross_margin_ttm" in enriched.columns
        assert "accounting_gross_margin_ttm_lag_1report" in enriched.columns

    def test_forward_fills_accounting_metrics_after_point_in_time_merge(self, tmp_path: Path) -> None:
        feature_path = tmp_path / "features.parquet"
        output_parquet = tmp_path / "accounting_features.parquet"
        output_csv = tmp_path / "accounting_features_sample.csv"
        history_parquet = tmp_path / "history.parquet"
        history_csv = tmp_path / "history.csv"
        _make_feature_dataset().to_parquet(feature_path, index=False)
        accounting_history = _make_accounting_history()
        accounting_history.loc[accounting_history["period_end"] == pd.Timestamp("2014-06-30"), "gross_profit"] = pd.NA

        with (
            patch(
                "core.src.accounting_feature_engineering.main.build_accounting_history_for_universe",
                return_value=accounting_history,
            ),
            patch(
                "core.src.accounting_feature_engineering.main.ACCOUNTING_FEATURES_OUTPUT_PARQUET",
                output_parquet,
            ),
            patch(
                "core.src.accounting_feature_engineering.main.ACCOUNTING_FEATURES_OUTPUT_SAMPLE_CSV",
                output_csv,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_HISTORY_PARQUET",
                history_parquet,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_HISTORY_SAMPLE_CSV",
                history_csv,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_FEATURES_OUTPUT_PARQUET",
                output_parquet,
            ),
            patch(
                "core.src.accounting_feature_engineering.io.ACCOUNTING_FEATURES_OUTPUT_SAMPLE_CSV",
                output_csv,
            ),
        ):
            enriched = run_accounting_feature_engineering(
                feature_parquet_path=feature_path,
                max_workers=1,
            )

        late_rows = enriched.loc[enriched["date"] >= pd.Timestamp("2014-08-20")]
        assert late_rows["accounting_gross_margin_ttm"].notna().any()
