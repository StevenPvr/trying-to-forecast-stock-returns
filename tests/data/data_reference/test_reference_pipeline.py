from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_reference.reference_pipeline import (
    ReferenceBuildConfig,
    build_earnings_history,
    build_fundamentals_history,
    build_membership_history,
    build_reference_outputs,
    build_wrds_fundamentals_history,
    ensure_earnings_history_output,
    merge_wrds_with_bootstrap_fallback,
    resolve_fundamentals_source,
    save_reference_outputs,
)


def test_build_membership_history_merges_contiguous_intervals_and_normalizes_symbols() -> None:
    historical_components = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06"]),
            "tickers": [
                "AAPL,BF-B",
                "BF-B",
                "BF-B,MSFT",
            ],
        }
    )
    current_constituents = pd.DataFrame(
        {
            "ticker": ["AAPL", "BF.B", "MSFT"],
            "company_name": ["Apple Inc.", "Brown-Forman", "Microsoft Corporation"],
        }
    )

    result = build_membership_history(
        historical_components=historical_components,
        current_constituents=current_constituents,
        start_date="2020-01-01",
        end_date="2020-01-10",
    )

    expected = pd.DataFrame(
        {
            "ticker": ["AAPL", "BF.B", "MSFT"],
            "start_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-06"]),
            "end_date": pd.to_datetime(["2020-01-02", "2020-01-10", "2020-01-10"]),
            "company_name": ["Apple Inc.", "Brown-Forman", "Microsoft Corporation"],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_build_fundamentals_history_adds_structural_and_proxy_rows() -> None:
    membership_history = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "start_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "end_date": pd.to_datetime(["2025-12-31", "2025-12-31"]),
            "company_name": ["Apple Inc.", "Microsoft Corporation"],
        }
    )
    current_constituents = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "company_name": ["Apple Inc.", "Microsoft Corporation"],
            "sector": ["Information Technology", "Information Technology"],
            "industry": ["Technology Hardware", "Systems Software"],
        }
    )
    current_financials = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "price": [200.0],
            "market_cap": [3_000_000_000_000.0],
            "trailing_p_e": [30.0],
            "price_to_book": [10.0],
            "trailing_eps": [6.5],
        }
    )

    result = build_fundamentals_history(
        membership_history=membership_history,
        current_constituents=current_constituents,
        current_financials=current_financials,
        start_date="2020-01-01",
        end_date="2025-12-31",
    )

    aapl_rows = result.loc[result["ticker"] == "AAPL"].reset_index(drop=True)
    assert len(aapl_rows) == 2
    assert aapl_rows.loc[0, "date"] == pd.Timestamp("2020-01-01")
    assert aapl_rows.loc[0, "sector"] == "Information Technology"
    assert pd.isna(aapl_rows.loc[0, "market_cap"])
    assert aapl_rows.loc[1, "date"] == pd.Timestamp("2025-12-31")
    assert aapl_rows.loc[1, "market_cap"] == pytest.approx(3_000_000_000_000.0)
    assert aapl_rows.loc[1, "trailing_p_e"] == pytest.approx(30.0)
    assert aapl_rows.loc[1, "price_to_book"] == pytest.approx(10.0)
    assert aapl_rows.loc[1, "book_value"] == pytest.approx(20.0)
    assert aapl_rows.loc[1, "trailing_eps"] == pytest.approx(6.5)

    msft_rows = result.loc[result["ticker"] == "MSFT"].reset_index(drop=True)
    assert len(msft_rows) == 1
    assert msft_rows.loc[0, "date"] == pd.Timestamp("2020-01-01")
    assert msft_rows.loc[0, "sector"] == "Information Technology"
    assert pd.isna(msft_rows.loc[0, "market_cap"])


def test_build_wrds_fundamentals_history_uses_report_date_and_computes_ratios() -> None:
    membership_history = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "start_date": pd.to_datetime(["2020-01-01"]),
            "end_date": pd.to_datetime(["2025-12-31"]),
            "company_name": ["Apple Inc."],
        }
    )
    current_constituents = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "company_name": ["Apple Inc."],
            "sector": ["Information Technology"],
            "industry": ["Technology Hardware"],
        }
    )
    wrds_fundq = pd.DataFrame(
        {
            "tic": ["AAPL"],
            "rdq": pd.to_datetime(["2020-05-05"]),
            "datadate": pd.to_datetime(["2020-03-31"]),
            "prccq": [250.0],
            "cshoq": [4_000.0],
            "epspxq": [3.5],
            "ceqq": [80_000.0],
            "actq": [150_000.0],
            "lctq": [50_000.0],
            "niq": [20_000.0],
            "saleq": [100_000.0],
            "dlttq": [15_000.0],
            "dlcq": [5_000.0],
            "cheq": [8_000.0],
        }
    )

    result = build_wrds_fundamentals_history(
        membership_history=membership_history,
        current_constituents=current_constituents,
        wrds_fundq=wrds_fundq,
        start_date="2020-01-01",
        end_date="2025-12-31",
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["date"] == pd.Timestamp("2020-05-05")
    assert row["ticker"] == "AAPL"
    assert row["sector"] == "Information Technology"
    assert row["industry"] == "Technology Hardware"
    assert row["market_cap"] == pytest.approx(1_000_000_000_000.0)
    assert row["trailing_eps"] == pytest.approx(3.5)
    assert row["book_value"] == pytest.approx(20.0)
    assert row["price_to_book"] == pytest.approx(12.5)
    assert row["current_ratio"] == pytest.approx(3.0)
    assert row["profit_margins"] == pytest.approx(0.2)
    assert row["return_on_equity"] == pytest.approx(0.25)
    assert row["enterprise_value"] == pytest.approx(1_012_000_000_000.0)
    assert "forward_eps" not in result.columns
    assert "forward_p_e" not in result.columns


def test_merge_wrds_with_bootstrap_fallback_backfills_unresolved_tickers() -> None:
    membership_history = pd.DataFrame(
        {
            "ticker": ["AAPL", "ATGE"],
            "start_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "end_date": pd.to_datetime(["2025-12-31", "2025-12-31"]),
            "company_name": ["Apple Inc.", "Adtalem Global Education Inc."],
        }
    )
    current_constituents = pd.DataFrame(
        {
            "ticker": ["AAPL", "ATGE"],
            "company_name": ["Apple Inc.", "Adtalem Global Education Inc."],
            "sector": ["Information Technology", "Consumer Discretionary"],
            "industry": ["Technology Hardware", "Education Services"],
        }
    )
    current_financials = pd.DataFrame(
        {
            "ticker": ["ATGE"],
            "company_name": ["Adtalem Global Education Inc."],
            "price": [90.0],
            "market_cap": [4_500_000_000.0],
            "trailing_p_e": [20.0],
            "price_to_book": [3.0],
            "trailing_eps": [4.5],
        }
    )
    wrds_history = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-05-05")],
            "ticker": ["AAPL"],
            "company_name": ["Apple Inc."],
            "sector": ["Information Technology"],
            "industry": ["Technology Hardware"],
            "market_cap": [1_000_000_000_000.0],
            "trailing_p_e": [25.0],
            "price_to_book": [12.5],
            "beta": [pd.NA],
            "profit_margins": [0.2],
            "return_on_equity": [0.25],
            "enterprise_value": [pd.NA],
            "revenue_growth": [pd.NA],
            "current_ratio": [3.0],
            "book_value": [20.0],
            "trailing_eps": [3.5],
        }
    )

    result = merge_wrds_with_bootstrap_fallback(
        membership_history=membership_history,
        current_constituents=current_constituents,
        current_financials=current_financials,
        wrds_history=wrds_history,
        start_date="2020-01-01",
        end_date="2025-12-31",
    )

    assert set(result["ticker"]) == {"AAPL", "ATGE"}
    atge_rows = result.loc[result["ticker"] == "ATGE"].sort_values("date").reset_index(drop=True)
    assert len(atge_rows) == 2
    assert atge_rows.loc[0, "date"] == pd.Timestamp("2020-01-01")
    assert atge_rows.loc[1, "date"] == pd.Timestamp("2025-12-31")
    assert atge_rows.loc[1, "market_cap"] == pytest.approx(4_500_000_000.0)


def test_resolve_fundamentals_source_prefers_wrds_extracts(tmp_path: Path) -> None:
    config = ReferenceBuildConfig(
        wrds_fundq_extract_csv=tmp_path / "fundq.csv",
    )
    config.wrds_fundq_extract_csv.write_text("tic,rdq,datadate\nAAPL,2020-05-05,2020-03-31\n")

    with (
        patch(
            "core.src.meta_model.data.data_reference.reference_pipeline.wrds_package_available",
            return_value=False,
        ),
        patch(
            "core.src.meta_model.data.data_reference.reference_pipeline.resolve_wrds_credentials",
            return_value=None,
        ),
    ):
        assert resolve_fundamentals_source(config) == "wrds_extract"


def test_resolve_fundamentals_source_prefers_wrds_direct_when_available(tmp_path: Path) -> None:
    config = ReferenceBuildConfig(
        wrds_fundq_extract_csv=tmp_path / "fundq.csv",
    )
    config.wrds_fundq_extract_csv.write_text("tic,rdq,datadate\nAAPL,2020-05-05,2020-03-31\n")

    with (
        patch(
            "core.src.meta_model.data.data_reference.reference_pipeline.wrds_package_available",
            return_value=True,
        ),
        patch(
            "core.src.meta_model.data.data_reference.reference_pipeline.resolve_wrds_credentials",
            return_value=object(),
        ),
    ):
        assert resolve_fundamentals_source(config) == "wrds_direct"


def test_build_reference_outputs_prefers_wrds_extracts(tmp_path: Path) -> None:
    config = ReferenceBuildConfig(
        xtb_instrument_specs_json=tmp_path / "xtb.json",
        wrds_fundq_extract_csv=tmp_path / "fundq.csv",
    )
    config.xtb_instrument_specs_json.write_text(
        '[{"symbol":"AAPL","instrument_group":"stock_cfd","currency":"USD","spread_bps":0.0,"slippage_bps":0.0,"long_swap_bps_daily":0.0,"short_swap_bps_daily":0.0,"margin_requirement":1.0,"max_adv_participation":0.05,"effective_from":"2000-01-01","effective_to":null}]',
        encoding="utf-8",
    )
    config.wrds_fundq_extract_csv.write_text(
        "tic,rdq,datadate,prccq,cshoq,epspxq,ceqq,actq,lctq,niq,saleq\n"
        "AAPL,2020-05-05,2020-03-31,250,4000,3.5,80000,150000,50000,20000,100000\n",
        encoding="utf-8",
    )

    historical_components = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "tickers": ["AAPL"],
        }
    )
    current_constituents = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "company_name": ["Apple Inc."],
            "sector": ["Information Technology"],
            "industry": ["Technology Hardware"],
        }
    )

    membership_history, fundamentals_history = build_reference_outputs(
        config=config,
        historical_components=historical_components,
        current_constituents=current_constituents,
        current_financials=pd.DataFrame(),
    )

    assert len(membership_history) == 1


def test_build_earnings_history_uses_after_close_and_calendar_quarters() -> None:
    fundamentals_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-14", "2024-05-15", "2024-05-15"]),
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "company_name": ["Apple", "Apple", "Microsoft"],
        }
    )

    result = build_earnings_history(fundamentals_history)

    expected = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "announcement_date": ["2024-02-14", "2024-05-15", "2024-05-15"],
            "announcement_session": ["after_close", "after_close", "after_close"],
            "fiscal_year": [2024, 2024, 2024],
            "fiscal_quarter": [1, 2, 2],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_save_reference_outputs_persists_earnings_history(tmp_path: Path) -> None:
    membership_history = pd.DataFrame({"ticker": ["AAPL"], "start_date": ["2024-01-01"], "end_date": ["2024-12-31"]})
    fundamentals_history = pd.DataFrame({"date": ["2024-02-14"], "ticker": ["AAPL"]})
    earnings_history = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "announcement_date": ["2024-02-14"],
            "announcement_session": ["after_close"],
            "fiscal_year": [2024],
            "fiscal_quarter": [1],
        }
    )
    membership_output = tmp_path / "membership.csv"
    fundamentals_output = tmp_path / "fundamentals.csv"
    earnings_output = tmp_path / "earnings.csv"

    output_paths = save_reference_outputs(
        membership_history=membership_history,
        fundamentals_history=fundamentals_history,
        earnings_history=earnings_history,
        membership_output_csv=membership_output,
        fundamentals_output_csv=fundamentals_output,
        earnings_output_csv=earnings_output,
    )

    assert output_paths["earnings_history_csv"] == earnings_output
    saved = pd.read_csv(earnings_output)
    assert saved.loc[0, "announcement_session"] == "after_close"
    saved_fundamentals = pd.read_csv(fundamentals_output)
    assert len(saved_fundamentals) == 1
    assert saved_fundamentals.loc[0, "date"] == "2024-02-14"


def test_ensure_earnings_history_output_rebuilds_canonical_csv(tmp_path: Path) -> None:
    fundamentals_path = tmp_path / "fundamentals.csv"
    earnings_output = tmp_path / "earnings.csv"
    pd.DataFrame(
        {
            "date": ["2024-02-14", "2024-05-15"],
            "ticker": ["AAPL", "AAPL"],
            "company_name": ["Apple", "Apple"],
        }
    ).to_csv(fundamentals_path, index=False)

    output_path = ensure_earnings_history_output(
        fundamentals_history_path=fundamentals_path,
        earnings_output_csv=earnings_output,
    )

    saved = pd.read_csv(output_path)
    assert output_path == earnings_output
    assert saved["announcement_session"].tolist() == ["after_close", "after_close"]


def test_build_reference_outputs_falls_back_when_wrds_direct_fails(tmp_path: Path) -> None:
    config = ReferenceBuildConfig(
        xtb_instrument_specs_json=tmp_path / "xtb.json",
    )
    config.xtb_instrument_specs_json.write_text(
        '[{"symbol":"AAPL","instrument_group":"stock_cfd","currency":"USD","spread_bps":0.0,"slippage_bps":0.0,"long_swap_bps_daily":0.0,"short_swap_bps_daily":0.0,"margin_requirement":1.0,"max_adv_participation":0.05,"effective_from":"2000-01-01","effective_to":null}]',
        encoding="utf-8",
    )
    historical_components = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "tickers": ["AAPL"],
        }
    )
    current_constituents = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "company_name": ["Apple Inc."],
            "sector": ["Information Technology"],
            "industry": ["Technology Hardware"],
        }
    )
    current_financials = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "company_name": ["Apple Inc."],
            "price": [200.0],
            "market_cap": [3_000_000_000_000.0],
            "trailing_p_e": [30.0],
            "price_to_book": [10.0],
            "trailing_eps": [6.5],
        }
    )

    with (
        patch(
            "core.src.meta_model.data.data_reference.reference_pipeline.wrds_package_available",
            return_value=True,
        ),
        patch(
            "core.src.meta_model.data.data_reference.reference_pipeline.resolve_wrds_credentials",
            return_value=object(),
        ),
        patch(
            "core.src.meta_model.data.data_reference.reference_pipeline.fetch_fundq_history",
            side_effect=RuntimeError("auth failed"),
        ),
    ):
        _, fundamentals_history = build_reference_outputs(
            config=config,
            historical_components=historical_components,
            current_constituents=current_constituents,
            current_financials=current_financials,
        )

    assert fundamentals_history["date"].iloc[0] == pd.Timestamp(config.start_date)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
