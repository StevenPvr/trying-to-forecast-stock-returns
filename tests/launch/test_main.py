from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.launch.main import build_launch_readiness_report


def test_launch_readiness_reports_missing_reference_files(
    tmp_path: Path,
) -> None:
    with (
        patch(
            "core.src.meta_model.launch.main.MEMBERSHIP_HISTORY_CSV",
            tmp_path / "missing_membership.csv",
        ),
        patch(
            "core.src.meta_model.launch.main.FUNDAMENTALS_HISTORY_CSV",
            tmp_path / "missing_fundamentals.csv",
        ),
        patch(
            "core.src.meta_model.launch.main.XTB_INSTRUMENT_SPECS_REFERENCE_JSON",
            tmp_path / "missing_xtb_snapshot.json",
        ),
    ):
        report = build_launch_readiness_report()

    assert report.is_ready is False
    assert len(report.missing_paths) == 3


def test_launch_readiness_passes_with_required_files_present(
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "sp500_membership_history.csv"
    fundamentals_path = tmp_path / "sp500_fundamentals_history.csv"
    xtb_snapshot_path = tmp_path / "xtb_instrument_specs.json"
    membership_path.write_text("ticker,start_date,end_date\nAAPL,2024-01-01,2024-12-31\n")
    fundamentals_path.write_text("date,ticker,company_market_cap_usd\n2024-01-02,AAPL,100\n")
    xtb_snapshot_path.write_text(
        '[{"symbol":"AAPL","instrument_group":"stock_cash","currency":"USD","spread_bps":0.0,"slippage_bps":0.0,"long_swap_bps_daily":0.0,"short_swap_bps_daily":0.0,"margin_requirement":1.0,"max_adv_participation":0.05,"effective_from":"2000-01-01","effective_to":null}]',
        encoding="utf-8",
    )

    with (
        patch(
            "core.src.meta_model.launch.main.MEMBERSHIP_HISTORY_CSV",
            membership_path,
        ),
        patch(
            "core.src.meta_model.launch.main.FUNDAMENTALS_HISTORY_CSV",
            fundamentals_path,
        ),
        patch(
            "core.src.meta_model.launch.main.XTB_INSTRUMENT_SPECS_REFERENCE_JSON",
            xtb_snapshot_path,
        ),
        patch.dict("os.environ", {"FRED_API_KEY": "x" * 32}, clear=True),
    ):
        report = build_launch_readiness_report()

    assert report.is_ready is True
    assert report.missing_paths == []
    assert report.stock_cash_count == 1
    assert report.fred_api_key_available is True


def test_launch_readiness_requires_fred_api_key(
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "sp500_membership_history.csv"
    fundamentals_path = tmp_path / "sp500_fundamentals_history.csv"
    xtb_snapshot_path = tmp_path / "xtb_instrument_specs.json"
    membership_path.write_text("ticker,start_date,end_date\nAAPL,2024-01-01,2024-12-31\n")
    fundamentals_path.write_text("date,ticker,company_market_cap_usd\n2024-01-02,AAPL,100\n")
    xtb_snapshot_path.write_text(
        '[{"symbol":"AAPL","instrument_group":"stock_cash","currency":"USD","spread_bps":0.0,"slippage_bps":0.0,"long_swap_bps_daily":0.0,"short_swap_bps_daily":0.0,"margin_requirement":1.0,"max_adv_participation":0.05,"effective_from":"2000-01-01","effective_to":null}]',
        encoding="utf-8",
    )

    with (
        patch(
            "core.src.meta_model.launch.main.MEMBERSHIP_HISTORY_CSV",
            membership_path,
        ),
        patch(
            "core.src.meta_model.launch.main.FUNDAMENTALS_HISTORY_CSV",
            fundamentals_path,
        ),
        patch(
            "core.src.meta_model.launch.main.XTB_INSTRUMENT_SPECS_REFERENCE_JSON",
            xtb_snapshot_path,
        ),
        patch.dict("os.environ", {}, clear=True),
    ):
        report = build_launch_readiness_report()

    assert report.is_ready is False
    assert report.fred_api_key_available is False


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
