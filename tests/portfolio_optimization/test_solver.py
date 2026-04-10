from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.portfolio_optimization.solver import (
    PortfolioSolveInput,
    solve_portfolio_miqp,
)


def test_solve_portfolio_miqp_produces_integer_share_counts_and_full_investment() -> None:
    result = solve_portfolio_miqp(
        PortfolioSolveInput(
            tickers=["AAA", "BBB"],
            sectors=["Tech", "Tech"],
            expected_return=np.asarray([0.20, 0.10], dtype=np.float64),
            covariance=np.asarray([[0.01, 0.0], [0.0, 0.01]], dtype=np.float64),
            existing_weight=np.asarray([0.0, 0.0], dtype=np.float64),
            existing_notional_eur=np.asarray([0.0, 0.0], dtype=np.float64),
            price_eur=np.asarray([600.0, 400.0], dtype=np.float64),
            cost_rate_proxy=np.asarray([0.0, 0.0], dtype=np.float64),
            adv_cap_shares=np.asarray([5, 5], dtype=np.int64),
            min_lot_shares=np.asarray([1, 1], dtype=np.int64),
            gross_cap_fraction=1.0,
            max_position_weight=1.0,
            max_sector_weight=1.0,
            no_trade_buffer_rate=0.0,
            lambda_risk=0.0,
            lambda_turnover=0.0,
            lambda_cost=0.0,
            total_equity_eur=1_000.0,
            cash_available_eur=1_000.0,
            time_limit_seconds=1.0,
            relative_gap=0.0,
            primary_objective_tolerance_bps=0.5,
        ),
    )

    assert result.solver_status == "optimal"
    assert result.target_shares_new.tolist() == [1, 1]
    assert result.cash_amount_eur == pytest.approx(0.0, abs=1e-6)
    assert result.cash_weight == pytest.approx(0.0, abs=1e-6)


def test_solve_portfolio_miqp_respects_budget_when_expensive_asset_is_unaffordable() -> None:
    result = solve_portfolio_miqp(
        PortfolioSolveInput(
            tickers=["EXPENSIVE", "CHEAP"],
            sectors=["Tech", "Tech"],
            expected_return=np.asarray([0.40, 0.08], dtype=np.float64),
            covariance=np.asarray([[0.01, 0.0], [0.0, 0.01]], dtype=np.float64),
            existing_weight=np.asarray([0.0, 0.0], dtype=np.float64),
            existing_notional_eur=np.asarray([0.0, 0.0], dtype=np.float64),
            price_eur=np.asarray([1_200.0, 400.0], dtype=np.float64),
            cost_rate_proxy=np.asarray([0.0, 0.0], dtype=np.float64),
            adv_cap_shares=np.asarray([5, 5], dtype=np.int64),
            min_lot_shares=np.asarray([1, 1], dtype=np.int64),
            gross_cap_fraction=1.0,
            max_position_weight=1.0,
            max_sector_weight=1.0,
            no_trade_buffer_rate=0.0,
            lambda_risk=0.0,
            lambda_turnover=0.0,
            lambda_cost=0.0,
            total_equity_eur=1_000.0,
            cash_available_eur=1_000.0,
            time_limit_seconds=1.0,
            relative_gap=0.0,
            primary_objective_tolerance_bps=0.5,
        ),
    )

    assert result.solver_status == "optimal"
    assert result.target_shares_new.tolist() == [0, 2]
    assert result.cash_amount_eur == pytest.approx(200.0, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
