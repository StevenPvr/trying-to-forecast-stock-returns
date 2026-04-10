from __future__ import annotations

"""Canonical MIQP portfolio solver for retail XTB execution."""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from pyscipopt import Model, quicksum

COEFFICIENT_EPSILON: float = 1e-12


@dataclass(frozen=True)
class PortfolioSolveInput:
    tickers: list[str]
    sectors: list[str]
    expected_return: np.ndarray
    covariance: np.ndarray
    existing_weight: np.ndarray
    existing_notional_eur: np.ndarray
    price_eur: np.ndarray
    cost_rate_proxy: np.ndarray
    adv_cap_shares: np.ndarray
    min_lot_shares: np.ndarray
    gross_cap_fraction: float
    max_position_weight: float
    max_sector_weight: float
    no_trade_buffer_rate: float
    lambda_risk: float
    lambda_turnover: float
    lambda_cost: float
    total_equity_eur: float
    cash_available_eur: float
    time_limit_seconds: float
    relative_gap: float
    primary_objective_tolerance_bps: float


@dataclass(frozen=True)
class PortfolioSolveResult:
    solver_status: str
    target_weight_total: np.ndarray
    target_weight_new: np.ndarray
    target_shares_new: np.ndarray
    target_notional_new_eur: np.ndarray
    expected_portfolio_return: float
    expected_portfolio_volatility: float
    expected_net_alpha: float
    constraint_binding_sector_share: float
    constraint_binding_liquidity_share: float
    cash_weight: float
    cash_amount_eur: float
    line_count_new: int
    solve_time_seconds: float
    mip_gap: float


@dataclass(frozen=True)
class _MiqpModelBundle:
    model: Any
    share_vars: list[Any]
    cash_var: Any
    primary_var: Any


def _empty_result(status: str) -> PortfolioSolveResult:
    empty = np.zeros(0, dtype=np.float64)
    return PortfolioSolveResult(
        solver_status=status,
        target_weight_total=empty,
        target_weight_new=empty,
        target_shares_new=np.zeros(0, dtype=np.int64),
        target_notional_new_eur=empty,
        expected_portfolio_return=0.0,
        expected_portfolio_volatility=0.0,
        expected_net_alpha=0.0,
        constraint_binding_sector_share=0.0,
        constraint_binding_liquidity_share=0.0,
        cash_weight=1.0,
        cash_amount_eur=0.0,
        line_count_new=0,
        solve_time_seconds=0.0,
        mip_gap=float("nan"),
    )


def _cash_only_result(
    status: str,
    *,
    cash_amount_eur: float,
    total_equity_eur: float,
) -> PortfolioSolveResult:
    empty = _empty_result(status)
    return PortfolioSolveResult(
        solver_status=empty.solver_status,
        target_weight_total=empty.target_weight_total,
        target_weight_new=empty.target_weight_new,
        target_shares_new=empty.target_shares_new,
        target_notional_new_eur=empty.target_notional_new_eur,
        expected_portfolio_return=empty.expected_portfolio_return,
        expected_portfolio_volatility=empty.expected_portfolio_volatility,
        expected_net_alpha=empty.expected_net_alpha,
        constraint_binding_sector_share=empty.constraint_binding_sector_share,
        constraint_binding_liquidity_share=empty.constraint_binding_liquidity_share,
        cash_weight=float(cash_amount_eur / max(total_equity_eur, 1e-12)),
        cash_amount_eur=float(cash_amount_eur),
        line_count_new=empty.line_count_new,
        solve_time_seconds=empty.solve_time_seconds,
        mip_gap=empty.mip_gap,
    )


def _build_primary_expression(
    inputs: PortfolioSolveInput,
    share_vars: list[Any],
) -> Any:
    asset_count = len(inputs.tickers)
    weight_per_share = np.divide(
        inputs.price_eur,
        max(inputs.total_equity_eur, 1e-12),
        dtype=np.float64,
    )
    covariance = 0.5 * (inputs.covariance + inputs.covariance.T)
    existing_risk_vector = covariance @ inputs.existing_weight
    expression: Any = 0.0
    for idx in range(asset_count):
        variable = share_vars[idx]
        linear_coefficient = float(
            inputs.expected_return[idx] * weight_per_share[idx]
            - inputs.lambda_turnover * weight_per_share[idx]
            - inputs.lambda_cost * inputs.cost_rate_proxy[idx] * weight_per_share[idx]
            - 2.0 * inputs.lambda_risk * existing_risk_vector[idx] * weight_per_share[idx]
        )
        if not math.isclose(linear_coefficient, 0.0, abs_tol=COEFFICIENT_EPSILON):
            expression += linear_coefficient * variable
    for row_index in range(asset_count):
        for column_index in range(row_index, asset_count):
            covariance_value = float(covariance[row_index, column_index])
            if math.isclose(covariance_value, 0.0, abs_tol=COEFFICIENT_EPSILON):
                continue
            quadratic_multiplier = 2.0 if row_index != column_index else 1.0
            quadratic_coefficient = float(
                -inputs.lambda_risk
                * quadratic_multiplier
                * covariance_value
                * weight_per_share[row_index]
                * weight_per_share[column_index]
            )
            if math.isclose(quadratic_coefficient, 0.0, abs_tol=COEFFICIENT_EPSILON):
                continue
            expression += (
                quadratic_coefficient
                * share_vars[row_index]
                * share_vars[column_index]
            )
    return expression


def _build_miqp_model(inputs: PortfolioSolveInput) -> _MiqpModelBundle:
    model = Model("portfolio_miqp")
    model.hideOutput()
    model.setParam("display/verblevel", 0)
    model.setParam("limits/time", float(inputs.time_limit_seconds))
    model.setParam("limits/gap", float(inputs.relative_gap))
    share_vars: list[Any] = []
    asset_count = len(inputs.tickers)
    gross_notional_limit_eur = float(inputs.gross_cap_fraction * inputs.total_equity_eur)
    total_existing_notional = float(inputs.existing_notional_eur.sum())
    budget_terms: list[Any] = []
    gross_terms: list[Any] = []
    sector_terms: dict[str, list[Any]] = {}
    weight_terms: list[Any] = []
    for idx in range(asset_count):
        ub_shares = int(max(int(inputs.adv_cap_shares[idx]), 0))
        position_headroom_weight = float(
            max(inputs.max_position_weight - float(inputs.existing_weight[idx]), 0.0),
        )
        price_eur = float(inputs.price_eur[idx])
        if price_eur > 0.0:
            position_headroom_shares = int(
                math.floor(position_headroom_weight * inputs.total_equity_eur / price_eur),
            )
        else:
            position_headroom_shares = 0
        ub_shares = max(0, min(ub_shares, position_headroom_shares))
        share_var = model.addVar(
            vtype="INTEGER",
            lb=0,
            ub=ub_shares,
            name=f"q_{idx}",
        )
        select_var = model.addVar(
            vtype="BINARY",
            lb=0.0,
            ub=0.0 if ub_shares == 0 else 1.0,
            name=f"z_{idx}",
        )
        share_vars.append(share_var)
        min_lot_shares = int(max(int(inputs.min_lot_shares[idx]), 1))
        if ub_shares > 0:
            model.addCons(share_var <= ub_shares * select_var)
            model.addCons(share_var >= min_lot_shares * select_var)
        notional_expr = price_eur * share_var
        budget_terms.append((1.0 + float(inputs.cost_rate_proxy[idx])) * notional_expr)
        gross_terms.append(notional_expr)
        weight_terms.append(notional_expr / max(inputs.total_equity_eur, 1e-12))
        sector_terms.setdefault(inputs.sectors[idx], []).append(notional_expr)
    cash_var = model.addVar(
        vtype="CONTINUOUS",
        lb=0.0,
        ub=max(float(inputs.cash_available_eur), 0.0),
        name="cash_eur",
    )
    primary_var = model.addVar(
        vtype="CONTINUOUS",
        lb=-1e6,
        ub=1e6,
        name="primary_objective",
    )
    model.addCons(quicksum(budget_terms) + cash_var == float(inputs.cash_available_eur))
    model.addCons(total_existing_notional + quicksum(gross_terms) <= gross_notional_limit_eur)
    if weight_terms:
        model.addCons(float(inputs.existing_weight.sum()) + quicksum(weight_terms) <= float(inputs.gross_cap_fraction))
    for sector_name, sector_exprs in sector_terms.items():
        current_sector_weight = sum(
            float(inputs.existing_weight[idx])
            for idx, candidate_sector in enumerate(inputs.sectors)
            if candidate_sector == sector_name
        )
        model.addCons(
            current_sector_weight + quicksum(sector_exprs) / max(inputs.total_equity_eur, 1e-12)
            <= float(inputs.max_sector_weight),
        )
    primary_expression = _build_primary_expression(inputs, share_vars)
    model.addCons(primary_var <= primary_expression)
    model.setObjective(primary_var, "maximize")
    return _MiqpModelBundle(
        model=model,
        share_vars=share_vars,
        cash_var=cash_var,
        primary_var=primary_var,
    )


def _solve_lexicographic_model(inputs: PortfolioSolveInput) -> tuple[_MiqpModelBundle, float]:
    primary_bundle = _build_miqp_model(inputs)
    primary_bundle.model.optimize()
    primary_status = str(primary_bundle.model.getStatus())
    if primary_status not in {"optimal", "timelimit", "gaplimit"}:
        return primary_bundle, float("nan")
    best_primary = float(primary_bundle.model.getObjVal())
    cash_bundle = _build_miqp_model(inputs)
    tolerance = float(inputs.primary_objective_tolerance_bps) / 10_000.0
    cash_bundle.model.addCons(cash_bundle.primary_var >= best_primary - tolerance)
    cash_bundle.model.setObjective(-cash_bundle.cash_var, "maximize")
    cash_bundle.model.optimize()
    cash_status = str(cash_bundle.model.getStatus())
    if cash_status not in {"optimal", "timelimit", "gaplimit"}:
        return primary_bundle, best_primary
    return cash_bundle, best_primary


def solve_portfolio_miqp(inputs: PortfolioSolveInput) -> PortfolioSolveResult:
    asset_count = len(inputs.tickers)
    if asset_count == 0:
        empty = _empty_result("empty_universe")
        return empty
    if inputs.total_equity_eur <= 0.0 or inputs.cash_available_eur <= 0.0:
        return _empty_result("no_capital")
    active_mask = (
        (inputs.price_eur > 0.0)
        & (inputs.adv_cap_shares > 0)
        & (inputs.min_lot_shares > 0)
        & ((inputs.expected_return - inputs.cost_rate_proxy - inputs.no_trade_buffer_rate) > 0.0)
    )
    if not bool(np.any(active_mask)):
        return _cash_only_result(
            "no_feasible_candidates",
            cash_amount_eur=float(inputs.cash_available_eur),
            total_equity_eur=float(inputs.total_equity_eur),
        )
    model_bundle, _ = _solve_lexicographic_model(inputs)
    status = str(model_bundle.model.getStatus())
    if status not in {"optimal", "timelimit", "gaplimit"}:
        return _empty_result(status)
    share_values = np.asarray(
        [int(round(float(model_bundle.model.getVal(variable)))) for variable in model_bundle.share_vars],
        dtype=np.int64,
    )
    target_notional_new_eur = inputs.price_eur * share_values.astype(np.float64)
    target_weight_new = np.divide(
        target_notional_new_eur,
        max(inputs.total_equity_eur, 1e-12),
        dtype=np.float64,
    )
    target_weight_total = inputs.existing_weight + target_weight_new
    expected_portfolio_return = float(inputs.expected_return @ target_weight_total)
    risk_variance = float(target_weight_total @ inputs.covariance @ target_weight_total)
    expected_portfolio_volatility = float(np.sqrt(max(risk_variance, 0.0)))
    expected_net_alpha = float((inputs.expected_return - inputs.cost_rate_proxy) @ target_weight_total)
    sector_binding_share = 0.0
    unique_sectors = sorted(set(inputs.sectors))
    if unique_sectors:
        sector_bindings = 0
        for sector_name in unique_sectors:
            sector_weight = float(
                sum(
                    target_weight_total[idx]
                    for idx, candidate_sector in enumerate(inputs.sectors)
                    if candidate_sector == sector_name
                ),
            )
            if sector_weight >= inputs.max_sector_weight - 1e-6:
                sector_bindings += 1
        sector_binding_share = float(sector_bindings / len(unique_sectors))
    liquidity_binding_share = float(
        np.mean(share_values >= (inputs.adv_cap_shares - 1))
    ) if asset_count > 0 else 0.0
    cash_amount_eur = float(max(model_bundle.model.getVal(model_bundle.cash_var), 0.0))
    cash_weight = float(cash_amount_eur / max(inputs.total_equity_eur, 1e-12))
    return PortfolioSolveResult(
        solver_status=status,
        target_weight_total=target_weight_total,
        target_weight_new=target_weight_new,
        target_shares_new=share_values,
        target_notional_new_eur=target_notional_new_eur,
        expected_portfolio_return=expected_portfolio_return,
        expected_portfolio_volatility=expected_portfolio_volatility,
        expected_net_alpha=expected_net_alpha,
        constraint_binding_sector_share=sector_binding_share,
        constraint_binding_liquidity_share=liquidity_binding_share,
        cash_weight=cash_weight,
        cash_amount_eur=cash_amount_eur,
        line_count_new=int(np.count_nonzero(share_values)),
        solve_time_seconds=float(model_bundle.model.getSolvingTime()),
        mip_gap=float(model_bundle.model.getGap()),
    )
