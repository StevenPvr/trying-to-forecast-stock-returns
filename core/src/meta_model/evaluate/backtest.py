from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
import time
from typing import Any, Mapping, TypeAlias, cast

import numpy as np
import pandas as pd

from core.src.meta_model.broker_xtb.costs import BrokerCostEstimate, estimate_trade_cost
from core.src.meta_model.broker_xtb.specs import (
    BrokerSpecProvider,
    XtbInstrumentSpec,
    build_default_spec_provider,
)
from core.src.meta_model.evaluate.config import STARTING_CASH_EUR
from core.src.meta_model.portfolio_optimization.solver import (
    PortfolioSolveInput,
    solve_portfolio_miqp,
)
from core.src.meta_model.model_contract import (
    DATE_COLUMN,
    PREDICTION_COLUMN,
    REALIZED_RETURN_COLUMN,
    TICKER_COLUMN,
)

EXPECTED_RETURN_COLUMN: str = "expected_return_5d"
DiagnosticValue: TypeAlias = str | float


def _format_duration(seconds: float) -> str:
    rounded_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(rounded_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


@dataclass(frozen=True)
class PortfolioOptimizerArtifacts:
    covariance: pd.DataFrame


@dataclass(frozen=True)
class ActiveTrade:
    ticker: str
    side: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    notional: float
    predicted_return: float
    realized_log_return: float
    signal_rank: int
    spec: XtbInstrumentSpec
    share_count: int = 0
    reference_price_eur: float = 0.0
    entry_transaction_cost_amount: float = 0.0
    entry_commission_amount: float = 0.0
    entry_fx_conversion_amount: float = 0.0
    expected_entry_cost_rate: float = 0.0
    capacity_bound: bool = False


@dataclass(frozen=True)
class ClosedTrade:
    ticker: str
    side: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    notional: float
    predicted_return: float
    realized_log_return: float
    signal_rank: int
    gross_return: float
    transaction_cost: float
    net_return: float
    pnl_amount: float
    exit_cash_flow_amount: float
    share_count: int = 0
    reference_price_eur: float = 0.0
    gross_pnl_amount: float = 0.0
    entry_transaction_cost_amount: float = 0.0
    exit_transaction_cost_amount: float = 0.0
    total_transaction_cost_amount: float = 0.0
    entry_commission_amount: float = 0.0
    exit_commission_amount: float = 0.0
    total_commission_amount: float = 0.0
    entry_fx_conversion_amount: float = 0.0
    exit_fx_conversion_amount: float = 0.0
    total_fx_conversion_amount: float = 0.0
    net_pnl_amount: float = 0.0


@dataclass(frozen=True)
class XtbCostConfig:
    account_currency: str = "EUR"
    broker_spec_provider: BrokerSpecProvider = field(default_factory=build_default_spec_provider)


@dataclass(frozen=True)
class OptimizerAllocationRequest:
    trade_date: pd.Timestamp
    daily_predictions: pd.DataFrame
    active_trades: list[ActiveTrade]
    current_equity: float
    cash_balance: float
    unique_dates: pd.Index
    month_to_date_turnover_eur: float
    hold_period_days: int
    gross_cap_fraction: float
    adv_participation_limit: float
    open_hurdle_bps: float
    account_currency: str
    cost_config: XtbCostConfig
    artifacts: PortfolioOptimizerArtifacts
    lambda_risk: float
    lambda_turnover: float
    lambda_cost: float
    max_position_weight: float
    max_sector_weight: float
    min_target_weight: float
    no_trade_buffer_bps: float
    miqp_time_limit_seconds: float
    miqp_relative_gap: float
    miqp_candidate_pool_size: int
    miqp_primary_objective_tolerance_bps: float


@dataclass(frozen=True)
class BacktestRuntimeConfig:
    unique_dates: pd.Index
    top_fraction: float
    allocation_fraction: float
    action_cap_fraction: float
    gross_cap_fraction: float
    adv_participation_limit: float
    neutrality_mode: str
    open_hurdle_bps: float
    apply_prediction_hurdle: bool
    hold_period_days: int
    cost_config: XtbCostConfig
    portfolio_construction_mode: str
    optimizer_artifacts: PortfolioOptimizerArtifacts | None
    lambda_risk: float
    lambda_turnover: float
    lambda_cost: float
    max_position_weight: float
    max_sector_weight: float
    min_target_weight: float
    no_trade_buffer_bps: float
    miqp_time_limit_seconds: float
    miqp_relative_gap: float
    miqp_candidate_pool_size: int
    miqp_primary_objective_tolerance_bps: float
    benchmark_returns_by_date: dict[pd.Timestamp, float] = field(default_factory=lambda: {})


@dataclass(frozen=True)
class BacktestRunConfig:
    runtime: BacktestRuntimeConfig
    starting_cash_eur: float = STARTING_CASH_EUR


@dataclass(frozen=True)
class BacktestProgressConfig:
    logger: Any | None = None
    progress_label: str | None = None
    progress_log_every: int = 0


@dataclass(frozen=True)
class DailyAccountingSnapshot:
    trade_date: pd.Timestamp
    opening_equity: float
    opening_cash: float
    realized_pnl_today: float
    gross_pnl_exits: float
    entry_cost_amount: float
    exit_cost_amount: float
    custody_fee_amount: float
    opened_notional: float
    closed_notional: float
    closed_trade_count: float
    active_notional_end: float


@dataclass
class BacktestState:
    initial_equity: float = STARTING_CASH_EUR
    current_equity: float = STARTING_CASH_EUR
    cash_balance: float = STARTING_CASH_EUR
    active_trades: list[ActiveTrade] = field(default_factory=lambda: [])
    closed_trades: list[ClosedTrade] = field(default_factory=lambda: [])
    daily_rows: list[dict[str, object]] = field(default_factory=lambda: [])
    allocation_rows: list[dict[str, object]] = field(default_factory=lambda: [])
    optimizer_daily_rows: list[dict[str, object]] = field(default_factory=lambda: [])
    monthly_turnover_eur: dict[str, float] = field(default_factory=lambda: {})
    monthly_market_values_eur: dict[str, list[float]] = field(default_factory=lambda: {})


def _month_key(trade_date: pd.Timestamp) -> str:
    return f"{trade_date.year:04d}-{trade_date.month:02d}"


def _resolve_month_to_date_turnover(state: BacktestState, trade_date: pd.Timestamp) -> float:
    return float(state.monthly_turnover_eur.get(_month_key(trade_date), 0.0))


def _register_order_turnover(state: BacktestState, trade_date: pd.Timestamp, order_value_eur: float) -> None:
    month_key = _month_key(trade_date)
    state.monthly_turnover_eur[month_key] = _resolve_month_to_date_turnover(state, trade_date) + max(order_value_eur, 0.0)


def _resolve_adv_cap_notional(
    *,
    open_price: float,
    trading_volume: float,
    max_adv_participation: float,
) -> float:
    if open_price <= 0.0 or trading_volume <= 0.0:
        return float("inf")
    return max(0.0, open_price * trading_volume * max_adv_participation)


def _resolve_exit_date(
    trade_date: pd.Timestamp,
    unique_dates: pd.Index,
    hold_period_days: int,
) -> pd.Timestamp | None:
    matching_positions = np.flatnonzero(unique_dates == trade_date)
    if matching_positions.size == 0:
        return None
    exit_position = int(matching_positions[0]) + hold_period_days
    if exit_position >= len(unique_dates):
        return None
    return _coerce_timestamp(unique_dates.tolist()[exit_position])


def _coerce_timestamp(value: object) -> pd.Timestamp | None:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (str, int, float, dt.date, dt.datetime, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        return timestamp
    return None


def _resolve_price_value(row: pd.Series, column_name: str, fallback_name: str) -> float:
    row_values = {str(key): value for key, value in row.to_dict().items()}
    primary_value = row_values.get(column_name)
    if primary_value is not None and pd.notna(primary_value):
        return _scalar_float(primary_value)
    fallback_value = row_values.get(fallback_name)
    if fallback_value is not None and pd.notna(fallback_value):
        return _scalar_float(fallback_value)
    return 0.0


def _resolve_daily_expected_return(
    daily_predictions: pd.DataFrame,
) -> pd.Series:
    if EXPECTED_RETURN_COLUMN in daily_predictions.columns:
        expected_series = daily_predictions.loc[:, EXPECTED_RETURN_COLUMN]
        return pd.to_numeric(
            expected_series,
            errors="coerce",
        ).fillna(0.0)
    prediction_series = daily_predictions.loc[:, PREDICTION_COLUMN]
    return pd.to_numeric(
        prediction_series,
        errors="coerce",
    ).fillna(0.0)


def _normalize_numeric_scalar(value: object) -> int | float:
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid numeric scalars.")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Unsupported numeric scalar type: {type(value)!r}")


def _scalar_float(value: object) -> float:
    return float(_normalize_numeric_scalar(value))


def _scalar_int(value: object) -> int:
    normalized = _normalize_numeric_scalar(value)
    if isinstance(normalized, float):
        if not normalized.is_integer():
            raise TypeError("Float scalar cannot be converted losslessly to int.")
        return int(normalized)
    return int(normalized)


def _build_empty_optimizer_diagnostics(
    *,
    solver_status: str,
    cash_balance: float,
    current_equity: float,
) -> dict[str, DiagnosticValue]:
    return {
        "solver_status": solver_status,
        "candidate_count": 0.0,
        "tradable_count": 0.0,
        "expected_portfolio_return": 0.0,
        "expected_portfolio_volatility": 0.0,
        "expected_net_alpha": 0.0,
        "constraint_binding_sector_share": 0.0,
        "constraint_binding_liquidity_share": 0.0,
        "cash_weight": max(0.0, cash_balance / max(current_equity, 1e-12)),
        "cash_amount_eur": float(cash_balance),
        "solve_time_seconds": 0.0,
        "mip_gap": float("nan"),
        "line_count_new": 0.0,
        "integer_shares_bought_total": 0.0,
    }


def _build_ordered_predictions(daily_predictions: pd.DataFrame) -> pd.DataFrame:
    ordered = pd.DataFrame(daily_predictions.copy())
    ordered["_expected_return_5d"] = _resolve_daily_expected_return(daily_predictions).to_numpy(
        dtype=np.float64,
    )
    return ordered.sort_values("_expected_return_5d", ascending=False).reset_index(drop=True)


def _build_buy_and_hold_benchmark_returns(
    predictions: pd.DataFrame,
) -> pd.Series:
    if predictions.empty:
        return pd.Series(dtype=np.float64)
    if "hl_context_stock_close_price" not in predictions.columns:
        raise ValueError(
            "Missing required benchmark column: hl_context_stock_close_price",
        )
    close_matrix = predictions.pivot(
        index=DATE_COLUMN,
        columns=TICKER_COLUMN,
        values="hl_context_stock_close_price",
    ).sort_index()
    if close_matrix.empty:
        return pd.Series(dtype=np.float64)
    first_date = cast(pd.Timestamp, pd.Timestamp(close_matrix.index.min()))
    starting_prices = pd.to_numeric(close_matrix.loc[first_date], errors="coerce")
    eligible_mask = starting_prices.notna() & (starting_prices > 0.0)
    eligible_tickers = starting_prices.index[eligible_mask]
    if len(eligible_tickers) == 0:
        return pd.Series(0.0, index=close_matrix.index, dtype=np.float64)
    filtered_prices = pd.DataFrame(close_matrix.loc[:, eligible_tickers].copy()).ffill()
    close_returns = filtered_prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    constituent_wealth = (1.0 + close_returns).cumprod()
    drifted_weights = constituent_wealth.shift(1)
    drifted_weights.iloc[0] = 1.0
    normalized_weights = drifted_weights.div(drifted_weights.sum(axis=1), axis=0).fillna(0.0)
    benchmark_returns = (normalized_weights * close_returns).sum(axis=1)
    return pd.Series(benchmark_returns, index=close_matrix.index, dtype=np.float64)


def _build_sector_series(ordered: pd.DataFrame) -> pd.Series:
    if "company_sector" not in ordered.columns:
        return pd.Series(["__unknown_sector__"] * len(ordered), dtype="object")
    sector_series = ordered.loc[:, "company_sector"]
    return sector_series.astype(str).fillna("__unknown_sector__")


def _resolve_adv_cap_shares(
    *,
    open_price: float,
    adv_cap_scaled: float,
    current_equity: float,
    max_position_weight: float,
) -> int:
    adv_cap_shares = 0
    if np.isfinite(adv_cap_scaled) and adv_cap_scaled > 0.0:
        adv_cap_shares = int(math.floor(adv_cap_scaled / open_price))
    position_headroom_shares = int(math.floor(max_position_weight * current_equity / open_price))
    return max(0, min(adv_cap_shares, position_headroom_shares))


def _build_candidate_row(
    *,
    request: OptimizerAllocationRequest,
    row: pd.Series,
    sector: str,
    projected_month_turnover: float,
    no_trade_buffer_rate: float,
) -> dict[str, object] | None:
    row_values = {str(key): value for key, value in row.to_dict().items()}
    ticker = str(row_values[TICKER_COLUMN])
    spec = request.cost_config.broker_spec_provider.resolve(ticker, request.trade_date)
    open_price = _resolve_price_value(row, "stock_open_price", "hl_context_stock_open_price")
    trading_volume = _resolve_price_value(
        row,
        "stock_trading_volume",
        "hl_context_stock_trading_volume",
    )
    if open_price <= 0.0:
        return None
    adv_cap_notional = _resolve_adv_cap_notional(
        open_price=open_price,
        trading_volume=trading_volume,
        max_adv_participation=spec.max_adv_participation,
    )
    adv_cap_scaled = adv_cap_notional * max(request.adv_participation_limit / 0.05, 0.0)
    adv_cap_shares = _resolve_adv_cap_shares(
        open_price=open_price,
        adv_cap_scaled=adv_cap_scaled,
        current_equity=request.current_equity,
        max_position_weight=request.max_position_weight,
    )
    min_lot_shares = int(max(math.ceil(spec.minimum_order_value_eur / open_price), 1))
    estimated = estimate_trade_cost(
        spec,
        order_value_eur=max(open_price * min_lot_shares, spec.minimum_order_value_eur),
        month_to_date_turnover_eur=projected_month_turnover,
        account_currency=request.account_currency,
    )
    expected_return = _scalar_float(row_values["_expected_return_5d"])
    net_edge = expected_return - float(estimated.total_cost_rate) - no_trade_buffer_rate
    if adv_cap_shares < min_lot_shares or net_edge <= 0.0:
        return None
    return {
        "ticker": ticker,
        "sector": sector,
        "expected_return_5d": expected_return,
        "cost_rate_proxy": float(estimated.total_cost_rate),
        "price_eur": float(open_price),
        "adv_cap_shares": adv_cap_shares,
        "min_lot_shares": min_lot_shares,
        "prediction": _scalar_float(row_values[PREDICTION_COLUMN]),
        "realized_log_return": _scalar_float(row_values[REALIZED_RETURN_COLUMN]),
        "spec": spec,
        "net_edge": net_edge,
    }


def _build_candidate_frame(
    request: OptimizerAllocationRequest,
    *,
    no_trade_buffer_rate: float,
) -> tuple[pd.DataFrame, int]:
    ordered = _build_ordered_predictions(request.daily_predictions)
    if ordered.empty:
        return pd.DataFrame(), 0
    sectors = _build_sector_series(ordered).astype(str).tolist()
    projected_month_turnover = max(request.month_to_date_turnover_eur, 0.0)
    candidate_rows: list[dict[str, object]] = []
    for idx, row in ordered.iterrows():
        candidate_row = _build_candidate_row(
            request=request,
            row=row,
            sector=sectors[idx],
            projected_month_turnover=projected_month_turnover,
            no_trade_buffer_rate=no_trade_buffer_rate,
        )
        if candidate_row is not None:
            candidate_rows.append(candidate_row)
    candidate_count = len(candidate_rows)
    if candidate_count == 0:
        return pd.DataFrame(), 0
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame = candidate_frame.sort_values("net_edge", ascending=False)
    candidate_frame = candidate_frame.head(request.miqp_candidate_pool_size).reset_index(drop=True)
    return candidate_frame, candidate_count


def _build_existing_position_vectors(
    *,
    tickers: list[str],
    active_trades: list[ActiveTrade],
    current_equity: float,
) -> tuple[np.ndarray, np.ndarray]:
    existing_notional_eur = np.zeros(len(tickers), dtype=np.float64)
    existing_weight = np.zeros(len(tickers), dtype=np.float64)
    for idx, ticker in enumerate(tickers):
        existing_notional = sum(trade.notional for trade in active_trades if trade.ticker == ticker)
        existing_notional_eur[idx] = max(0.0, existing_notional)
        existing_weight[idx] = max(0.0, existing_notional / max(current_equity, 1e-12))
    return existing_notional_eur, existing_weight


def _build_covariance_psd(*, covariance: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    dense_covariance = covariance.reindex(index=tickers, columns=tickers).to_numpy(dtype=np.float64)
    dense_covariance = np.nan_to_num(dense_covariance, nan=0.0, posinf=0.0, neginf=0.0)
    symmetric_cov = 0.5 * (dense_covariance + dense_covariance.T)
    eigvals, eigvecs = np.linalg.eigh(symmetric_cov)
    return eigvecs @ np.diag(np.clip(eigvals, 1e-8, None)) @ eigvecs.T


def _build_solver_input(
    request: OptimizerAllocationRequest,
    *,
    candidate_frame: pd.DataFrame,
    no_trade_buffer_rate: float,
) -> PortfolioSolveInput:
    ticker_series = candidate_frame.loc[:, "ticker"]
    sector_series = candidate_frame.loc[:, "sector"]
    expected_return_series = candidate_frame.loc[:, "expected_return_5d"]
    price_series = candidate_frame.loc[:, "price_eur"]
    cost_rate_series = candidate_frame.loc[:, "cost_rate_proxy"]
    adv_cap_series = candidate_frame.loc[:, "adv_cap_shares"]
    min_lot_series = candidate_frame.loc[:, "min_lot_shares"]
    tickers = ticker_series.astype(str).tolist()
    existing_notional_eur, existing_weight = _build_existing_position_vectors(
        tickers=tickers,
        active_trades=request.active_trades,
        current_equity=request.current_equity,
    )
    return PortfolioSolveInput(
        tickers=tickers,
        sectors=sector_series.astype(str).tolist(),
        expected_return=expected_return_series.to_numpy(dtype=np.float64),
        covariance=_build_covariance_psd(covariance=request.artifacts.covariance, tickers=tickers),
        existing_weight=existing_weight,
        existing_notional_eur=existing_notional_eur,
        price_eur=price_series.to_numpy(dtype=np.float64),
        cost_rate_proxy=cost_rate_series.to_numpy(dtype=np.float64),
        adv_cap_shares=adv_cap_series.to_numpy(dtype=np.int64),
        min_lot_shares=min_lot_series.to_numpy(dtype=np.int64),
        gross_cap_fraction=request.gross_cap_fraction,
        max_position_weight=request.max_position_weight,
        max_sector_weight=request.max_sector_weight,
        no_trade_buffer_rate=no_trade_buffer_rate,
        lambda_risk=request.lambda_risk,
        lambda_turnover=request.lambda_turnover,
        lambda_cost=request.lambda_cost,
        total_equity_eur=request.current_equity,
        cash_available_eur=request.cash_balance,
        time_limit_seconds=request.miqp_time_limit_seconds,
        relative_gap=request.miqp_relative_gap,
        primary_objective_tolerance_bps=request.miqp_primary_objective_tolerance_bps,
    )


def _build_allocation_row(
    *,
    trade_date: pd.Timestamp,
    ticker: str,
    candidate_row: pd.Series,
    target_weight: float,
    target_notional: float,
    target_shares: int,
) -> dict[str, object]:
    candidate_values = {str(key): value for key, value in candidate_row.to_dict().items()}
    return {
        "date": trade_date,
        "ticker": ticker,
        "expected_return_5d": _scalar_float(candidate_values["expected_return_5d"]),
        "target_weight": target_weight,
        "target_notional_eur": target_notional,
        "target_shares": target_shares,
        "price_eur": _scalar_float(candidate_values["price_eur"]),
        "sector": str(candidate_values["sector"]),
        "cost_rate_proxy": _scalar_float(candidate_values["cost_rate_proxy"]),
        "selected_by_optimizer": True,
    }


def _materialize_optimizer_output(
    request: OptimizerAllocationRequest,
    *,
    candidate_frame: pd.DataFrame,
    solve_result: Any,
    exit_date: pd.Timestamp,
) -> tuple[list[ActiveTrade], list[dict[str, object]]]:
    trades: list[ActiveTrade] = []
    allocation_rows: list[dict[str, object]] = []
    executed_month_turnover = max(request.month_to_date_turnover_eur, 0.0)
    ticker_series = candidate_frame.loc[:, "ticker"]
    tickers = ticker_series.astype(str).tolist()
    for idx, ticker in enumerate(tickers):
        candidate_row = candidate_frame.iloc[idx]
        candidate_values = {str(key): value for key, value in candidate_row.to_dict().items()}
        spec = cast(XtbInstrumentSpec, candidate_values["spec"])
        target_shares = int(solve_result.target_shares_new[idx])
        target_notional = float(solve_result.target_notional_new_eur[idx])
        if target_shares <= 0 or target_notional < spec.minimum_order_value_eur:
            continue
        entry_cost_estimate = estimate_trade_cost(
            spec,
            order_value_eur=target_notional,
            month_to_date_turnover_eur=executed_month_turnover,
            account_currency=request.account_currency,
        )
        required_cash = target_notional + entry_cost_estimate.total_cost_amount_eur
        if required_cash > request.cash_balance:
            continue
        trades.append(
            ActiveTrade(
                ticker=ticker,
                side="long",
                entry_date=request.trade_date,
                exit_date=exit_date,
                notional=target_notional,
                predicted_return=_scalar_float(candidate_values["prediction"]),
                realized_log_return=_scalar_float(candidate_values["realized_log_return"]),
                signal_rank=idx + 1,
                spec=spec,
                share_count=target_shares,
                reference_price_eur=_scalar_float(candidate_values["price_eur"]),
                entry_transaction_cost_amount=entry_cost_estimate.total_cost_amount_eur,
                entry_commission_amount=entry_cost_estimate.commission_amount_eur,
                entry_fx_conversion_amount=entry_cost_estimate.fx_conversion_amount_eur,
                expected_entry_cost_rate=entry_cost_estimate.total_cost_rate,
                capacity_bound=target_shares >= _scalar_int(candidate_values["adv_cap_shares"]),
            ),
        )
        allocation_rows.append(
            _build_allocation_row(
                trade_date=request.trade_date,
                ticker=ticker,
                candidate_row=candidate_row,
                target_weight=float(solve_result.target_weight_new[idx]),
                target_notional=target_notional,
                target_shares=target_shares,
            ),
        )
        executed_month_turnover = entry_cost_estimate.month_to_date_turnover_eur_after_order
    return trades, allocation_rows


def _build_optimizer_diagnostics(
    *,
    solve_result: Any,
    candidate_count: int,
    tradable_count: int,
) -> dict[str, DiagnosticValue]:
    return {
        "solver_status": solve_result.solver_status,
        "candidate_count": float(candidate_count),
        "tradable_count": float(tradable_count),
        "expected_portfolio_return": solve_result.expected_portfolio_return,
        "expected_portfolio_volatility": solve_result.expected_portfolio_volatility,
        "expected_net_alpha": solve_result.expected_net_alpha,
        "constraint_binding_sector_share": solve_result.constraint_binding_sector_share,
        "constraint_binding_liquidity_share": solve_result.constraint_binding_liquidity_share,
        "cash_weight": solve_result.cash_weight,
        "cash_amount_eur": solve_result.cash_amount_eur,
        "solve_time_seconds": solve_result.solve_time_seconds,
        "mip_gap": solve_result.mip_gap,
        "line_count_new": float(solve_result.line_count_new),
        "integer_shares_bought_total": float(solve_result.target_shares_new.sum()),
    }


def allocate_signal_candidates_optimizer_miqp(
    request: OptimizerAllocationRequest,
) -> tuple[list[ActiveTrade], dict[str, DiagnosticValue], list[dict[str, object]]]:
    if request.current_equity <= 0.0 or request.cash_balance <= 0.0:
        return [], {"solver_status": "no_capital"}, []
    exit_date = _resolve_exit_date(
        request.trade_date,
        request.unique_dates,
        request.hold_period_days,
    )
    if exit_date is None or request.daily_predictions.empty:
        diagnostics = _build_empty_optimizer_diagnostics(
            solver_status="no_exit_or_empty",
            cash_balance=request.cash_balance,
            current_equity=request.current_equity,
        )
        return [], diagnostics, []
    no_trade_buffer_rate = (
        request.no_trade_buffer_bps / 10_000.0 + request.open_hurdle_bps / 10_000.0
    )
    candidate_frame, candidate_count = _build_candidate_frame(
        request,
        no_trade_buffer_rate=no_trade_buffer_rate,
    )
    if candidate_count == 0:
        diagnostics = _build_empty_optimizer_diagnostics(
            solver_status="no_feasible_candidates",
            cash_balance=request.cash_balance,
            current_equity=request.current_equity,
        )
        return [], diagnostics, []
    solve_result = solve_portfolio_miqp(
        _build_solver_input(
            request,
            candidate_frame=candidate_frame,
            no_trade_buffer_rate=no_trade_buffer_rate,
        ),
    )
    if solve_result.solver_status not in {"optimal", "timelimit", "gaplimit"}:
        return [], {"solver_status": solve_result.solver_status}, []
    trades, allocation_rows = _materialize_optimizer_output(
        request,
        candidate_frame=candidate_frame,
        solve_result=solve_result,
        exit_date=exit_date,
    )
    diagnostics = _build_optimizer_diagnostics(
        solve_result=solve_result,
        candidate_count=candidate_count,
        tradable_count=len(candidate_frame),
    )
    return trades, diagnostics, allocation_rows


def finalize_trade(
    trade: ActiveTrade,
    *,
    exit_cost_estimate: BrokerCostEstimate,
) -> ClosedTrade:
    arithmetic_return = float(np.exp(trade.realized_log_return) - 1.0)
    gross_pnl_amount = trade.notional * arithmetic_return
    exit_transaction_cost_amount = exit_cost_estimate.total_cost_amount_eur
    total_transaction_cost_amount = trade.entry_transaction_cost_amount + exit_transaction_cost_amount
    transaction_cost = total_transaction_cost_amount / max(trade.notional, 1e-12)
    net_return = arithmetic_return - transaction_cost
    net_pnl_amount = gross_pnl_amount - total_transaction_cost_amount
    exit_cash_flow_amount = trade.notional + gross_pnl_amount - exit_transaction_cost_amount
    return ClosedTrade(
        ticker=trade.ticker,
        side=trade.side,
        entry_date=trade.entry_date,
        exit_date=trade.exit_date,
        notional=trade.notional,
        predicted_return=trade.predicted_return,
        realized_log_return=trade.realized_log_return,
        signal_rank=trade.signal_rank,
        gross_return=arithmetic_return,
        transaction_cost=transaction_cost,
        net_return=net_return,
        pnl_amount=net_pnl_amount,
        exit_cash_flow_amount=exit_cash_flow_amount,
        share_count=trade.share_count,
        reference_price_eur=trade.reference_price_eur,
        gross_pnl_amount=gross_pnl_amount,
        entry_transaction_cost_amount=trade.entry_transaction_cost_amount,
        exit_transaction_cost_amount=exit_transaction_cost_amount,
        total_transaction_cost_amount=total_transaction_cost_amount,
        entry_commission_amount=trade.entry_commission_amount,
        exit_commission_amount=exit_cost_estimate.commission_amount_eur,
        total_commission_amount=(
            trade.entry_commission_amount + exit_cost_estimate.commission_amount_eur
        ),
        entry_fx_conversion_amount=trade.entry_fx_conversion_amount,
        exit_fx_conversion_amount=exit_cost_estimate.fx_conversion_amount_eur,
        total_fx_conversion_amount=(
            trade.entry_fx_conversion_amount + exit_cost_estimate.fx_conversion_amount_eur
        ),
        net_pnl_amount=net_pnl_amount,
    )


def _apply_month_end_custody_fee(
    state: BacktestState,
    *,
    trade_date: pd.Timestamp,
    unique_dates: pd.Index,
    active_notional_end: float,
) -> float:
    month_key = _month_key(trade_date)
    state.monthly_market_values_eur.setdefault(month_key, []).append(active_notional_end)
    matching_positions = np.flatnonzero(unique_dates == trade_date)
    if matching_positions.size == 0:
        return 0.0
    next_position = int(matching_positions[0]) + 1
    next_month_key = None
    if next_position < len(unique_dates):
        next_timestamp = _coerce_timestamp(unique_dates.tolist()[next_position])
        if next_timestamp is not None:
            next_month_key = _month_key(next_timestamp)
    if next_month_key == month_key:
        return 0.0
    if not state.active_trades:
        return 0.0
    average_monthly_value = float(np.mean(state.monthly_market_values_eur[month_key]))
    reference_spec = state.active_trades[0].spec
    if average_monthly_value <= reference_spec.custody_fee_threshold_eur:
        return 0.0
    monthly_fee_amount = max(
        reference_spec.monthly_custody_fee_min_eur,
        average_monthly_value * reference_spec.annual_custody_fee_rate / 12.0,
    )
    state.cash_balance -= monthly_fee_amount
    return monthly_fee_amount


def _close_exiting_trades(
    *,
    state: BacktestState,
    trade_date: pd.Timestamp,
    cost_config: XtbCostConfig,
) -> tuple[list[ActiveTrade], dict[str, float]]:
    active_trades: list[ActiveTrade] = list(state.active_trades)
    exiting_today: list[ActiveTrade] = [
        trade for trade in active_trades if trade.exit_date == trade_date
    ]
    surviving_trades: list[ActiveTrade] = [
        trade for trade in active_trades if trade.exit_date != trade_date
    ]
    realized_pnl_today = 0.0
    gross_pnl_exits = 0.0
    exit_cost_amount = 0.0
    closed_notional = 0.0
    closed_trade_count = 0.0
    for trade in exiting_today:
        exit_cost_estimate = estimate_trade_cost(
            trade.spec,
            order_value_eur=trade.notional,
            month_to_date_turnover_eur=_resolve_month_to_date_turnover(state, trade_date),
            account_currency=cost_config.account_currency,
        )
        _register_order_turnover(state, trade_date, trade.notional)
        closed_trade = finalize_trade(trade, exit_cost_estimate=exit_cost_estimate)
        state.closed_trades.append(closed_trade)
        gross_pnl_exits += closed_trade.gross_pnl_amount
        exit_cost_amount += closed_trade.exit_transaction_cost_amount
        closed_notional += abs(closed_trade.notional)
        closed_trade_count += 1.0
        realized_pnl_today += closed_trade.gross_pnl_amount - closed_trade.exit_transaction_cost_amount
        state.cash_balance += closed_trade.exit_cash_flow_amount
    state.active_trades = surviving_trades
    return exiting_today, {
        "realized_pnl_today": realized_pnl_today,
        "gross_pnl_exits": gross_pnl_exits,
        "exit_cost_amount": exit_cost_amount,
        "closed_notional": closed_notional,
        "closed_trade_count": closed_trade_count,
    }


def _build_optimizer_request(
    *,
    trade_date: pd.Timestamp,
    daily_predictions: pd.DataFrame,
    state: BacktestState,
    runtime: BacktestRuntimeConfig,
    opening_equity: float,
) -> OptimizerAllocationRequest:
    if runtime.optimizer_artifacts is None:
        raise ValueError("optimizer_miqp mode requires optimizer_artifacts.")
    return OptimizerAllocationRequest(
        trade_date=trade_date,
        daily_predictions=daily_predictions,
        active_trades=list(state.active_trades),
        current_equity=max(opening_equity, 0.0),
        cash_balance=state.cash_balance,
        unique_dates=runtime.unique_dates,
        month_to_date_turnover_eur=_resolve_month_to_date_turnover(state, trade_date),
        hold_period_days=runtime.hold_period_days,
        gross_cap_fraction=runtime.gross_cap_fraction,
        adv_participation_limit=runtime.adv_participation_limit,
        open_hurdle_bps=runtime.open_hurdle_bps,
        account_currency=runtime.cost_config.account_currency,
        cost_config=runtime.cost_config,
        artifacts=runtime.optimizer_artifacts,
        lambda_risk=runtime.lambda_risk,
        lambda_turnover=runtime.lambda_turnover,
        lambda_cost=runtime.lambda_cost,
        max_position_weight=runtime.max_position_weight,
        max_sector_weight=runtime.max_sector_weight,
        min_target_weight=runtime.min_target_weight,
        no_trade_buffer_bps=runtime.no_trade_buffer_bps,
        miqp_time_limit_seconds=runtime.miqp_time_limit_seconds,
        miqp_relative_gap=runtime.miqp_relative_gap,
        miqp_candidate_pool_size=runtime.miqp_candidate_pool_size,
        miqp_primary_objective_tolerance_bps=runtime.miqp_primary_objective_tolerance_bps,
    )


def _open_new_trades(
    *,
    state: BacktestState,
    trade_date: pd.Timestamp,
    new_trades: list[ActiveTrade],
) -> tuple[float, float]:
    opened_notional = float(sum(abs(trade.notional) for trade in new_trades))
    entry_cost_amount = 0.0
    for trade in new_trades:
        entry_cost_amount += trade.entry_transaction_cost_amount
        _register_order_turnover(state, trade_date, trade.notional)
        state.cash_balance -= trade.notional + trade.entry_transaction_cost_amount
        state.active_trades.append(trade)
    return opened_notional, entry_cost_amount


def _build_daily_diagnostics_row(
    *,
    state: BacktestState,
    snapshot: DailyAccountingSnapshot,
    new_trades: list[ActiveTrade],
    benchmark_return: float,
    optimizer_diagnostics: Mapping[str, DiagnosticValue],
) -> dict[str, object]:
    turnover_notional = float(snapshot.closed_notional + snapshot.opened_notional)
    capacity_binding_share = 0.0
    if new_trades:
        capacity_binding_share = float(
            np.mean([1.0 if trade.capacity_bound else 0.0 for trade in new_trades]),
        )
    gross_exposure = 0.0
    if state.current_equity > 0.0:
        gross_exposure = snapshot.active_notional_end / state.current_equity
    realized_return = snapshot.realized_pnl_today / max(snapshot.opening_equity, 1e-12)
    reconciliation_error = state.current_equity - (state.cash_balance + snapshot.active_notional_end)
    if not math.isclose(reconciliation_error, 0.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Backtest cashflow reconciliation failed on {snapshot.trade_date.date()}: error={reconciliation_error:.10f}",
        )
    return {
        "date": snapshot.trade_date,
        "starting_equity": snapshot.opening_equity,
        "equity": state.current_equity,
        "ending_equity": state.current_equity,
        "cash_balance": state.cash_balance,
        "total_return": state.current_equity / state.initial_equity - 1.0,
        "realized_pnl": snapshot.realized_pnl_today,
        "realized_return": realized_return,
        "benchmark_return": benchmark_return,
        "gross_pnl_exits": snapshot.gross_pnl_exits,
        "entry_cost_amount": snapshot.entry_cost_amount,
        "exit_cost_amount": snapshot.exit_cost_amount,
        "custody_fee_amount": snapshot.custody_fee_amount,
        "net_cash_flow": state.cash_balance - snapshot.opening_cash,
        "turnover": turnover_notional / max(snapshot.opening_equity, 1e-12),
        "month_to_date_turnover_eur": _resolve_month_to_date_turnover(state, snapshot.trade_date),
        "opened_notional": snapshot.opened_notional,
        "closed_notional": snapshot.closed_notional,
        "active_notional_end": snapshot.active_notional_end,
        "gross_exposure": gross_exposure,
        "capacity_binding_share": capacity_binding_share,
        "active_trade_count": len(state.active_trades),
        "opened_trade_count": len(new_trades),
        "closed_trade_count": snapshot.closed_trade_count,
        "reconciliation_error": reconciliation_error,
        "solver_status": str(optimizer_diagnostics.get("solver_status", "not_used")),
        "candidate_count": float(optimizer_diagnostics.get("candidate_count", 0.0)),
        "tradable_count": float(optimizer_diagnostics.get("tradable_count", 0.0)),
        "expected_portfolio_return": float(optimizer_diagnostics.get("expected_portfolio_return", 0.0)),
        "expected_portfolio_volatility": float(optimizer_diagnostics.get("expected_portfolio_volatility", 0.0)),
        "expected_net_alpha": float(optimizer_diagnostics.get("expected_net_alpha", 0.0)),
        "constraint_binding_sector_share": float(
            optimizer_diagnostics.get("constraint_binding_sector_share", 0.0),
        ),
        "constraint_binding_liquidity_share": float(
            optimizer_diagnostics.get("constraint_binding_liquidity_share", 0.0),
        ),
        "cash_weight": float(optimizer_diagnostics.get("cash_weight", 1.0)),
        "cash_amount_eur": float(optimizer_diagnostics.get("cash_amount_eur", state.cash_balance)),
        "solve_time_seconds": float(optimizer_diagnostics.get("solve_time_seconds", 0.0)),
        "mip_gap": float(optimizer_diagnostics.get("mip_gap", float("nan"))),
        "line_count_new": float(optimizer_diagnostics.get("line_count_new", 0.0)),
        "integer_shares_bought_total": float(
            optimizer_diagnostics.get("integer_shares_bought_total", 0.0),
        ),
    }


def _build_optimizer_daily_row(
    *,
    trade_date: pd.Timestamp,
    state: BacktestState,
    optimizer_diagnostics: Mapping[str, DiagnosticValue],
) -> dict[str, object]:
    return {
        "date": trade_date,
        "solver_status": str(optimizer_diagnostics.get("solver_status", "not_used")),
        "candidate_count": float(optimizer_diagnostics.get("candidate_count", 0.0)),
        "tradable_count": float(optimizer_diagnostics.get("tradable_count", 0.0)),
        "warmup_mode": False,
        "expected_portfolio_return": float(optimizer_diagnostics.get("expected_portfolio_return", 0.0)),
        "expected_portfolio_volatility": float(
            optimizer_diagnostics.get("expected_portfolio_volatility", 0.0),
        ),
        "expected_net_alpha": float(optimizer_diagnostics.get("expected_net_alpha", 0.0)),
        "constraint_binding_sector_share": float(
            optimizer_diagnostics.get("constraint_binding_sector_share", 0.0),
        ),
        "constraint_binding_liquidity_share": float(
            optimizer_diagnostics.get("constraint_binding_liquidity_share", 0.0),
        ),
        "cash_weight": float(optimizer_diagnostics.get("cash_weight", 1.0)),
        "cash_amount_eur": float(optimizer_diagnostics.get("cash_amount_eur", state.cash_balance)),
        "solve_time_seconds": float(optimizer_diagnostics.get("solve_time_seconds", 0.0)),
        "mip_gap": float(optimizer_diagnostics.get("mip_gap", float("nan"))),
        "line_count_new": float(optimizer_diagnostics.get("line_count_new", 0.0)),
        "integer_shares_bought_total": float(
            optimizer_diagnostics.get("integer_shares_bought_total", 0.0),
        ),
    }


def _override_runtime_unique_dates(
    runtime: BacktestRuntimeConfig,
    *,
    unique_dates: pd.Index,
) -> BacktestRuntimeConfig:
    return BacktestRuntimeConfig(
        unique_dates=unique_dates,
        top_fraction=runtime.top_fraction,
        allocation_fraction=runtime.allocation_fraction,
        action_cap_fraction=runtime.action_cap_fraction,
        gross_cap_fraction=runtime.gross_cap_fraction,
        adv_participation_limit=runtime.adv_participation_limit,
        neutrality_mode=runtime.neutrality_mode,
        open_hurdle_bps=runtime.open_hurdle_bps,
        apply_prediction_hurdle=runtime.apply_prediction_hurdle,
        hold_period_days=runtime.hold_period_days,
        cost_config=runtime.cost_config,
        portfolio_construction_mode=runtime.portfolio_construction_mode,
        optimizer_artifacts=runtime.optimizer_artifacts,
        lambda_risk=runtime.lambda_risk,
        lambda_turnover=runtime.lambda_turnover,
        lambda_cost=runtime.lambda_cost,
        max_position_weight=runtime.max_position_weight,
        max_sector_weight=runtime.max_sector_weight,
        min_target_weight=runtime.min_target_weight,
        no_trade_buffer_bps=runtime.no_trade_buffer_bps,
        miqp_time_limit_seconds=runtime.miqp_time_limit_seconds,
        miqp_relative_gap=runtime.miqp_relative_gap,
        miqp_candidate_pool_size=runtime.miqp_candidate_pool_size,
        miqp_primary_objective_tolerance_bps=runtime.miqp_primary_objective_tolerance_bps,
        benchmark_returns_by_date=dict(runtime.benchmark_returns_by_date),
    )


def process_prediction_day(
    state: BacktestState,
    daily_predictions: pd.DataFrame,
    runtime: BacktestRuntimeConfig,
    *,
    logger: Any | None = None,
) -> None:
    if daily_predictions.empty:
        return
    trade_date_value = pd.to_datetime(daily_predictions[DATE_COLUMN]).iloc[0]
    trade_date = _coerce_timestamp(trade_date_value)
    if trade_date is None:
        raise ValueError("Could not resolve trade_date from daily_predictions.")
    opening_equity = float(state.current_equity)
    opening_cash = float(state.cash_balance)
    _, close_metrics = _close_exiting_trades(
        state=state,
        trade_date=trade_date,
        cost_config=runtime.cost_config,
    )
    realized_pnl_today = close_metrics["realized_pnl_today"]
    gross_pnl_exits = close_metrics["gross_pnl_exits"]
    exit_cost_amount = close_metrics["exit_cost_amount"]
    closed_notional = close_metrics["closed_notional"]
    closed_trade_count = close_metrics["closed_trade_count"]
    if runtime.portfolio_construction_mode != "optimizer_miqp":
        raise ValueError("portfolio_construction_mode must be optimizer_miqp.")
    new_trades, optimizer_diagnostics, allocation_rows = allocate_signal_candidates_optimizer_miqp(
        _build_optimizer_request(
            trade_date=trade_date,
            daily_predictions=daily_predictions,
            state=state,
            runtime=runtime,
            opening_equity=opening_equity,
        ),
    )
    state.allocation_rows.extend(allocation_rows)
    opened_notional, entry_cost_amount = _open_new_trades(
        state=state,
        trade_date=trade_date,
        new_trades=new_trades,
    )
    realized_pnl_today -= entry_cost_amount
    active_notional_end = float(sum(trade.notional for trade in state.active_trades))
    custody_fee_amount = _apply_month_end_custody_fee(
        state,
        trade_date=trade_date,
        unique_dates=runtime.unique_dates,
        active_notional_end=active_notional_end,
    )
    realized_pnl_today -= custody_fee_amount
    state.current_equity = state.cash_balance + active_notional_end
    snapshot = DailyAccountingSnapshot(
        trade_date=trade_date,
        opening_equity=opening_equity,
        opening_cash=opening_cash,
        realized_pnl_today=realized_pnl_today,
        gross_pnl_exits=gross_pnl_exits,
        entry_cost_amount=entry_cost_amount,
        exit_cost_amount=exit_cost_amount,
        custody_fee_amount=custody_fee_amount,
        opened_notional=opened_notional,
        closed_notional=closed_notional,
        closed_trade_count=closed_trade_count,
        active_notional_end=active_notional_end,
    )
    benchmark_return = float(runtime.benchmark_returns_by_date.get(trade_date, 0.0))
    state.daily_rows.append(
        _build_daily_diagnostics_row(
            state=state,
            snapshot=snapshot,
            new_trades=new_trades,
            benchmark_return=benchmark_return,
            optimizer_diagnostics=optimizer_diagnostics,
        ),
    )
    state.optimizer_daily_rows.append(
        _build_optimizer_daily_row(
            trade_date=trade_date,
            state=state,
            optimizer_diagnostics=optimizer_diagnostics,
        ),
    )
    if logger is not None:
        logger.info(
            "Portfolio day %s | total_return=%.4f%%",
            trade_date.date(),
            100.0 * (state.current_equity / state.initial_equity - 1.0),
        )


def finalize_backtest_state(
    state: BacktestState,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    trades_frame = pd.DataFrame([trade.__dict__ for trade in state.closed_trades])
    daily_frame = pd.DataFrame(state.daily_rows)
    if not daily_frame.empty and "spec" in trades_frame.columns:
        trades_frame = trades_frame.drop(columns=["spec"])
    if daily_frame.empty:
        summary = {
            "final_equity": state.initial_equity,
            "net_pnl": 0.0,
            "gross_pnl_before_costs": 0.0,
            "entry_cost_amount_total": 0.0,
            "exit_cost_amount_total": 0.0,
            "custody_fee_amount_total": 0.0,
            "transaction_cost_amount_total": 0.0,
            "net_pnl_after_costs": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_benchmark_return": 0.0,
            "alpha_over_benchmark_net": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "turnover_annualized": 0.0,
            "average_gross_exposure": 0.0,
            "capacity_binding_share": 0.0,
            "trade_count": 0.0,
            "win_rate": 0.0,
            "avg_entry_cost_rate": 0.0,
            "avg_exit_cost_rate": 0.0,
            "avg_roundtrip_cost_rate": 0.0,
            "cost_drag_total": 0.0,
            "cost_drag_annualized": 0.0,
            "reconciliation_error_max_abs": 0.0,
        }
        return trades_frame, daily_frame, summary

    equity_series = pd.Series(daily_frame["equity"], dtype=float)
    daily_returns = pd.Series(daily_frame["realized_return"], dtype=float)
    benchmark_returns = pd.Series(daily_frame["benchmark_return"], dtype=float)
    gross_pnl_before_costs = float(daily_frame["gross_pnl_exits"].sum())
    entry_cost_amount_total = float(daily_frame["entry_cost_amount"].sum())
    exit_cost_amount_total = float(daily_frame["exit_cost_amount"].sum())
    custody_fee_amount_total = float(daily_frame["custody_fee_amount"].sum())
    transaction_cost_amount_total = (
        entry_cost_amount_total + exit_cost_amount_total + custody_fee_amount_total
    )
    net_pnl_after_costs = float(gross_pnl_before_costs - transaction_cost_amount_total)
    running_peak = equity_series.cummax()
    drawdown = equity_series / running_peak - 1.0
    equity_growth = float(equity_series.iloc[-1] / max(state.initial_equity, 1e-12))
    annualized_return = float(equity_growth ** (252.0 / max(len(daily_frame), 1)) - 1.0)
    benchmark_equity = float(np.prod(1.0 + benchmark_returns.to_numpy(dtype=np.float64)))
    annualized_benchmark_return = float(benchmark_equity ** (252.0 / max(len(daily_frame), 1)) - 1.0)
    annualized_volatility = float(daily_returns.std(ddof=0) * np.sqrt(252.0))
    sharpe_ratio = 0.0
    if annualized_volatility > 0.0:
        sharpe_ratio = annualized_return / annualized_volatility
    max_drawdown = float(drawdown.min())
    calmar_ratio = 0.0
    if max_drawdown < 0.0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    win_rate = 0.0
    avg_entry_cost_rate = 0.0
    avg_exit_cost_rate = 0.0
    avg_roundtrip_cost_rate = 0.0
    if not trades_frame.empty:
        win_rate = float((trades_frame["net_return"] > 0.0).mean())
        avg_entry_cost_rate = float(
            trades_frame["entry_transaction_cost_amount"].sum() / trades_frame["notional"].sum(),
        )
        avg_exit_cost_rate = float(
            trades_frame["exit_transaction_cost_amount"].sum() / trades_frame["notional"].sum(),
        )
        avg_roundtrip_cost_rate = float(
            trades_frame["total_transaction_cost_amount"].sum() / trades_frame["notional"].sum(),
        )
    cost_drag_total = float(transaction_cost_amount_total)
    cost_drag_annualized = float(cost_drag_total / max(len(daily_frame), 1) * 252.0)
    reconciliation_error_max_abs = float(np.abs(daily_frame["reconciliation_error"]).max())
    summary = {
        "final_equity": float(equity_series.iloc[-1]),
        "net_pnl": float(equity_series.iloc[-1] - state.initial_equity),
        "gross_pnl_before_costs": gross_pnl_before_costs,
        "entry_cost_amount_total": entry_cost_amount_total,
        "exit_cost_amount_total": exit_cost_amount_total,
        "custody_fee_amount_total": custody_fee_amount_total,
        "transaction_cost_amount_total": transaction_cost_amount_total,
        "net_pnl_after_costs": net_pnl_after_costs,
        "total_return": float(equity_series.iloc[-1] / state.initial_equity - 1.0),
        "annualized_return": annualized_return,
        "annualized_benchmark_return": annualized_benchmark_return,
        "alpha_over_benchmark_net": annualized_return - annualized_benchmark_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "calmar_ratio": float(calmar_ratio),
        "max_drawdown": max_drawdown,
        "turnover_annualized": float(daily_frame["turnover"].mean() * 252.0),
        "average_gross_exposure": float(daily_frame["gross_exposure"].mean()),
        "capacity_binding_share": float(daily_frame["capacity_binding_share"].mean()),
        "trade_count": float(len(trades_frame)),
        "win_rate": float(win_rate),
        "avg_entry_cost_rate": avg_entry_cost_rate,
        "avg_exit_cost_rate": avg_exit_cost_rate,
        "avg_roundtrip_cost_rate": avg_roundtrip_cost_rate,
        "cost_drag_total": cost_drag_total,
        "cost_drag_annualized": cost_drag_annualized,
        "reconciliation_error_max_abs": reconciliation_error_max_abs,
    }
    return trades_frame, daily_frame, summary


def run_signal_backtest(
    predictions: pd.DataFrame,
    run_config: BacktestRunConfig,
    *,
    logger: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    ordered_predictions = predictions.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    unique_dates = pd.Index(pd.to_datetime(ordered_predictions[DATE_COLUMN]).drop_duplicates().sort_values())
    benchmark_returns = _build_buy_and_hold_benchmark_returns(ordered_predictions)
    runtime = _override_runtime_unique_dates(
        run_config.runtime,
        unique_dates=unique_dates,
    )
    runtime = BacktestRuntimeConfig(
        unique_dates=runtime.unique_dates,
        top_fraction=runtime.top_fraction,
        allocation_fraction=runtime.allocation_fraction,
        action_cap_fraction=runtime.action_cap_fraction,
        gross_cap_fraction=runtime.gross_cap_fraction,
        adv_participation_limit=runtime.adv_participation_limit,
        neutrality_mode=runtime.neutrality_mode,
        open_hurdle_bps=runtime.open_hurdle_bps,
        apply_prediction_hurdle=runtime.apply_prediction_hurdle,
        hold_period_days=runtime.hold_period_days,
        cost_config=runtime.cost_config,
        portfolio_construction_mode=runtime.portfolio_construction_mode,
        optimizer_artifacts=runtime.optimizer_artifacts,
        lambda_risk=runtime.lambda_risk,
        lambda_turnover=runtime.lambda_turnover,
        lambda_cost=runtime.lambda_cost,
        max_position_weight=runtime.max_position_weight,
        max_sector_weight=runtime.max_sector_weight,
        min_target_weight=runtime.min_target_weight,
        no_trade_buffer_bps=runtime.no_trade_buffer_bps,
        miqp_time_limit_seconds=runtime.miqp_time_limit_seconds,
        miqp_relative_gap=runtime.miqp_relative_gap,
        miqp_candidate_pool_size=runtime.miqp_candidate_pool_size,
        miqp_primary_objective_tolerance_bps=runtime.miqp_primary_objective_tolerance_bps,
        benchmark_returns_by_date={
            cast(pd.Timestamp, pd.Timestamp(date)): float(value)
            for date, value in benchmark_returns.items()
        },
    )
    state = BacktestState(
        initial_equity=run_config.starting_cash_eur,
        current_equity=run_config.starting_cash_eur,
        cash_balance=run_config.starting_cash_eur,
    )
    for trade_date in unique_dates:
        process_prediction_day(
            state,
            pd.DataFrame(
                ordered_predictions.loc[ordered_predictions[DATE_COLUMN] == trade_date].copy(),
            ),
            runtime,
            logger=logger,
        )
    return finalize_backtest_state(state)


def run_signal_backtest_with_diagnostics(
    predictions: pd.DataFrame,
    run_config: BacktestRunConfig,
    *,
    progress: BacktestProgressConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame, pd.DataFrame]:
    ordered_predictions = predictions.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    unique_dates = pd.Index(pd.to_datetime(ordered_predictions[DATE_COLUMN]).drop_duplicates().sort_values())
    benchmark_returns = _build_buy_and_hold_benchmark_returns(ordered_predictions)
    runtime = _override_runtime_unique_dates(
        run_config.runtime,
        unique_dates=unique_dates,
    )
    runtime = BacktestRuntimeConfig(
        unique_dates=runtime.unique_dates,
        top_fraction=runtime.top_fraction,
        allocation_fraction=runtime.allocation_fraction,
        action_cap_fraction=runtime.action_cap_fraction,
        gross_cap_fraction=runtime.gross_cap_fraction,
        adv_participation_limit=runtime.adv_participation_limit,
        neutrality_mode=runtime.neutrality_mode,
        open_hurdle_bps=runtime.open_hurdle_bps,
        apply_prediction_hurdle=runtime.apply_prediction_hurdle,
        hold_period_days=runtime.hold_period_days,
        cost_config=runtime.cost_config,
        portfolio_construction_mode=runtime.portfolio_construction_mode,
        optimizer_artifacts=runtime.optimizer_artifacts,
        lambda_risk=runtime.lambda_risk,
        lambda_turnover=runtime.lambda_turnover,
        lambda_cost=runtime.lambda_cost,
        max_position_weight=runtime.max_position_weight,
        max_sector_weight=runtime.max_sector_weight,
        min_target_weight=runtime.min_target_weight,
        no_trade_buffer_bps=runtime.no_trade_buffer_bps,
        miqp_time_limit_seconds=runtime.miqp_time_limit_seconds,
        miqp_relative_gap=runtime.miqp_relative_gap,
        miqp_candidate_pool_size=runtime.miqp_candidate_pool_size,
        miqp_primary_objective_tolerance_bps=runtime.miqp_primary_objective_tolerance_bps,
        benchmark_returns_by_date={
            cast(pd.Timestamp, pd.Timestamp(date)): float(value)
            for date, value in benchmark_returns.items()
        },
    )
    state = BacktestState(
        initial_equity=run_config.starting_cash_eur,
        current_equity=run_config.starting_cash_eur,
        cash_balance=run_config.starting_cash_eur,
    )
    progress_config = progress or BacktestProgressConfig()
    started_at: float | None = None
    if progress_config.logger is not None and progress_config.progress_label:
        started_at = time.perf_counter()
        progress_config.logger.info(
            "%s started: dates=%d",
            progress_config.progress_label,
            len(unique_dates),
        )
    for date_index, trade_date in enumerate(unique_dates, start=1):
        process_prediction_day(
            state,
            pd.DataFrame(
                ordered_predictions.loc[ordered_predictions[DATE_COLUMN] == trade_date].copy(),
            ),
            runtime,
            logger=None,
        )
        if (
            progress_config.logger is not None
            and progress_config.progress_label
            and progress_config.progress_log_every > 0
            and started_at is not None
            and (
                date_index == 1
                or date_index == len(unique_dates)
                or date_index % progress_config.progress_log_every == 0
            )
        ):
            elapsed_seconds = time.perf_counter() - started_at
            average_seconds = elapsed_seconds / max(1, date_index)
            eta_seconds = average_seconds * max(0, len(unique_dates) - date_index)
            progress_config.logger.info(
                "%s progress: %d/%d dates | trade_date=%s | elapsed=%s | eta=%s",
                progress_config.progress_label,
                date_index,
                len(unique_dates),
                pd.Timestamp(trade_date).date(),
                _format_duration(elapsed_seconds),
                _format_duration(eta_seconds),
            )
    trades_frame, daily_frame, summary = finalize_backtest_state(state)
    allocations = pd.DataFrame(state.allocation_rows)
    optimizer_daily = pd.DataFrame(state.optimizer_daily_rows)
    return trades_frame, daily_frame, summary, allocations, optimizer_daily
