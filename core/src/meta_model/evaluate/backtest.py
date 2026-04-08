from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from core.src.meta_model.broker_xtb.costs import BrokerCostEstimate, estimate_trade_cost
from core.src.meta_model.broker_xtb.specs import (
    BrokerSpecProvider,
    XtbInstrumentSpec,
    build_default_spec_provider,
)
from core.src.meta_model.model_contract import (
    DATE_COLUMN,
    PREDICTION_COLUMN,
    REALIZED_RETURN_COLUMN,
    TICKER_COLUMN,
)


@dataclass(frozen=True)
class SignalCandidate:
    ticker: str
    side: str
    prediction: float
    signal_rank: int
    realized_log_return: float
    spec: XtbInstrumentSpec
    sector: str = "__unknown_sector__"
    adv_cap_notional: float = float("inf")


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


@dataclass
class BacktestState:
    initial_equity: float = 100_000.0
    current_equity: float = 100_000.0
    cash_balance: float = 100_000.0
    active_trades: list[ActiveTrade] = field(default_factory=list)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    daily_rows: list[dict[str, object]] = field(default_factory=list)
    monthly_turnover_eur: dict[str, float] = field(default_factory=dict)
    monthly_market_values_eur: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class AllocationContext:
    trade_date: pd.Timestamp
    exit_date: pd.Timestamp
    current_equity: float
    cash_balance: float
    allocation_fraction: float
    action_cap_fraction: float
    adv_participation_limit: float
    open_hurdle_bps: float
    apply_prediction_hurdle: bool
    account_currency: str
    month_to_date_turnover_eur: float


def _month_key(trade_date: pd.Timestamp) -> str:
    return f"{trade_date.year:04d}-{trade_date.month:02d}"


def _resolve_month_to_date_turnover(state: BacktestState, trade_date: pd.Timestamp) -> float:
    return float(state.monthly_turnover_eur.get(_month_key(trade_date), 0.0))


def _register_order_turnover(state: BacktestState, trade_date: pd.Timestamp, order_value_eur: float) -> None:
    month_key = _month_key(trade_date)
    state.monthly_turnover_eur[month_key] = _resolve_month_to_date_turnover(state, trade_date) + max(order_value_eur, 0.0)


def _validate_candidate_inputs(
    daily_predictions: pd.DataFrame,
    *,
    top_fraction: float,
    neutrality_mode: str,
) -> None:
    if daily_predictions.empty:
        return
    if top_fraction <= 0.0 or top_fraction > 1.0:
        raise ValueError("top_fraction must be in the interval (0, 1.0].")
    if neutrality_mode != "long_only":
        raise ValueError("XTB cash-equity backtest only supports long_only.")


def _select_signal_block(
    daily_predictions: pd.DataFrame,
    *,
    top_fraction: float,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    selection_count = max(1, int(np.ceil(len(daily_predictions) * top_fraction)))
    long_block = pd.DataFrame(
        daily_predictions.nlargest(selection_count, columns=PREDICTION_COLUMN, keep="first").reset_index(drop=True),
    )
    trade_date = pd.Timestamp(pd.to_datetime(daily_predictions[DATE_COLUMN]).iloc[0])
    return long_block, trade_date


def _resolve_adv_cap_notional(
    *,
    open_price: float,
    trading_volume: float,
    max_adv_participation: float,
) -> float:
    if open_price <= 0.0 or trading_volume <= 0.0:
        return float("inf")
    return max(0.0, open_price * trading_volume * max_adv_participation)


def _build_signal_candidate(
    row: object,
    *,
    rank: int,
    trade_date: pd.Timestamp,
    cost_config: XtbCostConfig,
) -> SignalCandidate:
    open_price = float(getattr(row, "stock_open_price", 0.0) or 0.0)
    trading_volume = float(getattr(row, "stock_trading_volume", 0.0) or 0.0)
    ticker = str(getattr(row, TICKER_COLUMN))
    spec = cost_config.broker_spec_provider.resolve(ticker, trade_date)
    adv_cap_notional = _resolve_adv_cap_notional(
        open_price=open_price,
        trading_volume=trading_volume,
        max_adv_participation=spec.max_adv_participation,
    )
    return SignalCandidate(
        ticker=ticker,
        side="long",
        prediction=float(getattr(row, PREDICTION_COLUMN)),
        signal_rank=rank,
        realized_log_return=float(getattr(row, REALIZED_RETURN_COLUMN, 0.0)),
        spec=spec,
        sector=str(getattr(row, "company_sector", "__unknown_sector__") or "__unknown_sector__"),
        adv_cap_notional=adv_cap_notional,
    )


def build_daily_signal_candidates(
    daily_predictions: pd.DataFrame,
    *,
    top_fraction: float,
    cost_config: XtbCostConfig,
    expected_holding_days: int,
    neutrality_mode: str = "long_only",
) -> list[SignalCandidate]:
    del expected_holding_days
    if daily_predictions.empty:
        return []
    _validate_candidate_inputs(
        daily_predictions,
        top_fraction=top_fraction,
        neutrality_mode=neutrality_mode,
    )
    long_block, trade_date = _select_signal_block(
        daily_predictions,
        top_fraction=top_fraction,
    )
    return [
        _build_signal_candidate(
            row,
            rank=rank,
            trade_date=trade_date,
            cost_config=cost_config,
        )
        for rank, row in enumerate(long_block.itertuples(index=False), start=1)
    ]


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
    return pd.Timestamp(unique_dates.tolist()[exit_position])


def _existing_symbol_notional(symbol: str, active_trades: list[ActiveTrade], pending_trades: list[ActiveTrade]) -> float:
    return float(
        sum(trade.notional for trade in active_trades if trade.ticker == symbol)
        + sum(trade.notional for trade in pending_trades if trade.ticker == symbol)
    )


def _sector_candidate_budget(
    candidates: list[SignalCandidate],
    *,
    target_notional: float,
) -> tuple[list[str], float]:
    sector_names = sorted({candidate.sector for candidate in candidates})
    sector_budget = target_notional / max(len(sector_names), 1)
    return sector_names, sector_budget


def _estimate_order_cost(
    candidate: SignalCandidate,
    *,
    order_value_eur: float,
    month_to_date_turnover_eur: float,
    account_currency: str,
) -> BrokerCostEstimate:
    return estimate_trade_cost(
        candidate.spec,
        order_value_eur=order_value_eur,
        month_to_date_turnover_eur=month_to_date_turnover_eur,
        account_currency=account_currency,
    )


def _passes_hurdle(
    candidate: SignalCandidate,
    *,
    proposed_notional: float,
    month_to_date_turnover_eur: float,
    open_hurdle_bps: float,
    apply_prediction_hurdle: bool,
    account_currency: str,
) -> bool:
    if not apply_prediction_hurdle:
        return True
    cost_estimate = _estimate_order_cost(
        candidate,
        order_value_eur=proposed_notional,
        month_to_date_turnover_eur=month_to_date_turnover_eur,
        account_currency=account_currency,
    )
    hurdle_rate = cost_estimate.total_cost_rate + open_hurdle_bps / 10_000.0
    return candidate.prediction > hurdle_rate


def _resolve_allocation_notional(
    candidate: SignalCandidate,
    *,
    candidate_budget: float,
    current_equity: float,
    current_symbol_notional: float,
    cash_balance: float,
    reserved_cash: float,
    allocation_fraction: float,
    action_cap_fraction: float,
    adv_participation_limit: float,
) -> float:
    symbol_room_notional = max(0.0, action_cap_fraction * current_equity - current_symbol_notional)
    adv_room_notional = candidate.adv_cap_notional * max(adv_participation_limit / 0.05, 0.0)
    cash_room_notional = max(0.0, cash_balance - reserved_cash)
    proposed_notional = min(
        candidate_budget,
        allocation_fraction * current_equity,
        symbol_room_notional,
        adv_room_notional,
        cash_room_notional,
    )
    if proposed_notional < candidate.spec.minimum_order_value_eur:
        return 0.0
    return proposed_notional


def _build_active_trade(
    candidate: SignalCandidate,
    *,
    context: AllocationContext,
    allocation_notional: float,
    candidate_budget: float,
    entry_cost_estimate: BrokerCostEstimate,
) -> ActiveTrade:
    return ActiveTrade(
        ticker=candidate.ticker,
        side="long",
        entry_date=context.trade_date,
        exit_date=context.exit_date,
        notional=float(allocation_notional),
        predicted_return=candidate.prediction,
        realized_log_return=candidate.realized_log_return,
        signal_rank=candidate.signal_rank,
        spec=candidate.spec,
        entry_transaction_cost_amount=entry_cost_estimate.total_cost_amount_eur,
        entry_commission_amount=entry_cost_estimate.commission_amount_eur,
        entry_fx_conversion_amount=entry_cost_estimate.fx_conversion_amount_eur,
        expected_entry_cost_rate=entry_cost_estimate.total_cost_rate,
        capacity_bound=allocation_notional < candidate_budget,
    )


def _allocate_sector_candidates(
    sector_candidates: list[SignalCandidate],
    *,
    candidate_budget: float,
    active_trades: list[ActiveTrade],
    trades: list[ActiveTrade],
    context: AllocationContext,
    month_to_date_turnover_eur: float,
) -> float:
    reserved_cash = 0.0
    current_turnover = month_to_date_turnover_eur
    for trade in trades:
        reserved_cash += trade.notional + trade.entry_transaction_cost_amount
        current_turnover += trade.notional
    for candidate in sector_candidates:
        current_symbol_notional = _existing_symbol_notional(candidate.ticker, active_trades, trades)
        allocation_notional = _resolve_allocation_notional(
            candidate,
            candidate_budget=candidate_budget,
            current_equity=context.current_equity,
            current_symbol_notional=current_symbol_notional,
            cash_balance=context.cash_balance,
            reserved_cash=reserved_cash,
            allocation_fraction=context.allocation_fraction,
            action_cap_fraction=context.action_cap_fraction,
            adv_participation_limit=context.adv_participation_limit,
        )
        if allocation_notional <= 0.0:
            continue
        if not _passes_hurdle(
            candidate,
            proposed_notional=allocation_notional,
            month_to_date_turnover_eur=current_turnover,
            open_hurdle_bps=context.open_hurdle_bps,
            apply_prediction_hurdle=context.apply_prediction_hurdle,
            account_currency=context.account_currency,
        ):
            continue
        entry_cost_estimate = _estimate_order_cost(
            candidate,
            order_value_eur=allocation_notional,
            month_to_date_turnover_eur=current_turnover,
            account_currency=context.account_currency,
        )
        required_cash = allocation_notional + entry_cost_estimate.total_cost_amount_eur
        if required_cash > max(context.cash_balance - reserved_cash, 0.0):
            continue
        trades.append(
            _build_active_trade(
                candidate,
                context=context,
                allocation_notional=allocation_notional,
                candidate_budget=candidate_budget,
                entry_cost_estimate=entry_cost_estimate,
            ),
        )
        reserved_cash += required_cash
        current_turnover = entry_cost_estimate.month_to_date_turnover_eur_after_order
    return current_turnover


def allocate_signal_candidates(
    *,
    trade_date: pd.Timestamp,
    candidates: list[SignalCandidate],
    active_trades: list[ActiveTrade],
    current_equity: float,
    cash_balance: float,
    hold_period_days: int,
    allocation_fraction: float,
    action_cap_fraction: float,
    gross_cap_fraction: float,
    adv_participation_limit: float,
    neutrality_mode: str,
    open_hurdle_bps: float,
    apply_prediction_hurdle: bool,
    unique_dates: pd.Index,
    month_to_date_turnover_eur: float,
    account_currency: str,
) -> list[ActiveTrade]:
    if neutrality_mode != "long_only":
        raise ValueError("XTB cash-equity backtest only supports long_only.")
    if current_equity <= 0.0 or cash_balance <= 0.0:
        return []
    exit_date = _resolve_exit_date(trade_date, unique_dates, hold_period_days)
    if exit_date is None:
        return []
    ordered_candidates = sorted(candidates, key=lambda candidate: candidate.prediction, reverse=True)
    if not ordered_candidates:
        return []
    target_notional = min(gross_cap_fraction * current_equity, cash_balance)
    sector_names, sector_budget = _sector_candidate_budget(
        ordered_candidates,
        target_notional=target_notional,
    )
    context = AllocationContext(
        trade_date=trade_date,
        exit_date=exit_date,
        current_equity=current_equity,
        cash_balance=cash_balance,
        allocation_fraction=allocation_fraction,
        action_cap_fraction=action_cap_fraction,
        adv_participation_limit=adv_participation_limit,
        open_hurdle_bps=open_hurdle_bps,
        apply_prediction_hurdle=apply_prediction_hurdle,
        account_currency=account_currency,
        month_to_date_turnover_eur=month_to_date_turnover_eur,
    )
    sector_candidates_by_name: dict[str, list[SignalCandidate]] = {}
    for candidate in ordered_candidates:
        sector_candidates_by_name.setdefault(candidate.sector, []).append(candidate)
    trades: list[ActiveTrade] = []
    current_turnover = month_to_date_turnover_eur
    for sector_name in sector_names:
        sector_candidates = sector_candidates_by_name.get(sector_name, [])
        if not sector_candidates:
            continue
        candidate_budget = sector_budget / len(sector_candidates)
        current_turnover = _allocate_sector_candidates(
            sector_candidates,
            candidate_budget=candidate_budget,
            active_trades=active_trades,
            trades=trades,
            context=context,
            month_to_date_turnover_eur=current_turnover,
        )
    return trades


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
        next_month_key = _month_key(pd.Timestamp(unique_dates[next_position]))
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


def process_prediction_day(
    *,
    state: BacktestState,
    daily_predictions: pd.DataFrame,
    unique_dates: pd.Index,
    top_fraction: float,
    allocation_fraction: float,
    action_cap_fraction: float,
    gross_cap_fraction: float,
    adv_participation_limit: float,
    neutrality_mode: str,
    open_hurdle_bps: float,
    apply_prediction_hurdle: bool,
    hold_period_days: int,
    cost_config: XtbCostConfig,
    logger: Any | None = None,
) -> None:
    if daily_predictions.empty:
        return
    trade_date = pd.Timestamp(pd.to_datetime(daily_predictions[DATE_COLUMN]).iloc[0])
    opening_equity = float(state.current_equity)
    opening_cash = float(state.cash_balance)
    exiting_today = [trade for trade in state.active_trades if trade.exit_date == trade_date]
    surviving_trades = [trade for trade in state.active_trades if trade.exit_date != trade_date]

    realized_pnl_today = 0.0
    gross_pnl_exits = 0.0
    exit_cost_amount = 0.0
    closed_notional = 0.0
    closed_trade_count = 0
    for trade in exiting_today:
        exit_cost_estimate = estimate_trade_cost(
            trade.spec,
            order_value_eur=trade.notional,
            month_to_date_turnover_eur=_resolve_month_to_date_turnover(state, trade_date),
            account_currency=cost_config.account_currency,
        )
        _register_order_turnover(state, trade_date, trade.notional)
        closed_trade = finalize_trade(
            trade,
            exit_cost_estimate=exit_cost_estimate,
        )
        state.closed_trades.append(closed_trade)
        gross_pnl_exits += closed_trade.gross_pnl_amount
        exit_cost_amount += closed_trade.exit_transaction_cost_amount
        closed_notional += abs(closed_trade.notional)
        closed_trade_count += 1
        realized_pnl_today += closed_trade.gross_pnl_amount - closed_trade.exit_transaction_cost_amount
        state.cash_balance += closed_trade.exit_cash_flow_amount

    state.active_trades = surviving_trades
    candidates = build_daily_signal_candidates(
        daily_predictions,
        top_fraction=top_fraction,
        cost_config=cost_config,
        expected_holding_days=hold_period_days,
        neutrality_mode=neutrality_mode,
    )
    new_trades = allocate_signal_candidates(
        trade_date=trade_date,
        candidates=candidates,
        active_trades=state.active_trades,
        current_equity=max(opening_equity, 0.0),
        cash_balance=state.cash_balance,
        hold_period_days=hold_period_days,
        allocation_fraction=allocation_fraction,
        action_cap_fraction=action_cap_fraction,
        gross_cap_fraction=gross_cap_fraction,
        adv_participation_limit=adv_participation_limit,
        neutrality_mode=neutrality_mode,
        open_hurdle_bps=open_hurdle_bps,
        apply_prediction_hurdle=apply_prediction_hurdle,
        unique_dates=unique_dates,
        month_to_date_turnover_eur=_resolve_month_to_date_turnover(state, trade_date),
        account_currency=cost_config.account_currency,
    )
    opened_notional = float(sum(abs(trade.notional) for trade in new_trades))
    entry_cost_amount = 0.0
    for trade in new_trades:
        entry_cost_amount += trade.entry_transaction_cost_amount
        _register_order_turnover(state, trade_date, trade.notional)
        state.cash_balance -= trade.notional + trade.entry_transaction_cost_amount
        realized_pnl_today -= trade.entry_transaction_cost_amount
        state.active_trades.append(trade)

    active_notional_end = float(sum(trade.notional for trade in state.active_trades))
    custody_fee_amount = _apply_month_end_custody_fee(
        state,
        trade_date=trade_date,
        unique_dates=unique_dates,
        active_notional_end=active_notional_end,
    )
    realized_pnl_today -= custody_fee_amount
    state.current_equity = state.cash_balance + active_notional_end

    turnover_notional = float(
        sum(abs(trade.notional) for trade in exiting_today)
        + sum(abs(trade.notional) for trade in new_trades)
    )
    capacity_binding_share = 0.0
    if new_trades:
        capacity_binding_share = float(
            np.mean([1.0 if trade.capacity_bound else 0.0 for trade in new_trades]),
        )
    gross_exposure = 0.0
    if state.current_equity > 0.0:
        gross_exposure = active_notional_end / state.current_equity
    benchmark_return = float(
        np.mean(np.exp(daily_predictions[REALIZED_RETURN_COLUMN].to_numpy(dtype=np.float64)) - 1.0)
    )
    realized_return = realized_pnl_today / max(opening_equity, 1e-12)
    net_cash_flow = state.cash_balance - opening_cash
    reconciliation_error = state.current_equity - (state.cash_balance + active_notional_end)
    if not math.isclose(reconciliation_error, 0.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Backtest cashflow reconciliation failed on {trade_date.date()}: error={reconciliation_error:.10f}",
        )
    state.daily_rows.append({
        "date": trade_date,
        "starting_equity": opening_equity,
        "equity": state.current_equity,
        "ending_equity": state.current_equity,
        "cash_balance": state.cash_balance,
        "total_return": state.current_equity / state.initial_equity - 1.0,
        "realized_pnl": realized_pnl_today,
        "realized_return": realized_return,
        "benchmark_return": benchmark_return,
        "gross_pnl_exits": gross_pnl_exits,
        "entry_cost_amount": entry_cost_amount,
        "exit_cost_amount": exit_cost_amount,
        "custody_fee_amount": custody_fee_amount,
        "net_cash_flow": net_cash_flow,
        "turnover": turnover_notional / max(opening_equity, 1e-12),
        "month_to_date_turnover_eur": _resolve_month_to_date_turnover(state, trade_date),
        "opened_notional": opened_notional,
        "closed_notional": closed_notional,
        "active_notional_end": active_notional_end,
        "gross_exposure": gross_exposure,
        "capacity_binding_share": capacity_binding_share,
        "active_trade_count": len(state.active_trades),
        "opened_trade_count": len(new_trades),
        "closed_trade_count": closed_trade_count,
        "reconciliation_error": reconciliation_error,
    })
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
    *,
    top_fraction: float,
    allocation_fraction: float,
    action_cap_fraction: float,
    gross_cap_fraction: float,
    adv_participation_limit: float,
    neutrality_mode: str,
    open_hurdle_bps: float,
    apply_prediction_hurdle: bool,
    hold_period_days: int,
    cost_config: XtbCostConfig,
    logger: Any | None = None,
    starting_cash_eur: float = 100_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    ordered_predictions = predictions.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    unique_dates = pd.Index(pd.to_datetime(ordered_predictions[DATE_COLUMN]).drop_duplicates().sort_values())
    state = BacktestState(
        initial_equity=starting_cash_eur,
        current_equity=starting_cash_eur,
        cash_balance=starting_cash_eur,
    )
    for trade_date in unique_dates:
        process_prediction_day(
            state=state,
            daily_predictions=pd.DataFrame(
                ordered_predictions.loc[ordered_predictions[DATE_COLUMN] == trade_date].copy(),
            ),
            unique_dates=unique_dates,
            top_fraction=top_fraction,
            allocation_fraction=allocation_fraction,
            action_cap_fraction=action_cap_fraction,
            gross_cap_fraction=gross_cap_fraction,
            adv_participation_limit=adv_participation_limit,
            neutrality_mode=neutrality_mode,
            open_hurdle_bps=open_hurdle_bps,
            apply_prediction_hurdle=apply_prediction_hurdle,
            hold_period_days=hold_period_days,
            cost_config=cost_config,
            logger=logger,
        )
    return finalize_backtest_state(state)
