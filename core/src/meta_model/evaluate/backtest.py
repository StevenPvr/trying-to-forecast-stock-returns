from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from core.src.meta_model.broker_xtb.costs import estimate_trade_cost
from core.src.meta_model.broker_xtb.margin import estimate_margin
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
    sector: str = "__unknown_sector__"
    beta: float = 1.0
    adv_cap_notional: float = float("inf")
    instrument_group: str = "stock_cfd"
    expected_entry_cost_rate: float = 0.0
    expected_exit_cost_rate: float = 0.0
    expected_financing_cost_rate: float = 0.0
    expected_total_cost_rate: float = 0.0
    margin_requirement: float = 0.20


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
    entry_transaction_cost_amount: float = 0.0
    accumulated_financing_cost_amount: float = 0.0
    expected_entry_cost_rate: float = 0.0
    expected_exit_cost_rate: float = 0.0
    expected_financing_cost_rate: float = 0.0
    expected_total_cost_rate: float = 0.0
    margin_requirement: float = 0.20
    required_margin: float = 0.0
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
    financing_cost: float
    net_return: float
    pnl_amount: float
    exit_cash_flow_amount: float
    gross_pnl_amount: float = 0.0
    entry_transaction_cost_amount: float = 0.0
    exit_transaction_cost_amount: float = 0.0
    total_transaction_cost_amount: float = 0.0
    financing_cost_amount: float = 0.0
    net_pnl_amount: float = 0.0


@dataclass(frozen=True)
class XtbCostConfig:
    fx_conversion_bps: float = 5.0
    broker_spec_provider: BrokerSpecProvider = field(default_factory=build_default_spec_provider)


@dataclass(frozen=True)
class AllocationContext:
    trade_date: pd.Timestamp
    exit_date: pd.Timestamp
    current_equity: float
    allocation_fraction: float
    action_cap_fraction: float
    adv_participation_limit: float
    open_hurdle_bps: float
    apply_prediction_hurdle: bool


def _build_active_trades() -> list[ActiveTrade]:
    return []


def _build_closed_trades() -> list[ClosedTrade]:
    return []


def _build_daily_rows() -> list[dict[str, object]]:
    return []


@dataclass
class BacktestState:
    initial_equity: float = 1.0
    current_equity: float = 1.0
    active_trades: list[ActiveTrade] = field(default_factory=_build_active_trades)
    closed_trades: list[ClosedTrade] = field(default_factory=_build_closed_trades)
    daily_rows: list[dict[str, object]] = field(default_factory=_build_daily_rows)


def _validate_candidate_inputs(
    daily_predictions: pd.DataFrame,
    *,
    top_fraction: float,
) -> None:
    if daily_predictions.empty:
        return
    if top_fraction <= 0.0 or top_fraction > 0.5:
        raise ValueError("top_fraction must be in the interval (0, 0.5].")


def _select_signal_blocks(
    daily_predictions: pd.DataFrame,
    *,
    top_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    selection_count = max(1, int(np.ceil(len(daily_predictions) * top_fraction)))
    short_block = pd.DataFrame(
        daily_predictions.nsmallest(selection_count, columns=PREDICTION_COLUMN, keep="first").reset_index(drop=True),
    )
    long_block = pd.DataFrame(
        daily_predictions.nlargest(selection_count, columns=PREDICTION_COLUMN, keep="first").reset_index(drop=True),
    )
    trade_date = pd.Timestamp(pd.to_datetime(daily_predictions[DATE_COLUMN]).iloc[0])
    return long_block, short_block, trade_date


def _resolve_financing_daily_rate(
    *,
    expected_holding_days: int,
    financing_cost_rate: float,
) -> float:
    if expected_holding_days <= 0:
        return 0.0
    return financing_cost_rate / expected_holding_days


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
    side: str,
    rank: int,
    trade_date: pd.Timestamp,
    cost_config: XtbCostConfig,
    expected_holding_days: int,
) -> SignalCandidate:
    open_price = float(getattr(row, "stock_open_price", 0.0) or 0.0)
    trading_volume = float(getattr(row, "stock_trading_volume", 0.0) or 0.0)
    ticker = str(getattr(row, TICKER_COLUMN))
    spec = cost_config.broker_spec_provider.resolve(ticker, trade_date)
    cost_estimate = estimate_trade_cost(
        spec,
        side=side,
        expected_holding_days=expected_holding_days,
        fx_conversion_bps=cost_config.fx_conversion_bps,
    )
    financing_daily_rate = _resolve_financing_daily_rate(
        expected_holding_days=expected_holding_days,
        financing_cost_rate=cost_estimate.financing_cost_rate,
    )
    adv_cap_notional = _resolve_adv_cap_notional(
        open_price=open_price,
        trading_volume=trading_volume,
        max_adv_participation=spec.max_adv_participation,
    )
    return SignalCandidate(
        ticker=ticker,
        side=side,
        prediction=float(getattr(row, PREDICTION_COLUMN)),
        signal_rank=rank,
        realized_log_return=float(getattr(row, REALIZED_RETURN_COLUMN, 0.0)),
        sector=str(getattr(row, "company_sector", "__unknown_sector__") or "__unknown_sector__"),
        beta=max(0.1, abs(float(getattr(row, "company_beta", 1.0) or 1.0))),
        adv_cap_notional=adv_cap_notional,
        instrument_group=spec.instrument_group,
        expected_entry_cost_rate=cost_estimate.entry_cost_rate,
        expected_exit_cost_rate=cost_estimate.exit_cost_rate,
        expected_financing_cost_rate=financing_daily_rate,
        expected_total_cost_rate=cost_estimate.total_cost_rate,
        margin_requirement=spec.margin_requirement,
    )


def _build_candidates_from_block(
    block: pd.DataFrame,
    *,
    side: str,
    trade_date: pd.Timestamp,
    cost_config: XtbCostConfig,
    expected_holding_days: int,
) -> list[SignalCandidate]:
    return [
        _build_signal_candidate(
            row,
            side=side,
            rank=rank,
            trade_date=trade_date,
            cost_config=cost_config,
            expected_holding_days=expected_holding_days,
        )
        for rank, row in enumerate(block.itertuples(index=False), start=1)
    ]


def build_daily_signal_candidates(
    daily_predictions: pd.DataFrame,
    *,
    top_fraction: float,
    cost_config: XtbCostConfig,
    expected_holding_days: int,
) -> list[SignalCandidate]:
    if daily_predictions.empty:
        return []
    _validate_candidate_inputs(daily_predictions, top_fraction=top_fraction)
    long_block, short_block, trade_date = _select_signal_blocks(
        daily_predictions,
        top_fraction=top_fraction,
    )
    return [
        *_build_candidates_from_block(
            long_block,
            side="long",
            trade_date=trade_date,
            cost_config=cost_config,
            expected_holding_days=expected_holding_days,
        ),
        *_build_candidates_from_block(
            short_block,
            side="short",
            trade_date=trade_date,
            cost_config=cost_config,
            expected_holding_days=expected_holding_days,
        ),
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


def _prepare_side_candidates(
    candidates: list[SignalCandidate],
    *,
    neutrality_mode: str,
) -> tuple[list[SignalCandidate], list[SignalCandidate]]:
    ordered_candidates = sorted(candidates, key=lambda candidate: abs(candidate.prediction), reverse=True)
    long_candidates = [candidate for candidate in ordered_candidates if candidate.side == "long"]
    short_candidates = [candidate for candidate in ordered_candidates if candidate.side == "short"]
    if neutrality_mode not in {"sector_neutral", "sector_beta_neutral"}:
        return long_candidates, short_candidates
    long_sectors = {candidate.sector for candidate in long_candidates}
    short_sectors = {candidate.sector for candidate in short_candidates}
    eligible_sectors = long_sectors.intersection(short_sectors)
    return (
        [candidate for candidate in long_candidates if candidate.sector in eligible_sectors],
        [candidate for candidate in short_candidates if candidate.sector in eligible_sectors],
    )


def _resolve_side_target_notionals(
    long_candidates: list[SignalCandidate],
    short_candidates: list[SignalCandidate],
    *,
    current_equity: float,
    gross_cap_fraction: float,
    neutrality_mode: str,
) -> tuple[float, float]:
    gross_target_notional = gross_cap_fraction * current_equity
    if neutrality_mode != "sector_beta_neutral":
        half_gross = gross_target_notional / 2.0
        return half_gross, half_gross
    long_beta = float(np.mean([candidate.beta for candidate in long_candidates]))
    short_beta = float(np.mean([candidate.beta for candidate in short_candidates]))
    long_target_notional = gross_target_notional * short_beta / max(long_beta + short_beta, 1e-12)
    short_target_notional = gross_target_notional - long_target_notional
    return long_target_notional, short_target_notional


def _existing_symbol_net(symbol: str, active_trades: list[ActiveTrade], pending_trades: list[ActiveTrade]) -> float:
    active_symbol_net = sum(
        trade.notional if trade.side == "long" else -trade.notional
        for trade in active_trades
        if trade.ticker == symbol
    )
    pending_symbol_net = sum(
        trade.notional if trade.side == "long" else -trade.notional
        for trade in pending_trades
        if trade.ticker == symbol
    )
    return float(active_symbol_net + pending_symbol_net)


def _sector_candidate_budget(
    side_candidates: list[SignalCandidate],
    *,
    side_target_notional: float,
) -> tuple[list[str], float]:
    sector_names = sorted({candidate.sector for candidate in side_candidates})
    sector_budget = side_target_notional / max(len(sector_names), 1)
    return sector_names, sector_budget


def _passes_hurdle(
    candidate: SignalCandidate,
    *,
    open_hurdle_bps: float,
    apply_prediction_hurdle: bool,
) -> bool:
    if not apply_prediction_hurdle:
        return True
    hurdle_rate = candidate.expected_total_cost_rate + open_hurdle_bps / 10_000.0
    return abs(candidate.prediction) > hurdle_rate


def _is_opposite_to_existing_position(
    candidate: SignalCandidate,
    *,
    current_symbol_net: float,
) -> bool:
    return (
        (current_symbol_net > 0.0 and candidate.side == "short")
        or (current_symbol_net < 0.0 and candidate.side == "long")
    )


def _resolve_allocation_notional(
    candidate: SignalCandidate,
    *,
    candidate_budget: float,
    current_equity: float,
    current_symbol_net: float,
    allocation_fraction: float,
    action_cap_fraction: float,
    adv_participation_limit: float,
) -> float:
    symbol_room_notional = max(0.0, action_cap_fraction * current_equity - abs(current_symbol_net))
    adv_room_notional = candidate.adv_cap_notional * max(adv_participation_limit / 0.05, 0.0)
    return min(
        candidate_budget,
        allocation_fraction * current_equity,
        symbol_room_notional,
        adv_room_notional,
    )


def _estimate_margin_requirement(
    *,
    candidate: SignalCandidate,
    allocation_notional: float,
    context: AllocationContext,
    used_margin: float,
) -> float:
    margin_estimate = estimate_margin(
        XtbInstrumentSpec(
            symbol=candidate.ticker,
            instrument_group=candidate.instrument_group,
            currency="USD",
            spread_bps=0.0,
            slippage_bps=0.0,
            long_swap_bps_daily=0.0,
            short_swap_bps_daily=0.0,
            margin_requirement=candidate.margin_requirement,
            max_adv_participation=context.adv_participation_limit,
            effective_from=context.trade_date.isoformat(),
        ),
        notional=float(allocation_notional),
        available_equity=max(context.current_equity - used_margin, 0.0),
    )
    if margin_estimate.headroom_after_trade < 0.0:
        return -1.0
    return float(margin_estimate.required_margin)


def _build_active_trade(
    candidate: SignalCandidate,
    *,
    context: AllocationContext,
    allocation_notional: float,
    candidate_budget: float,
    required_margin: float,
) -> ActiveTrade:
    return ActiveTrade(
        ticker=candidate.ticker,
        side=candidate.side,
        entry_date=context.trade_date,
        exit_date=context.exit_date,
        notional=float(allocation_notional),
        predicted_return=candidate.prediction,
        realized_log_return=candidate.realized_log_return,
        signal_rank=candidate.signal_rank,
        expected_entry_cost_rate=candidate.expected_entry_cost_rate,
        expected_exit_cost_rate=candidate.expected_exit_cost_rate,
        expected_financing_cost_rate=candidate.expected_financing_cost_rate,
        expected_total_cost_rate=candidate.expected_total_cost_rate,
        margin_requirement=candidate.margin_requirement,
        required_margin=required_margin,
        capacity_bound=allocation_notional < candidate_budget,
    )


def _allocate_sector_candidates(
    sector_candidates: list[SignalCandidate],
    *,
    candidate_budget: float,
    active_trades: list[ActiveTrade],
    trades: list[ActiveTrade],
    context: AllocationContext,
) -> None:
    for candidate in sector_candidates:
        if not _passes_hurdle(
            candidate,
            open_hurdle_bps=context.open_hurdle_bps,
            apply_prediction_hurdle=context.apply_prediction_hurdle,
        ):
            continue
        current_symbol_net = _existing_symbol_net(candidate.ticker, active_trades, trades)
        if _is_opposite_to_existing_position(candidate, current_symbol_net=current_symbol_net):
            continue
        allocation_notional = _resolve_allocation_notional(
            candidate,
            candidate_budget=candidate_budget,
            current_equity=context.current_equity,
            current_symbol_net=current_symbol_net,
            allocation_fraction=context.allocation_fraction,
            action_cap_fraction=context.action_cap_fraction,
            adv_participation_limit=context.adv_participation_limit,
        )
        if allocation_notional <= 0.0:
            continue
        used_margin = sum(trade.required_margin for trade in [*active_trades, *trades])
        required_margin = _estimate_margin_requirement(
            candidate=candidate,
            allocation_notional=allocation_notional,
            context=context,
            used_margin=used_margin,
        )
        if required_margin < 0.0:
            continue
        trades.append(
            _build_active_trade(
                candidate,
                context=context,
                allocation_notional=allocation_notional,
                candidate_budget=candidate_budget,
                required_margin=required_margin,
            ),
        )


def _allocate_side_candidates(
    side_candidates: list[SignalCandidate],
    *,
    side_target_notional: float,
    active_trades: list[ActiveTrade],
    context: AllocationContext,
) -> list[ActiveTrade]:
    if not side_candidates or side_target_notional <= 0.0:
        return []
    sector_names, sector_budget = _sector_candidate_budget(
        side_candidates,
        side_target_notional=side_target_notional,
    )
    sector_candidates_by_name: dict[str, list[SignalCandidate]] = {}
    for candidate in side_candidates:
        sector_candidates_by_name.setdefault(candidate.sector, []).append(candidate)
    trades: list[ActiveTrade] = []
    for sector_name in sector_names:
        sector_candidates = sector_candidates_by_name.get(sector_name, [])
        if not sector_candidates:
            continue
        candidate_budget = sector_budget / len(sector_candidates)
        _allocate_sector_candidates(
            sector_candidates,
            candidate_budget=candidate_budget,
            active_trades=active_trades,
            trades=trades,
            context=context,
        )
    return trades


def allocate_signal_candidates(
    *,
    trade_date: pd.Timestamp,
    candidates: list[SignalCandidate],
    active_trades: list[ActiveTrade],
    current_equity: float,
    hold_period_days: int,
    allocation_fraction: float,
    action_cap_fraction: float,
    gross_cap_fraction: float,
    adv_participation_limit: float,
    neutrality_mode: str,
    open_hurdle_bps: float,
    apply_prediction_hurdle: bool,
    unique_dates: pd.Index,
) -> list[ActiveTrade]:
    if current_equity <= 0.0:
        return []
    exit_date = _resolve_exit_date(trade_date, unique_dates, hold_period_days)
    if exit_date is None:
        return []
    long_candidates, short_candidates = _prepare_side_candidates(
        candidates,
        neutrality_mode=neutrality_mode,
    )
    if not long_candidates or not short_candidates:
        return []
    long_target_notional, short_target_notional = _resolve_side_target_notionals(
        long_candidates,
        short_candidates,
        current_equity=current_equity,
        gross_cap_fraction=gross_cap_fraction,
        neutrality_mode=neutrality_mode,
    )
    context = AllocationContext(
        trade_date=trade_date,
        exit_date=exit_date,
        current_equity=current_equity,
        allocation_fraction=allocation_fraction,
        action_cap_fraction=action_cap_fraction,
        adv_participation_limit=adv_participation_limit,
        open_hurdle_bps=open_hurdle_bps,
        apply_prediction_hurdle=apply_prediction_hurdle,
    )
    return [
        *_allocate_side_candidates(
            long_candidates,
            side_target_notional=long_target_notional,
            active_trades=active_trades,
            context=context,
        ),
        *_allocate_side_candidates(
            short_candidates,
            side_target_notional=short_target_notional,
            active_trades=active_trades,
            context=context,
        ),
    ]


def _apply_daily_financing(
    active_trades: list[ActiveTrade],
    *,
    cost_config: XtbCostConfig,
) -> tuple[list[ActiveTrade], float]:
    del cost_config
    updated_trades: list[ActiveTrade] = []
    total_financing_amount = 0.0
    for trade in active_trades:
        daily_financing_rate = max(trade.expected_financing_cost_rate, 0.0)
        financing_amount = trade.notional * daily_financing_rate
        total_financing_amount += financing_amount
        updated_trades.append(
            replace(
                trade,
                accumulated_financing_cost_amount=(
                    trade.accumulated_financing_cost_amount + financing_amount
                ),
            ),
        )
    return updated_trades, total_financing_amount


def finalize_trade(
    trade: ActiveTrade,
    *,
    cost_config: XtbCostConfig,
) -> ClosedTrade:
    del cost_config
    arithmetic_return = float(np.exp(trade.realized_log_return) - 1.0)
    signed_gross_return = arithmetic_return if trade.side == "long" else -arithmetic_return
    close_transaction_cost_amount = trade.notional * trade.expected_exit_cost_rate
    total_transaction_cost_amount = trade.entry_transaction_cost_amount + close_transaction_cost_amount
    financing_cost_amount = trade.accumulated_financing_cost_amount
    financing_cost = financing_cost_amount / trade.notional
    transaction_cost = total_transaction_cost_amount / trade.notional
    net_return = signed_gross_return - transaction_cost - financing_cost
    gross_pnl_amount = trade.notional * signed_gross_return
    net_pnl_amount = gross_pnl_amount - total_transaction_cost_amount - financing_cost_amount
    pnl_amount = net_pnl_amount
    exit_cash_flow_amount = gross_pnl_amount - close_transaction_cost_amount
    return ClosedTrade(
        ticker=trade.ticker,
        side=trade.side,
        entry_date=trade.entry_date,
        exit_date=trade.exit_date,
        notional=trade.notional,
        predicted_return=trade.predicted_return,
        realized_log_return=trade.realized_log_return,
        signal_rank=trade.signal_rank,
        gross_return=signed_gross_return,
        transaction_cost=transaction_cost,
        financing_cost=financing_cost,
        net_return=net_return,
        pnl_amount=pnl_amount,
        exit_cash_flow_amount=exit_cash_flow_amount,
        gross_pnl_amount=gross_pnl_amount,
        entry_transaction_cost_amount=trade.entry_transaction_cost_amount,
        exit_transaction_cost_amount=close_transaction_cost_amount,
        total_transaction_cost_amount=total_transaction_cost_amount,
        financing_cost_amount=financing_cost_amount,
        net_pnl_amount=net_pnl_amount,
    )


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
    state.active_trades, financing_amount_today = _apply_daily_financing(
        state.active_trades,
        cost_config=cost_config,
    )
    state.current_equity -= financing_amount_today
    exiting_today = [trade for trade in state.active_trades if trade.exit_date == trade_date]
    surviving_trades = [trade for trade in state.active_trades if trade.exit_date != trade_date]

    realized_pnl_today = -financing_amount_today
    gross_pnl_exits = 0.0
    exit_cost_amount = 0.0
    closed_notional = 0.0
    closed_trade_count = 0
    for trade in exiting_today:
        closed_trade = finalize_trade(
            trade,
            cost_config=cost_config,
        )
        state.closed_trades.append(closed_trade)
        gross_pnl_exits += closed_trade.gross_pnl_amount
        exit_cost_amount += closed_trade.exit_transaction_cost_amount
        closed_notional += abs(closed_trade.notional)
        closed_trade_count += 1
        realized_pnl_today += closed_trade.exit_cash_flow_amount
        state.current_equity += closed_trade.exit_cash_flow_amount

    state.active_trades = surviving_trades
    candidates = build_daily_signal_candidates(
        daily_predictions,
        top_fraction=top_fraction,
        cost_config=cost_config,
        expected_holding_days=hold_period_days,
    )
    new_trades = allocate_signal_candidates(
        trade_date=trade_date,
        candidates=candidates,
        active_trades=state.active_trades,
        current_equity=state.current_equity,
        hold_period_days=hold_period_days,
        allocation_fraction=allocation_fraction,
        action_cap_fraction=action_cap_fraction,
        gross_cap_fraction=gross_cap_fraction,
        adv_participation_limit=adv_participation_limit,
        neutrality_mode=neutrality_mode,
        open_hurdle_bps=open_hurdle_bps,
        apply_prediction_hurdle=apply_prediction_hurdle,
        unique_dates=unique_dates,
    )
    opened_notional = float(sum(abs(trade.notional) for trade in new_trades))
    entry_cost_amount = 0.0
    for trade in new_trades:
        entry_transaction_cost_amount = trade.notional * trade.expected_entry_cost_rate
        entry_cost_amount += entry_transaction_cost_amount
        state.current_equity -= entry_transaction_cost_amount
        realized_pnl_today -= entry_transaction_cost_amount
        state.active_trades.append(
            replace(
                trade,
                entry_transaction_cost_amount=entry_transaction_cost_amount,
            ),
        )
    if hold_period_days == 0 and new_trades:
        intraday_closures = [trade for trade in state.active_trades if trade.entry_date == trade_date]
        remaining_trades = [trade for trade in state.active_trades if trade.entry_date != trade_date]
        for trade in intraday_closures:
            closed_trade = finalize_trade(trade, cost_config=cost_config)
            state.closed_trades.append(closed_trade)
            gross_pnl_exits += closed_trade.gross_pnl_amount
            exit_cost_amount += closed_trade.exit_transaction_cost_amount
            closed_notional += abs(closed_trade.notional)
            closed_trade_count += 1
            realized_pnl_today += closed_trade.exit_cash_flow_amount
            state.current_equity += closed_trade.exit_cash_flow_amount
        state.active_trades = remaining_trades

    return_base_equity = max(state.current_equity - realized_pnl_today, 1e-12)
    signed_notionals = np.asarray(
        [
            trade.notional if trade.side == "long" else -trade.notional
            for trade in state.active_trades
        ],
        dtype=np.float64,
    )
    gross_exposure = (
        float(np.abs(signed_notionals).sum() / max(state.current_equity, 1e-12))
        if signed_notionals.size > 0
        else 0.0
    )
    net_exposure = (
        float(signed_notionals.sum() / max(state.current_equity, 1e-12))
        if signed_notionals.size > 0
        else 0.0
    )
    turnover_notional = float(
        sum(abs(trade.notional) for trade in exiting_today)
        + sum(abs(trade.notional) for trade in new_trades)
    )
    capacity_binding_share = 0.0
    if new_trades:
        capacity_binding_share = float(
            np.mean([1.0 if trade.capacity_bound else 0.0 for trade in new_trades]),
        )
    used_margin = float(sum(trade.required_margin for trade in state.active_trades))
    margin_headroom = float(max(state.current_equity - used_margin, 0.0))
    active_notional_end = float(sum(abs(trade.notional) for trade in state.active_trades))
    benchmark_return = float(
        np.mean(np.exp(daily_predictions[REALIZED_RETURN_COLUMN].to_numpy(dtype=np.float64)) - 1.0)
    )
    realized_return = (
        realized_pnl_today
        if math.isclose(state.current_equity, 0.0, rel_tol=0.0, abs_tol=1e-12)
        else realized_pnl_today / return_base_equity
    )
    net_cash_flow = gross_pnl_exits - entry_cost_amount - exit_cost_amount - financing_amount_today
    reconciliation_error = state.current_equity - (opening_equity + net_cash_flow)
    if not math.isclose(reconciliation_error, 0.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Backtest cashflow reconciliation failed on {trade_date.date()}: error={reconciliation_error:.10f}",
        )
    state.daily_rows.append({
        "date": trade_date,
        "starting_equity": opening_equity,
        "equity": state.current_equity,
        "ending_equity": state.current_equity,
        "total_return": state.current_equity / state.initial_equity - 1.0,
        "realized_pnl": realized_pnl_today,
        "realized_return": realized_return,
        "benchmark_return": benchmark_return,
        "gross_pnl_exits": gross_pnl_exits,
        "entry_cost_amount": entry_cost_amount,
        "exit_cost_amount": exit_cost_amount,
        "financing_amount": financing_amount_today,
        "net_cash_flow": net_cash_flow,
        "turnover": turnover_notional / return_base_equity,
        "opened_notional": opened_notional,
        "closed_notional": closed_notional,
        "active_notional_end": active_notional_end,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "capacity_binding_share": capacity_binding_share,
        "margin_headroom": margin_headroom,
        "reconciliation_error": reconciliation_error,
        "active_trade_count": len(state.active_trades),
        "opened_trade_count": len(new_trades),
        "closed_trade_count": closed_trade_count,
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
    if daily_frame.empty:
        summary = {
            "final_equity": 1.0,
            "net_pnl": 0.0,
            "gross_pnl_before_costs": 0.0,
            "entry_cost_amount_total": 0.0,
            "exit_cost_amount_total": 0.0,
            "transaction_cost_amount_total": 0.0,
            "financing_cost_amount_total": 0.0,
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
            "average_net_exposure": 0.0,
            "capacity_binding_share": 0.0,
            "margin_headroom": 0.0,
            "realized_beta": 0.0,
            "trade_count": 0.0,
            "win_rate": 0.0,
            "avg_entry_cost_rate": 0.0,
            "avg_exit_cost_rate": 0.0,
            "avg_roundtrip_cost_rate": 0.0,
            "avg_financing_cost_rate": 0.0,
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
    transaction_cost_amount_total = entry_cost_amount_total + exit_cost_amount_total
    financing_cost_amount_total = float(daily_frame["financing_amount"].sum())
    net_pnl_after_costs = float(
        gross_pnl_before_costs - transaction_cost_amount_total - financing_cost_amount_total,
    )
    running_peak = equity_series.cummax()
    drawdown = equity_series / running_peak - 1.0
    annualized_return = float(equity_series.iloc[-1] ** (252.0 / max(len(daily_frame), 1)) - 1.0)
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
    realized_beta = 0.0
    benchmark_variance = float(benchmark_returns.var(ddof=0))
    if benchmark_variance > 0.0 and len(daily_returns) > 1:
        realized_beta = float(np.cov(daily_returns, benchmark_returns, ddof=0)[0, 1] / benchmark_variance)
    win_rate = 0.0
    if not trades_frame.empty:
        win_rate = float((trades_frame["net_return"] > 0.0).mean())
    avg_entry_cost_rate = 0.0
    avg_exit_cost_rate = 0.0
    avg_roundtrip_cost_rate = 0.0
    avg_financing_cost_rate = 0.0
    if not trades_frame.empty:
        avg_entry_cost_rate = float(trades_frame["entry_transaction_cost_amount"].sum() / trades_frame["notional"].sum())
        avg_exit_cost_rate = float(trades_frame["exit_transaction_cost_amount"].sum() / trades_frame["notional"].sum())
        avg_roundtrip_cost_rate = float(
            trades_frame["total_transaction_cost_amount"].sum() / trades_frame["notional"].sum(),
        )
        avg_financing_cost_rate = float(trades_frame["financing_cost_amount"].sum() / trades_frame["notional"].sum())
    cost_drag_total = float(transaction_cost_amount_total + financing_cost_amount_total)
    cost_drag_annualized = float(cost_drag_total / max(len(daily_frame), 1) * 252.0)
    reconciliation_error_max_abs = float(np.abs(daily_frame["reconciliation_error"]).max())
    summary = {
        "final_equity": float(equity_series.iloc[-1]),
        "net_pnl": float(equity_series.iloc[-1] - state.initial_equity),
        "gross_pnl_before_costs": gross_pnl_before_costs,
        "entry_cost_amount_total": entry_cost_amount_total,
        "exit_cost_amount_total": exit_cost_amount_total,
        "transaction_cost_amount_total": transaction_cost_amount_total,
        "financing_cost_amount_total": financing_cost_amount_total,
        "net_pnl_after_costs": net_pnl_after_costs,
        "total_return": float(equity_series.iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_benchmark_return": annualized_benchmark_return,
        "alpha_over_benchmark_net": annualized_return - annualized_benchmark_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "calmar_ratio": float(calmar_ratio),
        "max_drawdown": max_drawdown,
        "turnover_annualized": float(daily_frame["turnover"].mean() * 252.0),
        "average_gross_exposure": float(daily_frame["gross_exposure"].mean()),
        "average_net_exposure": float(daily_frame["net_exposure"].mean()),
        "capacity_binding_share": float(daily_frame["capacity_binding_share"].mean()),
        "margin_headroom": float(daily_frame["margin_headroom"].mean()),
        "realized_beta": realized_beta,
        "trade_count": float(len(trades_frame)),
        "win_rate": float(win_rate),
        "avg_entry_cost_rate": avg_entry_cost_rate,
        "avg_exit_cost_rate": avg_exit_cost_rate,
        "avg_roundtrip_cost_rate": avg_roundtrip_cost_rate,
        "avg_financing_cost_rate": avg_financing_cost_rate,
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    ordered_predictions = predictions.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    unique_dates = pd.Index(pd.to_datetime(ordered_predictions[DATE_COLUMN]).drop_duplicates().sort_values())
    state = BacktestState()

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
