from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import pandas as pd

from core.src.meta_model.evaluate.config import DATE_COLUMN, PREDICTION_COLUMN, TARGET_COLUMN, TICKER_COLUMN


@dataclass(frozen=True)
class SignalCandidate:
    ticker: str
    side: str
    prediction: float
    signal_rank: int
    realized_log_return: float


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


@dataclass(frozen=True)
class XtbCostConfig:
    transaction_cost_rate_per_side: float = 0.003
    long_daily_financing_rate: float = 0.0002269
    short_daily_financing_rate: float = 0.0000231


@dataclass
class BacktestState:
    initial_equity: float = 1.0
    current_equity: float = 1.0
    active_trades: list[ActiveTrade] = field(default_factory=list)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    daily_rows: list[dict[str, object]] = field(default_factory=list)


def build_daily_signal_candidates(
    daily_predictions: pd.DataFrame,
    *,
    top_fraction: float,
) -> list[SignalCandidate]:
    if daily_predictions.empty:
        return []
    if top_fraction <= 0.0 or top_fraction > 0.5:
        raise ValueError("top_fraction must be in the interval (0, 0.5].")

    ordered = daily_predictions.sort_values(PREDICTION_COLUMN).reset_index(drop=True)
    selection_count = max(1, int(np.ceil(len(ordered) * top_fraction)))
    short_block = ordered.head(selection_count).copy()
    long_block = ordered.tail(selection_count).sort_values(PREDICTION_COLUMN, ascending=False).copy()

    candidates: list[SignalCandidate] = []
    for rank, row in enumerate(long_block.itertuples(index=False), start=1):
        candidates.append(
            SignalCandidate(
                ticker=str(getattr(row, TICKER_COLUMN)),
                side="long",
                prediction=float(getattr(row, PREDICTION_COLUMN)),
                signal_rank=rank,
                realized_log_return=float(getattr(row, TARGET_COLUMN, 0.0)),
            ),
        )
    for rank, row in enumerate(short_block.itertuples(index=False), start=1):
        candidates.append(
            SignalCandidate(
                ticker=str(getattr(row, TICKER_COLUMN)),
                side="short",
                prediction=float(getattr(row, PREDICTION_COLUMN)),
                signal_rank=rank,
                realized_log_return=float(getattr(row, TARGET_COLUMN, 0.0)),
            ),
        )
    return candidates


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
    return cast(pd.Timestamp, pd.Timestamp(unique_dates.tolist()[exit_position]))


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
    unique_dates: pd.Index,
) -> list[ActiveTrade]:
    if current_equity <= 0.0:
        return []

    ordered_candidates = sorted(candidates, key=lambda candidate: abs(candidate.prediction), reverse=True)
    active_notionals = active_trades.copy()
    new_trades: list[ActiveTrade] = []

    for candidate in ordered_candidates:
        exit_date = _resolve_exit_date(trade_date, unique_dates, hold_period_days)
        if exit_date is None:
            continue

        gross_used_notional = float(sum(abs(trade.notional) for trade in active_notionals))
        remaining_capacity_notional = max(0.0, gross_cap_fraction * current_equity - gross_used_notional)
        if remaining_capacity_notional <= 0.0:
            break

        current_symbol_net = float(
            sum(
                trade.notional if trade.side == "long" else -trade.notional
                for trade in active_notionals
                if trade.ticker == candidate.ticker
            ),
        )
        if current_symbol_net > 0.0 and candidate.side == "short":
            continue
        if current_symbol_net < 0.0 and candidate.side == "long":
            continue

        symbol_room_notional = max(0.0, action_cap_fraction * current_equity - abs(current_symbol_net))
        allocation_notional = min(
            allocation_fraction * remaining_capacity_notional,
            symbol_room_notional,
            remaining_capacity_notional,
        )
        if allocation_notional <= 0.0:
            continue

        new_trade = ActiveTrade(
            ticker=candidate.ticker,
            side=candidate.side,
            entry_date=trade_date,
            exit_date=exit_date,
            notional=float(allocation_notional),
            predicted_return=candidate.prediction,
            realized_log_return=candidate.realized_log_return,
            signal_rank=candidate.signal_rank,
        )
        new_trades.append(new_trade)
        active_notionals.append(new_trade)
    return new_trades


def _apply_daily_financing(
    active_trades: list[ActiveTrade],
    *,
    cost_config: XtbCostConfig,
) -> tuple[list[ActiveTrade], float]:
    updated_trades: list[ActiveTrade] = []
    total_financing_amount = 0.0
    for trade in active_trades:
        daily_financing_rate = (
            cost_config.long_daily_financing_rate
            if trade.side == "long"
            else cost_config.short_daily_financing_rate
        )
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
    arithmetic_return = float(np.exp(trade.realized_log_return) - 1.0)
    signed_gross_return = arithmetic_return if trade.side == "long" else -arithmetic_return
    close_transaction_cost_amount = trade.notional * cost_config.transaction_cost_rate_per_side
    total_transaction_cost_amount = trade.entry_transaction_cost_amount + close_transaction_cost_amount
    financing_cost = trade.accumulated_financing_cost_amount / trade.notional
    transaction_cost = total_transaction_cost_amount / trade.notional
    net_return = signed_gross_return - transaction_cost - financing_cost
    gross_pnl_amount = trade.notional * signed_gross_return
    pnl_amount = gross_pnl_amount - total_transaction_cost_amount - trade.accumulated_financing_cost_amount
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
    hold_period_days: int,
    cost_config: XtbCostConfig,
    logger: Any | None = None,
) -> None:
    if daily_predictions.empty:
        return

    trade_date = cast(pd.Timestamp, pd.Timestamp(pd.to_datetime(daily_predictions[DATE_COLUMN]).iloc[0]))
    state.active_trades, financing_amount_today = _apply_daily_financing(
        state.active_trades,
        cost_config=cost_config,
    )
    state.current_equity -= financing_amount_today
    exiting_today = [trade for trade in state.active_trades if trade.exit_date == trade_date]
    surviving_trades = [trade for trade in state.active_trades if trade.exit_date != trade_date]

    realized_pnl_today = -financing_amount_today
    for trade in exiting_today:
        closed_trade = finalize_trade(
            trade,
            cost_config=cost_config,
        )
        state.closed_trades.append(closed_trade)
        realized_pnl_today += closed_trade.exit_cash_flow_amount
        state.current_equity += closed_trade.exit_cash_flow_amount

    state.active_trades = surviving_trades
    candidates = build_daily_signal_candidates(daily_predictions, top_fraction=top_fraction)
    new_trades = allocate_signal_candidates(
        trade_date=trade_date,
        candidates=candidates,
        active_trades=state.active_trades,
        current_equity=state.current_equity,
        hold_period_days=hold_period_days,
        allocation_fraction=allocation_fraction,
        action_cap_fraction=action_cap_fraction,
        gross_cap_fraction=gross_cap_fraction,
        unique_dates=unique_dates,
    )
    for trade in new_trades:
        entry_transaction_cost_amount = trade.notional * cost_config.transaction_cost_rate_per_side
        state.current_equity -= entry_transaction_cost_amount
        realized_pnl_today -= entry_transaction_cost_amount
        state.active_trades.append(
            replace(
                trade,
                entry_transaction_cost_amount=entry_transaction_cost_amount,
            ),
        )

    realized_return = (
        realized_pnl_today if state.current_equity == 0.0
        else realized_pnl_today / max(state.current_equity - realized_pnl_today, 1e-12)
    )
    state.daily_rows.append({
        "date": trade_date,
        "equity": state.current_equity,
        "total_return": state.current_equity / state.initial_equity - 1.0,
        "realized_pnl": realized_pnl_today,
        "realized_return": realized_return,
        "active_trade_count": len(state.active_trades),
        "opened_trade_count": len(new_trades),
        "closed_trade_count": len(exiting_today),
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
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0.0,
            "win_rate": 0.0,
        }
        return trades_frame, daily_frame, summary

    equity_series = pd.Series(daily_frame["equity"], dtype=float)
    daily_returns = pd.Series(daily_frame["realized_return"], dtype=float)
    running_peak = equity_series.cummax()
    drawdown = equity_series / running_peak - 1.0
    annualized_return = float(equity_series.iloc[-1] ** (252.0 / max(len(daily_frame), 1)) - 1.0)
    annualized_volatility = float(daily_returns.std(ddof=0) * np.sqrt(252.0))
    sharpe_ratio = 0.0
    if annualized_volatility > 0.0:
        sharpe_ratio = annualized_return / annualized_volatility
    win_rate = 0.0
    if not trades_frame.empty:
        win_rate = float((trades_frame["net_return"] > 0.0).mean())
    summary = {
        "final_equity": float(equity_series.iloc[-1]),
        "total_return": float(equity_series.iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(drawdown.min()),
        "trade_count": float(len(trades_frame)),
        "win_rate": float(win_rate),
    }
    return trades_frame, daily_frame, summary


def run_signal_backtest(
    predictions: pd.DataFrame,
    *,
    top_fraction: float,
    allocation_fraction: float,
    action_cap_fraction: float,
    gross_cap_fraction: float,
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
            hold_period_days=hold_period_days,
            cost_config=cost_config,
            logger=logger,
        )

    return finalize_backtest_state(state)
