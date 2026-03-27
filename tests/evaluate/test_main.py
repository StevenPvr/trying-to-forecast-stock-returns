from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.evaluate.backtest import (
    ActiveTrade,
    BacktestState,
    XtbCostConfig,
    allocate_signal_candidates,
    build_daily_signal_candidates,
    finalize_backtest_state,
    finalize_trade,
    process_prediction_day,
)
from core.src.meta_model.evaluate.parameters import load_selected_xgboost_configuration
from core.src.meta_model.evaluate.training import build_available_training_frame


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def test_build_daily_signal_candidates_selects_top_and_bottom_one_percent() -> None:
    rows = [
        {"date": pd.Timestamp("2022-01-03"), "ticker": f"T{i:03d}", "prediction": float(i)}
        for i in range(100)
    ]
    scores = pd.DataFrame(rows)

    candidates = build_daily_signal_candidates(scores, top_fraction=0.01)

    assert len(candidates) == 2
    assert {candidate.ticker for candidate in candidates} == {"T000", "T099"}
    assert {candidate.side for candidate in candidates} == {"long", "short"}


def test_allocate_signal_candidates_caps_symbol_exposure_at_five_percent() -> None:
    active_trades = [
        ActiveTrade(
            ticker="AAA",
            side="long",
            entry_date=_ts("2022-01-03"),
            exit_date=_ts("2022-01-10"),
            notional=45_000.0,
            predicted_return=0.03,
            realized_log_return=0.0,
            signal_rank=1,
        ),
    ]
    signal_rows = pd.DataFrame([
        {"date": pd.Timestamp("2022-01-04"), "ticker": "AAA", "prediction": 0.04},
        {"date": pd.Timestamp("2022-01-04"), "ticker": "BBB", "prediction": -0.05},
    ])
    candidates = build_daily_signal_candidates(signal_rows, top_fraction=0.5)

    new_trades = allocate_signal_candidates(
        trade_date=_ts("2022-01-04"),
        candidates=candidates,
        active_trades=active_trades,
        current_equity=1_000_000.0,
        hold_period_days=5,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        unique_dates=pd.Index(pd.date_range("2022-01-03", periods=10, freq="B")),
    )

    aaa_trade = next(trade for trade in new_trades if trade.ticker == "AAA")
    assert aaa_trade.notional == 5_000.0
    assert abs(sum(trade.notional for trade in active_trades + [aaa_trade]) - 50_000.0) < 1e-9


def test_finalize_trade_applies_xtb_transaction_and_financing_costs() -> None:
    trade = ActiveTrade(
        ticker="AAA",
        side="long",
        entry_date=_ts("2022-01-03"),
        exit_date=_ts("2022-01-10"),
        notional=50_000.0,
        predicted_return=0.03,
        realized_log_return=0.09531017980432493,
        signal_rank=1,
        entry_transaction_cost_amount=150.0,
        accumulated_financing_cost_amount=56.725,
    )

    closed = finalize_trade(
        trade,
        cost_config=XtbCostConfig(),
    )

    assert round(closed.gross_return, 6) == 0.10
    assert round(closed.transaction_cost, 6) == 0.006
    assert round(closed.financing_cost, 6) == 0.001135
    assert round(closed.net_return, 6) == 0.092866
    assert round(closed.exit_cash_flow_amount, 6) == 4850.0


def test_load_selected_xgboost_configuration_prefers_one_standard_error_trial(
    tmp_path: Path,
) -> None:
    best_params_path = tmp_path / "best.json"
    trials_path = tmp_path / "trials.parquet"

    best_params_path.write_text(
        json.dumps({
            "config": {"boost_rounds": 3000},
            "selected_trial_one_standard_error": {
                "trial_number": 7,
                "params": {
                    "eta": 0.05,
                    "max_depth": 4,
                },
            },
        }),
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "trial_number": 7,
            "fold_1_best_iteration": 100,
            "fold_2_best_iteration": 110,
            "fold_3_best_iteration": 120,
            "fold_4_best_iteration": 130,
            "fold_5_best_iteration": 140,
        },
    ]).to_parquet(trials_path, index=False)

    configuration = load_selected_xgboost_configuration(best_params_path, trials_path)

    assert configuration.selected_trial_number == 7
    assert configuration.params["eta"] == 0.05
    assert configuration.training_rounds == 127


def test_build_available_training_frame_uses_only_labels_realized_by_prediction_date() -> None:
    dates = pd.date_range("2022-01-03", periods=12, freq="B")
    frame = pd.DataFrame([
        {
            "date": date,
            "ticker": "AAA",
            "dataset_split": "test" if idx >= 7 else "train",
            "target_main": 0.01,
            "feature_a": float(idx),
        }
        for idx, date in enumerate(dates)
    ])

    training_frame = build_available_training_frame(
        frame,
        prediction_date=_ts("2022-01-17"),
        hold_period_days=5,
    )

    assert training_frame["date"].max() == _ts("2022-01-10")
    assert _ts("2022-01-11") not in set(training_frame["date"])


def test_process_prediction_day_updates_daily_equity_immediately() -> None:
    state = BacktestState()
    day_one_predictions = pd.DataFrame([
        {"date": _ts("2022-01-03"), "ticker": f"L{i}", "prediction": float(100 + i), "target_main": 0.09531017980432493}
        for i in range(50)
    ] + [
        {"date": _ts("2022-01-03"), "ticker": f"S{i}", "prediction": float(i), "target_main": -0.09531017980432493}
        for i in range(50)
    ])
    day_two_predictions = pd.DataFrame([
        {"date": _ts("2022-01-10"), "ticker": f"L{i}", "prediction": float(100 + i), "target_main": 0.09531017980432493}
        for i in range(50)
    ] + [
        {"date": _ts("2022-01-10"), "ticker": f"S{i}", "prediction": float(i), "target_main": -0.09531017980432493}
        for i in range(50)
    ])
    unique_dates = pd.Index(pd.to_datetime(pd.Series([_ts("2022-01-03"), _ts("2022-01-10")])).sort_values())

    process_prediction_day(
        state=state,
        daily_predictions=day_one_predictions,
        unique_dates=unique_dates,
        top_fraction=0.01,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        hold_period_days=1,
        cost_config=XtbCostConfig(),
        logger=None,
    )
    process_prediction_day(
        state=state,
        daily_predictions=day_two_predictions,
        unique_dates=unique_dates,
        top_fraction=0.01,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        hold_period_days=1,
        cost_config=XtbCostConfig(),
        logger=None,
    )
    _, daily_frame, summary = finalize_backtest_state(state)

    assert len(daily_frame) == 2
    assert float(daily_frame.iloc[0]["total_return"]) < 0.0
    assert float(daily_frame.iloc[1]["total_return"]) > 0.0
    assert summary["final_equity"] > 1.0
