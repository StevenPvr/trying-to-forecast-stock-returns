# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.broker_xtb.specs import BrokerSpecProvider, XtbInstrumentSpec
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
from core.src.meta_model.evaluate.config import BacktestConfig, validate_backtest_config
from core.src.meta_model.evaluate.main import _select_evaluate_model_specs
from core.src.meta_model.evaluate.parameters import load_selected_xgboost_configuration
from core.src.meta_model.evaluate.training import build_available_training_frame, predict_test_frame
from core.src.meta_model.evaluate.training import resolve_training_threads
from core.src.meta_model.model_contract import REALIZED_RETURN_COLUMN
from core.src.meta_model.model_registry.main import ModelSpec


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def _build_test_cost_config(*, account_currency: str = "EUR") -> XtbCostConfig:
    provider = BrokerSpecProvider(
        specs=(
            XtbInstrumentSpec(
                symbol="__default_stock__",
                instrument_group="stock_cash",
                currency="USD",
                spread_bps=0.0,
                slippage_bps=0.0,
                long_swap_bps_daily=0.0,
                short_swap_bps_daily=0.0,
                margin_requirement=1.0,
                max_adv_participation=0.05,
                effective_from="2000-01-01",
                fx_conversion_bps=0.0,
            ),
        ),
    )
    return XtbCostConfig(
        account_currency=account_currency,
        broker_spec_provider=provider,
    )


class _FakeBooster:
    def predict(self, matrix: object) -> list[float]:
        return [0.25]


def test_build_daily_signal_candidates_keeps_only_top_longs_for_cash_equities() -> None:
    rows = [
        {"date": pd.Timestamp("2022-01-03"), "ticker": f"T{i:03d}", "prediction": float(i)}
        for i in range(100)
    ]
    scores = pd.DataFrame(rows)

    candidates = build_daily_signal_candidates(
        scores,
        top_fraction=0.01,
        cost_config=XtbCostConfig(),
        expected_holding_days=1,
        neutrality_mode="long_only",
    )

    assert len(candidates) == 1
    assert candidates[0].ticker == "T099"
    assert candidates[0].side == "long"


def test_select_evaluate_model_specs_keeps_only_xgboost() -> None:
    model_specs = [
        ModelSpec(model_name="ridge"),
        ModelSpec(model_name="xgboost", params={"eta": 0.1}, training_rounds=7),
        ModelSpec(model_name="lightgbm"),
    ]

    selected_specs = _select_evaluate_model_specs(model_specs)

    assert [model_spec.model_name for model_spec in selected_specs] == ["xgboost"]


def test_validate_backtest_config_rejects_invalid_cash_requirements() -> None:
    with pytest.raises(ValueError, match="execution_lag_days"):
        validate_backtest_config(BacktestConfig(execution_lag_days=-1))

    with pytest.raises(ValueError, match="neutrality_mode must remain long_only"):
        validate_backtest_config(BacktestConfig(neutrality_mode="sector_neutral"))

    with pytest.raises(ValueError, match="starting_cash_eur"):
        validate_backtest_config(BacktestConfig(starting_cash_eur=0.0))

    validate_backtest_config(BacktestConfig())


def test_predict_test_frame_keeps_signal_date_before_execution_date() -> None:
    test_frame = pd.DataFrame([
        {
            "date": _ts("2022-01-03"),
            "ticker": "AAA",
            "feature_a": 1.0,
        },
    ])

    with patch(
        "core.src.meta_model.evaluate.training.load_xgboost_module",
        return_value=SimpleNamespace(
            DMatrix=lambda data, feature_names=None: SimpleNamespace(),  # noqa: ARG005
        ),
    ):
        predicted = predict_test_frame(
            _FakeBooster(),
            test_frame,
            ["feature_a"],
            execution_date=_ts("2022-01-04"),
        )

    assert predicted.loc[0, "signal_date"] == _ts("2022-01-03")
    assert predicted.loc[0, "date"] == _ts("2022-01-04")
    assert predicted.loc[0, "date"] > predicted.loc[0, "signal_date"]


def test_resolve_training_threads_uses_all_detected_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.src.meta_model.runtime_parallelism.os.cpu_count",
        lambda: 10,
    )

    assert resolve_training_threads() == 10


def test_allocate_signal_candidates_respects_symbol_cap_cash_and_minimum_order() -> None:
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
            spec=_build_test_cost_config().broker_spec_provider.resolve("AAA", _ts("2022-01-03")),
        ),
    ]
    signal_rows = pd.DataFrame([
        {"date": pd.Timestamp("2022-01-04"), "ticker": "AAA", "prediction": 0.04},
        {"date": pd.Timestamp("2022-01-04"), "ticker": "BBB", "prediction": 0.05},
    ])
    candidates = build_daily_signal_candidates(
        signal_rows,
        top_fraction=0.5,
        cost_config=_build_test_cost_config(),
        expected_holding_days=1,
        neutrality_mode="long_only",
    )

    new_trades = allocate_signal_candidates(
        trade_date=_ts("2022-01-04"),
        candidates=candidates,
        active_trades=active_trades,
        current_equity=1_000_000.0,
        cash_balance=6_000.0,
        hold_period_days=5,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        adv_participation_limit=0.05,
        neutrality_mode="long_only",
        open_hurdle_bps=0.0,
        apply_prediction_hurdle=False,
        unique_dates=pd.Index(pd.date_range("2022-01-03", periods=10, freq="B")),
        month_to_date_turnover_eur=0.0,
        account_currency="EUR",
    )

    assert len(new_trades) == 1
    assert new_trades[0].ticker == "BBB"
    assert new_trades[0].notional == pytest.approx(6_000.0)


def test_allocate_signal_candidates_can_disable_prediction_hurdle_for_rank_scores() -> None:
    provider = BrokerSpecProvider(
        specs=(
            XtbInstrumentSpec(
                symbol="__default_stock__",
                instrument_group="stock_cash",
                currency="USD",
                spread_bps=0.0,
                slippage_bps=0.0,
                long_swap_bps_daily=0.0,
                short_swap_bps_daily=0.0,
                margin_requirement=1.0,
                max_adv_participation=0.05,
                effective_from="2000-01-01",
                fx_conversion_bps=50.0,
            ),
        ),
    )
    cost_config = XtbCostConfig(account_currency="EUR", broker_spec_provider=provider)
    candidates = build_daily_signal_candidates(
        pd.DataFrame([
            {
                "date": _ts("2022-01-04"),
                "ticker": "AAA",
                "prediction": 0.001,
                "stock_open_price": 100.0,
                "stock_trading_volume": 1_000_000.0,
                REALIZED_RETURN_COLUMN: 0.02,
            },
        ]),
        top_fraction=1.0,
        cost_config=cost_config,
        expected_holding_days=1,
        neutrality_mode="long_only",
    )

    blocked_trades = allocate_signal_candidates(
        trade_date=_ts("2022-01-04"),
        candidates=candidates,
        active_trades=[],
        current_equity=100_000.0,
        cash_balance=100_000.0,
        hold_period_days=1,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        adv_participation_limit=0.05,
        neutrality_mode="long_only",
        open_hurdle_bps=12.0,
        apply_prediction_hurdle=True,
        unique_dates=pd.Index(pd.date_range("2022-01-04", periods=2, freq="B")),
        month_to_date_turnover_eur=100_000.0,
        account_currency="EUR",
    )
    allowed_trades = allocate_signal_candidates(
        trade_date=_ts("2022-01-04"),
        candidates=candidates,
        active_trades=[],
        current_equity=100_000.0,
        cash_balance=100_000.0,
        hold_period_days=1,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        adv_participation_limit=0.05,
        neutrality_mode="long_only",
        open_hurdle_bps=12.0,
        apply_prediction_hurdle=False,
        unique_dates=pd.Index(pd.date_range("2022-01-04", periods=2, freq="B")),
        month_to_date_turnover_eur=100_000.0,
        account_currency="EUR",
    )

    assert blocked_trades == []
    assert len(allowed_trades) == 1


def test_finalize_trade_applies_cash_equity_entry_and_exit_costs() -> None:
    spec = _build_test_cost_config().broker_spec_provider.resolve("AAA", _ts("2022-01-03"))
    trade = ActiveTrade(
        ticker="AAA",
        side="long",
        entry_date=_ts("2022-01-03"),
        exit_date=_ts("2022-01-10"),
        notional=50_000.0,
        predicted_return=0.03,
        realized_log_return=0.09531017980432493,
        signal_rank=1,
        spec=spec,
        entry_transaction_cost_amount=100.0,
        entry_commission_amount=100.0,
        entry_fx_conversion_amount=0.0,
        expected_entry_cost_rate=0.002,
    )

    closed = finalize_trade(
        trade,
        exit_cost_estimate=SimpleNamespace(
            total_cost_amount_eur=100.0,
            commission_amount_eur=100.0,
            fx_conversion_amount_eur=0.0,
        ),
    )

    assert round(closed.gross_return, 6) == 0.10
    assert round(closed.transaction_cost, 6) == 0.004
    assert round(closed.net_return, 6) == 0.096
    assert round(closed.exit_cash_flow_amount, 6) == 54900.0


def test_process_prediction_day_tracks_cash_and_monthly_turnover() -> None:
    provider = BrokerSpecProvider(
        specs=(
            XtbInstrumentSpec(
                symbol="__default_stock__",
                instrument_group="stock_cash",
                currency="EUR",
                spread_bps=0.0,
                slippage_bps=0.0,
                long_swap_bps_daily=0.0,
                short_swap_bps_daily=0.0,
                margin_requirement=1.0,
                max_adv_participation=0.05,
                effective_from="2000-01-01",
                fx_conversion_bps=0.0,
            ),
        ),
    )
    cost_config = XtbCostConfig(account_currency="EUR", broker_spec_provider=provider)
    state = BacktestState(
        initial_equity=100_000.0,
        current_equity=100_000.0,
        cash_balance=100_000.0,
    )
    daily_predictions = pd.DataFrame([
        {
            "date": _ts("2022-01-03"),
            "ticker": "AAA",
            "prediction": 0.9,
            "stock_open_price": 100.0,
            "stock_trading_volume": 1_000_000.0,
            REALIZED_RETURN_COLUMN: 0.09531017980432493,
        },
    ])
    unique_dates = pd.Index([_ts("2022-01-03"), _ts("2022-01-04")])

    process_prediction_day(
        state=state,
        daily_predictions=daily_predictions,
        unique_dates=unique_dates,
        top_fraction=1.0,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        adv_participation_limit=0.05,
        neutrality_mode="long_only",
        open_hurdle_bps=0.0,
        apply_prediction_hurdle=False,
        hold_period_days=1,
        cost_config=cost_config,
    )

    daily_row = state.daily_rows[0]
    assert daily_row["opened_notional"] == pytest.approx(5_000.0)
    assert daily_row["entry_cost_amount"] == pytest.approx(0.0)
    assert daily_row["cash_balance"] == pytest.approx(95_000.0)
    assert daily_row["active_notional_end"] == pytest.approx(5_000.0)
    assert daily_row["month_to_date_turnover_eur"] == pytest.approx(5_000.0)
    assert state.current_equity == pytest.approx(100_000.0)


def test_load_selected_xgboost_configuration_prefers_best_trial(
    tmp_path: Path,
) -> None:
    best_params_path = tmp_path / "best.json"
    trials_path = tmp_path / "trials.parquet"

    best_params_path.write_text(
        json.dumps({
            "best_trial_number": 133,
            "params": {
                "eta": 0.025,
                "max_depth": 2,
            },
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
            "trial_number": 133,
            "fold_1_best_iteration": 10,
            "fold_2_best_iteration": 20,
            "fold_3_best_iteration": 30,
            "fold_4_best_iteration": 40,
            "fold_5_best_iteration": 50,
        },
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

    assert configuration.selected_trial_number == 133
    assert configuration.params["eta"] == 0.025
    assert configuration.training_rounds == 37


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
        for idx, date in enumerate(cast(pd.DatetimeIndex, dates))
    ])

    training_frame = build_available_training_frame(
        frame,
        prediction_date=_ts("2022-01-17"),
        label_embargo_days=5,
    )

    assert training_frame["date"].max() == _ts("2022-01-10")
    assert _ts("2022-01-11") not in set(cast(pd.Series, training_frame["date"]).tolist())


def test_finalize_backtest_state_reports_cash_equity_metrics() -> None:
    state = BacktestState(
        initial_equity=100_000.0,
        current_equity=101_000.0,
        cash_balance=101_000.0,
        daily_rows=[
            {
                "date": _ts("2022-01-03"),
                "equity": 100_500.0,
                "realized_return": 0.005,
                "benchmark_return": 0.001,
                "gross_pnl_exits": 600.0,
                "entry_cost_amount": 50.0,
                "exit_cost_amount": 25.0,
                "custody_fee_amount": 0.0,
                "turnover": 0.05,
                "gross_exposure": 0.05,
                "capacity_binding_share": 0.0,
                "reconciliation_error": 0.0,
            },
            {
                "date": _ts("2022-01-04"),
                "equity": 101_000.0,
                "realized_return": 0.004975124378109453,
                "benchmark_return": 0.001,
                "gross_pnl_exits": 500.0,
                "entry_cost_amount": 50.0,
                "exit_cost_amount": 25.0,
                "custody_fee_amount": 0.0,
                "turnover": 0.05,
                "gross_exposure": 0.04,
                "capacity_binding_share": 0.0,
                "reconciliation_error": 0.0,
            },
        ],
    )

    _, daily_frame, summary = finalize_backtest_state(state)

    assert len(daily_frame) == 2
    assert summary["final_equity"] == pytest.approx(101_000.0)
    assert summary["transaction_cost_amount_total"] == pytest.approx(150.0)
    assert "capacity_binding_share" in summary
    assert "average_gross_exposure" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
