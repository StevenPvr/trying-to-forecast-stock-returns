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
from core.src.meta_model.model_registry.main import ModelSpec
from core.src.meta_model.model_contract import REALIZED_RETURN_COLUMN


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def _build_test_cost_config() -> XtbCostConfig:
    provider = BrokerSpecProvider(
        specs=(
            XtbInstrumentSpec(
                symbol="__default_stock__",
                instrument_group="stock_cfd",
                currency="USD",
                spread_bps=20.0,
                slippage_bps=5.0,
                long_swap_bps_daily=0.0,
                short_swap_bps_daily=0.0,
                margin_requirement=0.20,
                max_adv_participation=0.05,
                effective_from="2000-01-01",
            ),
            XtbInstrumentSpec(
                symbol="__default_index__",
                instrument_group="index_cfd",
                currency="USD",
                spread_bps=8.0,
                slippage_bps=2.0,
                long_swap_bps_daily=0.0,
                short_swap_bps_daily=0.0,
                margin_requirement=0.05,
                max_adv_participation=0.20,
                effective_from="2000-01-01",
            ),
        ),
    )
    return XtbCostConfig(fx_conversion_bps=0.0, broker_spec_provider=provider)


class _FakeBooster:
    def predict(self, matrix: object) -> list[float]:
        return [0.25]


def test_build_daily_signal_candidates_selects_top_and_bottom_one_percent() -> None:
    rows = [
        {"date": pd.Timestamp("2022-01-03"), "ticker": f"T{i:03d}", "prediction": float(i)}
        for i in range(100)
    ]
    scores = pd.DataFrame(rows)

    candidates = build_daily_signal_candidates(
        scores,
        top_fraction=0.01,
        cost_config=XtbCostConfig(),
        expected_holding_days=0,
    )

    assert len(candidates) == 2
    assert {candidate.ticker for candidate in candidates} == {"T000", "T099"}
    assert {candidate.side for candidate in candidates} == {"long", "short"}


def test_build_daily_signal_candidates_long_only_keeps_top_longs() -> None:
    rows = [
        {"date": pd.Timestamp("2022-01-03"), "ticker": f"T{i:03d}", "prediction": float(i)}
        for i in range(100)
    ]
    scores = pd.DataFrame(rows)

    candidates = build_daily_signal_candidates(
        scores,
        top_fraction=0.01,
        cost_config=XtbCostConfig(),
        expected_holding_days=0,
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


def test_validate_backtest_config_rejects_invalid_realism_requirements() -> None:
    with pytest.raises(ValueError, match="execution_lag_days"):
        validate_backtest_config(
            BacktestConfig(
                execution_lag_days=-1,
                benchmark_mode="universe_equal_weight",
                neutrality_mode="sector_beta_neutral",
            ),
        )

    validate_backtest_config(
        BacktestConfig(
            execution_lag_days=0,
            benchmark_mode="universe_equal_weight",
            neutrality_mode="sector_beta_neutral",
        ),
    )

    with pytest.raises(ValueError, match="benchmark_mode"):
        validate_backtest_config(
            BacktestConfig(
                benchmark_mode="",
                neutrality_mode="sector_beta_neutral",
            ),
        )

    with pytest.raises(ValueError, match="neutrality_mode"):
        validate_backtest_config(
            BacktestConfig(
                benchmark_mode="universe_equal_weight",
                neutrality_mode="",
            ),
        )


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
    candidates = build_daily_signal_candidates(
        signal_rows,
        top_fraction=0.5,
        cost_config=XtbCostConfig(),
        expected_holding_days=0,
    )

    new_trades = allocate_signal_candidates(
        trade_date=_ts("2022-01-04"),
        candidates=candidates,
        active_trades=active_trades,
        current_equity=1_000_000.0,
        hold_period_days=5,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        adv_participation_limit=0.05,
        neutrality_mode="dollar_neutral",
        open_hurdle_bps=0.0,
        apply_prediction_hurdle=False,
        unique_dates=pd.Index(pd.date_range("2022-01-03", periods=10, freq="B")),
    )

    aaa_trade = next(trade for trade in new_trades if trade.ticker == "AAA")
    assert aaa_trade.notional == 5_000.0
    assert abs(sum(trade.notional for trade in active_trades + [aaa_trade]) - 50_000.0) < 1e-9


def test_allocate_signal_candidates_respects_adv_cap_and_sector_neutrality() -> None:
    candidates = [
        build_daily_signal_candidates(
            pd.DataFrame([
                {
                    "date": _ts("2022-01-04"),
                    "ticker": "AAA",
                    "prediction": 0.8,
                    "company_sector": "Tech",
                    "company_beta": 1.2,
                    "stock_open_price": 100.0,
                    "stock_trading_volume": 1_000.0,
                    REALIZED_RETURN_COLUMN: 0.02,
                },
                {
                    "date": _ts("2022-01-04"),
                    "ticker": "BBB",
                    "prediction": 0.7,
                    "company_sector": "Health",
                    "company_beta": 0.9,
                    "stock_open_price": 100.0,
                    "stock_trading_volume": 50_000.0,
                    REALIZED_RETURN_COLUMN: 0.02,
                },
                {
                    "date": _ts("2022-01-04"),
                    "ticker": "CCC",
                    "prediction": -0.7,
                    "company_sector": "Tech",
                    "company_beta": 1.1,
                    "stock_open_price": 100.0,
                    "stock_trading_volume": 50_000.0,
                    REALIZED_RETURN_COLUMN: -0.02,
                },
                {
                    "date": _ts("2022-01-04"),
                    "ticker": "DDD",
                    "prediction": -0.8,
                    "company_sector": "Energy",
                    "company_beta": 1.3,
                    "stock_open_price": 100.0,
                    "stock_trading_volume": 50_000.0,
                    REALIZED_RETURN_COLUMN: -0.02,
                },
            ]),
            top_fraction=0.5,
            cost_config=XtbCostConfig(),
            expected_holding_days=0,
        )
    ][0]

    new_trades = allocate_signal_candidates(
        trade_date=_ts("2022-01-04"),
        candidates=candidates,
        active_trades=[],
        current_equity=1_000_000.0,
        hold_period_days=5,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        adv_participation_limit=0.05,
        neutrality_mode="sector_neutral",
        open_hurdle_bps=0.0,
        apply_prediction_hurdle=False,
        unique_dates=pd.Index(pd.date_range("2022-01-03", periods=10, freq="B")),
    )

    assert {trade.ticker for trade in new_trades} == {"AAA", "CCC"}
    aaa_trade = next(trade for trade in new_trades if trade.ticker == "AAA")
    ccc_trade = next(trade for trade in new_trades if trade.ticker == "CCC")
    assert aaa_trade.notional == 5_000.0
    assert ccc_trade.notional == 50_000.0


def test_allocate_signal_candidates_can_disable_prediction_magnitude_hurdle_for_rank_scores() -> None:
    candidates = build_daily_signal_candidates(
        pd.DataFrame([
            {
                "date": _ts("2022-01-04"),
                "ticker": "AAA",
                "prediction": 0.0015,
                "company_sector": "Tech",
                "company_beta": 1.0,
                "stock_open_price": 100.0,
                "stock_trading_volume": 1_000_000.0,
                REALIZED_RETURN_COLUMN: 0.02,
            },
            {
                "date": _ts("2022-01-04"),
                "ticker": "BBB",
                "prediction": -0.0015,
                "company_sector": "Tech",
                "company_beta": 1.0,
                "stock_open_price": 100.0,
                "stock_trading_volume": 1_000_000.0,
                REALIZED_RETURN_COLUMN: -0.02,
            },
        ]),
        top_fraction=0.5,
        cost_config=_build_test_cost_config(),
        expected_holding_days=0,
    )

    blocked_trades = allocate_signal_candidates(
        trade_date=_ts("2022-01-04"),
        candidates=candidates,
        active_trades=[],
        current_equity=1_000_000.0,
        hold_period_days=0,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        adv_participation_limit=0.05,
        neutrality_mode="dollar_neutral",
        open_hurdle_bps=12.0,
        apply_prediction_hurdle=True,
        unique_dates=pd.Index(pd.date_range("2022-01-04", periods=1, freq="B")),
    )
    allowed_trades = allocate_signal_candidates(
        trade_date=_ts("2022-01-04"),
        candidates=candidates,
        active_trades=[],
        current_equity=1_000_000.0,
        hold_period_days=0,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        adv_participation_limit=0.05,
        neutrality_mode="dollar_neutral",
        open_hurdle_bps=12.0,
        apply_prediction_hurdle=False,
        unique_dates=pd.Index(pd.date_range("2022-01-04", periods=1, freq="B")),
    )

    assert blocked_trades == []
    assert len(allowed_trades) == 2


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
        entry_transaction_cost_amount=100.0,
        accumulated_financing_cost_amount=25.0,
        expected_exit_cost_rate=0.002,
        expected_financing_cost_rate=0.0005,
    )

    closed = finalize_trade(
        trade,
        cost_config=XtbCostConfig(),
    )

    assert round(closed.gross_return, 6) == 0.10
    assert round(closed.transaction_cost, 6) == 0.004
    assert round(closed.financing_cost, 6) == 0.0005
    assert round(closed.net_return, 6) == 0.0955
    assert round(closed.exit_cash_flow_amount, 6) == 4900.0


def test_finalize_trade_exposes_trade_cashflow_breakdown() -> None:
    trade = ActiveTrade(
        ticker="AAA",
        side="long",
        entry_date=_ts("2022-01-03"),
        exit_date=_ts("2022-01-03"),
        notional=50_000.0,
        predicted_return=0.03,
        realized_log_return=0.09531017980432493,
        signal_rank=1,
        entry_transaction_cost_amount=75.0,
        accumulated_financing_cost_amount=10.0,
        expected_exit_cost_rate=0.0015,
    )

    closed = finalize_trade(
        trade,
        cost_config=_build_test_cost_config(),
    )

    assert closed.gross_pnl_amount == pytest.approx(5_000.0)
    assert closed.entry_transaction_cost_amount == pytest.approx(75.0)
    assert closed.exit_transaction_cost_amount == pytest.approx(75.0)
    assert closed.total_transaction_cost_amount == pytest.approx(150.0)
    assert closed.financing_cost_amount == pytest.approx(10.0)
    assert closed.net_pnl_amount == pytest.approx(4_840.0)
    assert closed.pnl_amount == pytest.approx(4_840.0)
    assert closed.exit_cash_flow_amount == pytest.approx(4_925.0)


def test_process_prediction_day_records_daily_reconciliation_breakdown() -> None:
    state = BacktestState(initial_equity=1_000_000.0, current_equity=1_000_000.0)
    daily_predictions = pd.DataFrame([
        {
            "date": _ts("2022-01-03"),
            "ticker": "AAA",
            "prediction": 0.9,
            "stock_open_price": 100.0,
            "stock_trading_volume": 1_000_000.0,
            REALIZED_RETURN_COLUMN: 0.09531017980432493,
        },
        {
            "date": _ts("2022-01-03"),
            "ticker": "BBB",
            "prediction": -0.9,
            "stock_open_price": 100.0,
            "stock_trading_volume": 1_000_000.0,
            REALIZED_RETURN_COLUMN: -0.10536051565782628,
        },
    ])

    process_prediction_day(
        state=state,
        daily_predictions=daily_predictions,
        unique_dates=pd.Index([_ts("2022-01-03")]),
        top_fraction=0.5,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        adv_participation_limit=0.05,
        neutrality_mode="dollar_neutral",
        open_hurdle_bps=0.0,
        apply_prediction_hurdle=False,
        hold_period_days=0,
        cost_config=_build_test_cost_config(),
    )

    daily_row = state.daily_rows[0]

    assert len(state.closed_trades) == 2
    assert state.current_equity == pytest.approx(1_009_700.0)
    assert daily_row["starting_equity"] == pytest.approx(1_000_000.0)
    assert daily_row["gross_pnl_exits"] == pytest.approx(10_000.0)
    assert daily_row["entry_cost_amount"] == pytest.approx(150.0)
    assert daily_row["exit_cost_amount"] == pytest.approx(150.0)
    assert daily_row["financing_amount"] == pytest.approx(0.0)
    assert daily_row["net_cash_flow"] == pytest.approx(9_700.0)
    assert daily_row["ending_equity"] == pytest.approx(1_009_700.0)
    assert daily_row["opened_notional"] == pytest.approx(100_000.0)
    assert daily_row["closed_notional"] == pytest.approx(100_000.0)
    assert daily_row["active_notional_end"] == pytest.approx(0.0)


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


def test_process_prediction_day_updates_daily_equity_immediately() -> None:
    state = BacktestState()
    day_one_predictions = pd.DataFrame([
        {"date": _ts("2022-01-03"), "ticker": f"L{i}", "prediction": float(100 + i), REALIZED_RETURN_COLUMN: 0.13976194237515863, "company_sector": "Tech", "company_beta": 1.0, "stock_open_price": 100.0, "stock_trading_volume": 50_000.0}
        for i in range(50)
    ] + [
        {"date": _ts("2022-01-03"), "ticker": f"S{i}", "prediction": float(i), REALIZED_RETURN_COLUMN: -0.13976194237515863, "company_sector": "Tech", "company_beta": 1.0, "stock_open_price": 100.0, "stock_trading_volume": 50_000.0}
        for i in range(50)
    ])
    day_two_predictions = pd.DataFrame([
        {"date": _ts("2022-01-10"), "ticker": f"L{i}", "prediction": float(100 + i), REALIZED_RETURN_COLUMN: 0.13976194237515863, "company_sector": "Tech", "company_beta": 1.0, "stock_open_price": 100.0, "stock_trading_volume": 50_000.0}
        for i in range(50)
    ] + [
        {"date": _ts("2022-01-10"), "ticker": f"S{i}", "prediction": float(i), REALIZED_RETURN_COLUMN: -0.13976194237515863, "company_sector": "Tech", "company_beta": 1.0, "stock_open_price": 100.0, "stock_trading_volume": 50_000.0}
        for i in range(50)
    ])
    unique_dates = pd.Index(
        cast(
            pd.Series,
            pd.to_datetime(pd.Series([_ts("2022-01-03"), _ts("2022-01-10")])),
        ).sort_values(),
    )

    process_prediction_day(
        state=state,
        daily_predictions=day_one_predictions,
        unique_dates=unique_dates,
        top_fraction=0.01,
        allocation_fraction=0.05,
        action_cap_fraction=0.05,
        gross_cap_fraction=1.0,
        adv_participation_limit=0.05,
        neutrality_mode="sector_beta_neutral",
        open_hurdle_bps=0.0,
        apply_prediction_hurdle=False,
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
        adv_participation_limit=0.05,
        neutrality_mode="sector_beta_neutral",
        open_hurdle_bps=0.0,
        apply_prediction_hurdle=False,
        hold_period_days=1,
        cost_config=XtbCostConfig(),
        logger=None,
    )
    _, daily_frame, summary = finalize_backtest_state(state)

    assert len(daily_frame) == 2
    assert float(daily_frame.iloc[0]["total_return"]) < 0.0
    assert float(daily_frame.iloc[1]["total_return"]) > 0.0
    assert summary["final_equity"] > 1.0
    assert "benchmark_return" in daily_frame.columns
    assert "turnover" in daily_frame.columns
    assert "gross_exposure" in daily_frame.columns
    assert "net_exposure" in daily_frame.columns
    assert "alpha_over_benchmark_net" in summary
    assert "turnover_annualized" in summary
    assert "calmar_ratio" in summary
    assert "capacity_binding_share" in summary
    assert "margin_headroom" in summary


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
