# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
import logging
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
    BacktestProgressConfig,
    BacktestRunConfig,
    BacktestRuntimeConfig,
    BacktestState,
    OptimizerAllocationRequest,
    PortfolioOptimizerArtifacts,
    XtbCostConfig,
    allocate_signal_candidates_optimizer_miqp,
    finalize_backtest_state,
    finalize_trade,
    process_prediction_day,
    run_signal_backtest_with_diagnostics,
)
from core.src.meta_model.evaluate.config import (
    STARTING_CASH_EUR,
    BacktestConfig,
    validate_backtest_config,
)
from core.src.meta_model.evaluate import backtest as evaluate_backtest
from core.src.meta_model.evaluate.main import _select_evaluate_model_specs
from core.src.meta_model.evaluate.parameters import load_selected_xgboost_configuration
from core.src.meta_model.evaluate.training import build_available_training_frame, predict_test_frame
from core.src.meta_model.evaluate.training import iter_model_prediction_days_frozen_train_only
from core.src.meta_model.evaluate.training import resolve_training_threads
from core.src.meta_model.model_contract import REALIZED_RETURN_COLUMN
from core.src.meta_model.model_registry.main import ModelSpec
from core.src.meta_model.meta_labeling.calibration import FittedProbabilityCalibrator
from core.src.meta_model.meta_labeling.model import MetaModelArtifact
from core.src.meta_model.evaluate import main as evaluate_main


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def _build_test_cost_config(*, account_currency: str = "EUR") -> XtbCostConfig:
    provider = BrokerSpecProvider(
        specs=(
            XtbInstrumentSpec(
                symbol="AAA",
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
            XtbInstrumentSpec(
                symbol="BBB",
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
        fallback_to_defaults=False,
    )
    return XtbCostConfig(
        account_currency=account_currency,
        broker_spec_provider=provider,
    )


def _build_test_optimizer_artifacts() -> PortfolioOptimizerArtifacts:
    covariance = pd.DataFrame(
        [[0.01]],
        index=["AAA"],
        columns=["AAA"],
    )
    return PortfolioOptimizerArtifacts(covariance=covariance)


def _build_test_runtime(
    *,
    unique_dates: pd.Index,
    cost_config: XtbCostConfig | None = None,
    optimizer_artifacts: PortfolioOptimizerArtifacts | None = None,
    portfolio_construction_mode: str = "optimizer_miqp",
) -> BacktestRuntimeConfig:
    return BacktestRuntimeConfig(
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
        cost_config=cost_config or _build_test_cost_config(),
        portfolio_construction_mode=portfolio_construction_mode,
        optimizer_artifacts=optimizer_artifacts,
        lambda_risk=0.0,
        lambda_turnover=0.0,
        lambda_cost=0.0,
        max_position_weight=0.02,
        max_sector_weight=0.20,
        min_target_weight=0.0,
        no_trade_buffer_bps=0.0,
        miqp_time_limit_seconds=2.0,
        miqp_relative_gap=0.0,
        miqp_candidate_pool_size=10,
        miqp_primary_objective_tolerance_bps=0.0,
    )


class _FakeBooster:
    def predict(self, matrix: object) -> list[float]:
        return [0.25]


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


def test_validate_backtest_config_rejects_non_miqp_modes() -> None:
    with pytest.raises(ValueError, match="portfolio_construction_mode"):
        validate_backtest_config(BacktestConfig(portfolio_construction_mode="invalid_mode"))

    with pytest.raises(ValueError, match="solver_backend"):
        validate_backtest_config(BacktestConfig(solver_backend="cvxpy_osqp"))


def test_backtest_config_defaults_to_realistic_starting_cash() -> None:
    assert STARTING_CASH_EUR == pytest.approx(10_000.0)
    assert BacktestConfig().starting_cash_eur == pytest.approx(10_000.0)


def test_backtest_config_defaults_to_miqp_only_mode() -> None:
    config = BacktestConfig()
    assert config.portfolio_construction_mode == "optimizer_miqp"
    assert config.solver_backend == "scip_miqp"


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


def test_apply_meta_refinement_overwrites_prediction_and_expected_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = MetaModelArtifact(
        feature_names=["feature_a", "primary_prediction"],
        params={},
        training_rounds=10,
        fitted_object=object(),
    )
    calibrator = FittedProbabilityCalibrator(
        x_thresholds=[0.0, 1.0],
        y_thresholds=[0.0, 1.0],
    )
    predicted_day = pd.DataFrame([
        {
            "date": _ts("2022-01-04"),
            "ticker": "AAA",
            "feature_a": 1.0,
            "prediction": 0.4,
            "expected_return_5d": 0.1,
        },
        {
            "date": _ts("2022-01-04"),
            "ticker": "BBB",
            "feature_a": 2.0,
            "prediction": -0.3,
            "expected_return_5d": -0.06,
        },
    ])

    monkeypatch.setattr(
        evaluate_main,
        "predict_meta_model",
        lambda loaded_artifact, frame: [0.9, 0.4],
    )

    refined = evaluate_main._apply_meta_refinement_to_prediction_day(
        predicted_day,
        meta_artifact=artifact,
        meta_probability_calibrator=calibrator,
    )

    assert refined["primary_prediction"].tolist() == pytest.approx([0.4, -0.3])
    assert refined["prediction"].tolist() == pytest.approx([0.36, 0.0])
    assert refined["expected_return_5d"].tolist() == pytest.approx([0.09, 0.0])


def test_resolve_training_threads_uses_all_detected_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.src.meta_model.runtime_parallelism.os.cpu_count",
        lambda: 10,
    )

    assert resolve_training_threads() == 10


def test_run_signal_backtest_with_diagnostics_logs_progress(
    caplog: pytest.LogCaptureFixture,
) -> None:
    predictions = pd.DataFrame([
        {
            "date": _ts("2022-01-03"),
            "ticker": "AAA",
            "prediction": 0.1,
            REALIZED_RETURN_COLUMN: 0.01,
            "hl_context_stock_close_price": 100.0,
        },
        {
            "date": _ts("2022-01-04"),
            "ticker": "AAA",
            "prediction": 0.1,
            REALIZED_RETURN_COLUMN: 0.01,
            "hl_context_stock_close_price": 101.0,
        },
        {
            "date": _ts("2022-01-05"),
            "ticker": "AAA",
            "prediction": 0.1,
            REALIZED_RETURN_COLUMN: 0.01,
            "hl_context_stock_close_price": 102.0,
        },
    ])
    caplog.set_level(logging.INFO)
    run_config = BacktestRunConfig(
        runtime=_build_test_runtime(
            unique_dates=pd.Index([]),
            cost_config=_build_test_cost_config(),
            optimizer_artifacts=_build_test_optimizer_artifacts(),
        ),
        starting_cash_eur=100_000.0,
    )
    run_signal_backtest_with_diagnostics(
        predictions,
        run_config,
        progress=BacktestProgressConfig(
            logger=logging.getLogger("test-backtest-progress"),
            progress_label="test backtest",
            progress_log_every=2,
        ),
    )
    assert "test backtest started" in caplog.text
    assert "test backtest progress" in caplog.text


def test_allocate_signal_candidates_optimizer_miqp_uses_integer_shares() -> None:
    provider = BrokerSpecProvider(
        specs=(
            XtbInstrumentSpec(
                symbol="AAA",
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
            XtbInstrumentSpec(
                symbol="BBB",
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
        fallback_to_defaults=False,
    )
    cost_config = XtbCostConfig(account_currency="EUR", broker_spec_provider=provider)
    predictions = pd.DataFrame([
        {
            "date": _ts("2022-01-04"),
            "ticker": "AAA",
            "prediction": 0.08,
            "expected_return_5d": 0.08,
            "stock_open_price": 600.0,
            "stock_trading_volume": 1_000_000.0,
            "company_sector": "Tech",
            REALIZED_RETURN_COLUMN: 0.02,
        },
        {
            "date": _ts("2022-01-04"),
            "ticker": "BBB",
            "prediction": 0.06,
            "expected_return_5d": 0.06,
            "stock_open_price": 400.0,
            "stock_trading_volume": 1_000_000.0,
            "company_sector": "Tech",
            REALIZED_RETURN_COLUMN: 0.01,
        },
    ])
    optimizer_artifacts = PortfolioOptimizerArtifacts(
        covariance=pd.DataFrame(
            [[0.01, 0.0], [0.0, 0.01]],
            index=["AAA", "BBB"],
            columns=["AAA", "BBB"],
        ),
    )
    new_trades, diagnostics, allocation_rows = allocate_signal_candidates_optimizer_miqp(
        OptimizerAllocationRequest(
            trade_date=_ts("2022-01-04"),
            daily_predictions=predictions,
            active_trades=[],
            current_equity=1_000.0,
            cash_balance=1_000.0,
            unique_dates=pd.Index(pd.date_range("2022-01-04", periods=2, freq="B")),
            month_to_date_turnover_eur=0.0,
            hold_period_days=1,
            gross_cap_fraction=1.0,
            adv_participation_limit=0.05,
            open_hurdle_bps=0.0,
            account_currency="EUR",
            cost_config=cost_config,
            artifacts=optimizer_artifacts,
            lambda_risk=0.0,
            lambda_turnover=0.0,
            lambda_cost=0.0,
            max_position_weight=1.0,
            max_sector_weight=1.0,
            min_target_weight=0.0,
            no_trade_buffer_bps=0.0,
            miqp_time_limit_seconds=2.0,
            miqp_relative_gap=0.0,
            miqp_candidate_pool_size=10,
            miqp_primary_objective_tolerance_bps=0.0,
        ),
    )

    assert diagnostics["solver_status"] in {"optimal", "timelimit", "gaplimit"}
    assert len(new_trades) == 2
    assert {trade.share_count for trade in new_trades} == {1}
    assert {trade.reference_price_eur for trade in new_trades} == {400.0, 600.0}
    assert len(allocation_rows) == 2
    assert {int(row["target_shares"]) for row in allocation_rows} == {1}


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
                symbol="AAA",
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
        fallback_to_defaults=False,
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
        state,
        daily_predictions,
        _build_test_runtime(
            unique_dates=unique_dates,
            cost_config=cost_config,
            optimizer_artifacts=_build_test_optimizer_artifacts(),
        ),
    )

    daily_row = state.daily_rows[0]
    assert daily_row["opened_notional"] == pytest.approx(2_000.0)
    assert daily_row["entry_cost_amount"] == pytest.approx(0.0)
    assert daily_row["cash_balance"] == pytest.approx(98_000.0)
    assert daily_row["active_notional_end"] == pytest.approx(2_000.0)
    assert daily_row["month_to_date_turnover_eur"] == pytest.approx(2_000.0)
    assert state.current_equity == pytest.approx(100_000.0)


def test_process_prediction_day_rejects_non_miqp_mode() -> None:
    state = BacktestState(
        initial_equity=10_000.0,
        current_equity=10_000.0,
        cash_balance=10_000.0,
    )
    daily_predictions = pd.DataFrame([
        {
            "date": _ts("2022-01-03"),
            "ticker": "AAA",
            "prediction": 0.5,
            "stock_open_price": 100.0,
            "stock_trading_volume": 1_000_000.0,
            REALIZED_RETURN_COLUMN: 0.01,
        },
    ])

    with pytest.raises(ValueError, match="portfolio_construction_mode must be optimizer_miqp"):
        process_prediction_day(
            state,
            daily_predictions,
            _build_test_runtime(
                unique_dates=pd.Index([_ts("2022-01-03"), _ts("2022-01-04")]),
                cost_config=_build_test_cost_config(),
                optimizer_artifacts=_build_test_optimizer_artifacts(),
                portfolio_construction_mode="invalid_mode",
            ),
        )


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


def test_build_buy_and_hold_benchmark_returns_uses_drifted_weights() -> None:
    predictions = pd.DataFrame([
        {
            "date": _ts("2022-01-03"),
            "ticker": "AAA",
            "hl_context_stock_close_price": 100.0,
        },
        {
            "date": _ts("2022-01-03"),
            "ticker": "BBB",
            "hl_context_stock_close_price": 100.0,
        },
        {
            "date": _ts("2022-01-04"),
            "ticker": "AAA",
            "hl_context_stock_close_price": 200.0,
        },
        {
            "date": _ts("2022-01-04"),
            "ticker": "BBB",
            "hl_context_stock_close_price": 50.0,
        },
        {
            "date": _ts("2022-01-05"),
            "ticker": "AAA",
            "hl_context_stock_close_price": 100.0,
        },
        {
            "date": _ts("2022-01-05"),
            "ticker": "BBB",
            "hl_context_stock_close_price": 50.0,
        },
    ])

    observed = evaluate_backtest._build_buy_and_hold_benchmark_returns(predictions)

    assert observed.loc[_ts("2022-01-03")] == pytest.approx(0.0)
    assert observed.loc[_ts("2022-01-04")] == pytest.approx(0.25)
    assert observed.loc[_ts("2022-01-05")] == pytest.approx(-0.4)


def test_iter_model_prediction_days_frozen_train_only_uses_train_fit() -> None:
    dates = pd.date_range("2022-01-03", periods=8, freq="B")
    rows: list[dict[str, object]] = []
    for idx, date in enumerate(cast(pd.DatetimeIndex, dates)):
        split = "train" if idx < 5 else "test"
        for ticker in ["AAA", "BBB"]:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "dataset_split": split,
                    "target_week_hold_net_cs_zscore": float(idx),
                    "feature_a": float(idx if ticker == "AAA" else -idx),
                },
            )
    frame = pd.DataFrame(rows)
    model_spec = ModelSpec(model_name="ridge", params={"alpha": 1.0})
    prediction_days = list(
        iter_model_prediction_days_frozen_train_only(
            frame,
            feature_columns=["feature_a"],
            model_spec=model_spec,
            execution_lag_days=1,
            logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        ),
    )
    assert prediction_days
    for day in prediction_days:
        assert set(day["dataset_split"].astype(str).unique()) == {"test"}
        assert (day["date"] > day["signal_date"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
