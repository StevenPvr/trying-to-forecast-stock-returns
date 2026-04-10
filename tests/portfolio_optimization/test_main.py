from __future__ import annotations

from concurrent.futures import Future
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.portfolio_optimization.config import PortfolioOptimizationConfig
from core.src.meta_model.portfolio_optimization.alpha_calibration import FittedAlphaCalibrator
from core.src.meta_model.portfolio_optimization import main as portfolio_main
from core.src.meta_model.runtime_parallelism import resolve_available_cpu_count


def test_build_trial_grid_is_compact_and_exhaustive() -> None:
    config = PortfolioOptimizationConfig()
    grid = portfolio_main._build_trial_grid(config)
    assert len(grid) == 64
    assert grid[0].trial_index == 0
    assert grid[-1].trial_index == 63


def test_trial_results_to_ledger_uses_readable_trial_state() -> None:
    result = portfolio_main.PortfolioTrialResult(
        trial_index=7,
        state="COMPLETE",
        params={"lambda_risk": 2.0},
        objective_score=1.23,
        metrics={"sharpe_ratio": 1.1},
        daily_row_count=12,
        allocation_row_count=3,
        elapsed_seconds=4.0,
    )
    ledger = portfolio_main._trial_results_to_ledger([result], lambda_cost=1.0)
    assert str(ledger.loc[0, "state"]) == "COMPLETE"
    assert float(ledger.loc[0, "lambda_cost"]) == 1.0
    assert float(ledger.loc[0, "lambda_risk"]) == 2.0
def test_portfolio_turnover_search_range_is_small_enough_for_weight_objective() -> None:
    config = PortfolioOptimizationConfig()
    assert max(config.lambda_turnover_grid) <= 0.01
    assert config.lambda_turnover <= max(config.lambda_turnover_grid)


def test_portfolio_grid_search_defaults_to_all_available_workers() -> None:
    config = PortfolioOptimizationConfig()
    assert config.trial_parallel_workers is None
    resolved_workers = portfolio_main.resolve_executor_worker_count(
        task_count=64,
        requested_workers=config.trial_parallel_workers,
    )
    assert resolved_workers == min(64, resolve_available_cpu_count())


def test_execute_trial_grid_uses_process_pool_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    tasks = [
        portfolio_main.PortfolioTrialTask(
            trial_index=0,
            params={"lambda_risk": 1.0},
        ),
    ]
    predictions = pd.DataFrame({"prediction": [0.1]})
    runtime = PortfolioOptimizationConfig()
    config = portfolio_main.BacktestConfig()
    covariance_by_lookback = {63: pd.DataFrame([[1.0]])}

    class _InlineProcessExecutor:
        def __init__(
            self,
            *,
            max_workers: int,
            mp_context: object,
            initializer: object,
            initargs: tuple[object, ...],
        ) -> None:
            captured["max_workers"] = max_workers
            captured["mp_context"] = mp_context
            captured["initializer"] = initializer
            captured["initargs"] = initargs

        def __enter__(self) -> "_InlineProcessExecutor":
            initializer = captured["initializer"]
            if not callable(initializer):
                raise TypeError("initializer must be callable")
            initializer(*tuple(captured["initargs"]))
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type, exc, tb
            return False

        def submit(self, fn: object, *args: object, **kwargs: object) -> Future[portfolio_main.PortfolioTrialResult]:
            future: Future[portfolio_main.PortfolioTrialResult] = Future()
            callable_fn = fn
            if not callable(callable_fn):
                raise TypeError("submit expected a callable")
            result = callable_fn(*args, **kwargs)
            future.set_result(result)
            return future

    def fake_as_completed(
        futures: list[Future[portfolio_main.PortfolioTrialResult]],
    ) -> list[Future[portfolio_main.PortfolioTrialResult]]:
        return futures

    def fake_run_grid_trial(
        task: portfolio_main.PortfolioTrialTask,
        *,
        predictions: pd.DataFrame,
        runtime: PortfolioOptimizationConfig,
        config: portfolio_main.BacktestConfig,
        covariance_by_lookback: dict[int, pd.DataFrame],
    ) -> portfolio_main.PortfolioTrialResult:
        del predictions, runtime, config, covariance_by_lookback
        return portfolio_main.PortfolioTrialResult(
            trial_index=task.trial_index,
            state="COMPLETE",
            params=task.params,
            objective_score=1.0,
            metrics={"sharpe_ratio": 1.0},
            daily_row_count=1,
            allocation_row_count=1,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(portfolio_main, "ProcessPoolExecutor", _InlineProcessExecutor)
    monkeypatch.setattr(portfolio_main, "as_completed", fake_as_completed)
    monkeypatch.setattr(portfolio_main, "_run_grid_trial", fake_run_grid_trial)

    results = portfolio_main._execute_trial_grid(
        tasks,
        predictions=predictions,
        runtime=runtime,
        config=config,
        covariance_by_lookback=covariance_by_lookback,
        trial_workers=3,
    )

    assert len(results) == 1
    assert results[0].trial_index == 0
    assert captured["max_workers"] == 3


def test_attach_refined_expected_returns_combines_primary_calibration_and_meta_confidence() -> None:
    predictions = pd.DataFrame({
        "prediction": [0.4, -0.2],
        "primary_prediction": [0.4, -0.2],
        "meta_probability": [0.80, 0.45],
    })

    refined = portfolio_main._attach_refined_expected_returns(
        predictions,
        calibrator=FittedAlphaCalibrator(
            x_thresholds=[-1.0, 1.0],
            y_thresholds=[-0.5, 0.5],
        ),
    )

    assert refined["expected_return_5d"].tolist() == pytest.approx([0.16, 0.0])
    assert refined["meta_confidence"].tolist() == pytest.approx([0.8, 0.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
