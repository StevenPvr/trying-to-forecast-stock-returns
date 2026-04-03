from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.overfitting import (
    build_overfitting_diagnostics,
    compute_deflated_sharpe_ratio,
    estimate_probability_of_backtest_overfitting,
)


def test_deflated_sharpe_ratio_penalizes_many_trials() -> None:
    returns = pd.Series([0.01, 0.008, 0.012, 0.007, 0.011, 0.009] * 10, dtype=float)

    low_trial_dsr = compute_deflated_sharpe_ratio(returns, trial_count=1)
    high_trial_dsr = compute_deflated_sharpe_ratio(returns, trial_count=100)

    assert high_trial_dsr <= low_trial_dsr


def test_probability_of_backtest_overfitting_detects_unstable_trials() -> None:
    trials = pd.DataFrame([
        {"trial_number": 0, "fold_1_daily_rank_ic": 0.09, "fold_2_daily_rank_ic": 0.08, "fold_3_daily_rank_ic": -0.03, "fold_4_daily_rank_ic": -0.02},
        {"trial_number": 1, "fold_1_daily_rank_ic": 0.01, "fold_2_daily_rank_ic": 0.02, "fold_3_daily_rank_ic": 0.03, "fold_4_daily_rank_ic": 0.02},
        {"trial_number": 2, "fold_1_daily_rank_ic": 0.07, "fold_2_daily_rank_ic": -0.04, "fold_3_daily_rank_ic": 0.08, "fold_4_daily_rank_ic": -0.05},
    ])

    pbo = estimate_probability_of_backtest_overfitting(trials)

    assert 0.0 <= pbo <= 1.0
    assert pbo > 0.0


def test_build_overfitting_diagnostics_returns_consistent_payload() -> None:
    returns = pd.Series([0.01, -0.005, 0.012, -0.004, 0.008, 0.006] * 12, dtype=float)
    trials = pd.DataFrame([
        {"trial_number": 0, "fold_1_daily_rank_ic": 0.03, "fold_2_daily_rank_ic": 0.02},
        {"trial_number": 1, "fold_1_daily_rank_ic": 0.01, "fold_2_daily_rank_ic": 0.00},
    ])

    diagnostics = build_overfitting_diagnostics(returns, trial_count=2, trials_frame=trials)

    assert diagnostics.trial_count == 2
    assert 0.0 <= diagnostics.pbo <= 1.0
    assert diagnostics.minimum_track_record_length > 0.0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
