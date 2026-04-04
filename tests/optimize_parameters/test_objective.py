from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.optimize_parameters.objective import (
    aggregate_fold_rank_ic,
    bootstrap_rank_ic_objective_standard_error,
)


def _reference_bootstrap_rank_ic_objective_standard_error(
    fold_rank_ic: list[float],
    fold_window_std: list[float],
    *,
    stability_penalty_alpha: float,
    train_window_stability_alpha: float,
    bootstrap_samples: int,
    random_seed: int,
) -> float:
    rank_ic = np.asarray(fold_rank_ic, dtype=np.float64)
    window_std = np.asarray(fold_window_std, dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    indices = np.arange(rank_ic.size, dtype=np.int64)
    bootstrap_scores = np.empty(bootstrap_samples, dtype=np.float64)
    for sample_index in range(bootstrap_samples):
        sampled_indices = rng.choice(indices, size=rank_ic.size, replace=True)
        aggregate = aggregate_fold_rank_ic(
            rank_ic[sampled_indices].tolist(),
            window_std[sampled_indices].tolist(),
            stability_penalty_alpha=stability_penalty_alpha,
            train_window_stability_alpha=train_window_stability_alpha,
            complexity_penalty=0.0,
            objective_standard_error=0.0,
        )
        bootstrap_scores[sample_index] = aggregate["objective_base_score"]
    return float(bootstrap_scores.std(ddof=1))


class TestBootstrapRankIcObjectiveStandardError:
    def test_matches_reference_loop_exactly_for_fixed_seed(self) -> None:
        fold_rank_ic = [0.01, 0.02, 0.015, 0.03, 0.01, 0.025, 0.021, 0.017, 0.019, 0.023, 0.011, 0.018]
        fold_window_std = [0.001, 0.002, 0.0015, 0.0025, 0.0012, 0.0021, 0.0018, 0.0011, 0.0017, 0.0022, 0.0014, 0.0019]

        expected = _reference_bootstrap_rank_ic_objective_standard_error(
            fold_rank_ic,
            fold_window_std,
            stability_penalty_alpha=0.10,
            train_window_stability_alpha=0.05,
            bootstrap_samples=1024,
            random_seed=7,
        )
        actual = bootstrap_rank_ic_objective_standard_error(
            fold_rank_ic,
            fold_window_std,
            stability_penalty_alpha=0.10,
            train_window_stability_alpha=0.05,
            bootstrap_samples=1024,
            random_seed=7,
        )

        assert actual == pytest.approx(expected, abs=0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
