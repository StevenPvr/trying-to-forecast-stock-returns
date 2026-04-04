from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.optimize_parameters.metric_context import (
    build_daily_rank_ic_context,
    compute_mean_daily_rank_ic_from_context,
)
from core.src.meta_model.research_metrics import compute_mean_daily_spearman_ic


class TestDailyRankIcContext:
    def test_matches_reference_on_unsorted_dates_with_ties(self) -> None:
        predictions = np.asarray([0.3, 1.2, 0.3, 0.9, np.nan, 0.4], dtype=np.float64)
        targets = np.asarray([1.0, -1.0, 1.0, 0.7, 0.5, np.nan], dtype=np.float64)
        dates = np.asarray([
            "2024-01-03",
            "2024-01-02",
            "2024-01-03",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
        ])

        context = build_daily_rank_ic_context(targets, dates)
        actual = compute_mean_daily_rank_ic_from_context(predictions, context)
        expected = compute_mean_daily_spearman_ic(predictions, targets, dates)

        assert actual == pytest.approx(expected, abs=1e-12)

    def test_matches_reference_across_randomized_inputs(self) -> None:
        rng = np.random.default_rng(7)

        for size in (32, 128, 512):
            predictions = rng.normal(size=size).astype(np.float64)
            targets = rng.normal(size=size).astype(np.float64)
            predictions[rng.integers(0, size, size=max(1, size // 8))] = 0.0
            targets[rng.integers(0, size, size=max(1, size // 8))] = 1.0
            predictions[rng.integers(0, size, size=max(1, size // 16))] = np.nan
            targets[rng.integers(0, size, size=max(1, size // 16))] = np.nan
            day_ids = rng.integers(0, max(4, size // 8), size=size)
            dates = (
                pd.Timestamp("2020-01-01") + pd.to_timedelta(day_ids, unit="D")
            ).to_numpy()

            context = build_daily_rank_ic_context(targets, dates)
            actual = compute_mean_daily_rank_ic_from_context(predictions, context)
            expected = compute_mean_daily_spearman_ic(predictions, targets, dates)

            assert actual == pytest.approx(expected, abs=1e-12)

    def test_returns_zero_for_empty_valid_groups(self) -> None:
        predictions = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        targets = np.asarray([0.0, 0.0, 0.5], dtype=np.float64)
        dates = np.asarray(["2024-01-02", "2024-01-02", "2024-01-03"])

        context = build_daily_rank_ic_context(targets, dates)

        assert compute_mean_daily_rank_ic_from_context(predictions, context) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
