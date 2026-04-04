from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.model_contract import DATE_COLUMN, PREDICTION_COLUMN
from core.src.meta_model.research_metrics import compute_mean_daily_spearman_ic


def _reference_mean_daily_spearman_ic(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
) -> float:
    frame = pd.DataFrame({
        DATE_COLUMN: pd.to_datetime(dates),
        PREDICTION_COLUMN: predictions,
        "target": targets,
    })
    daily_rank_ic: list[float] = []
    for _, group in frame.groupby(DATE_COLUMN, sort=False):
        if len(group) < 2:
            continue
        prediction_ranks = group[PREDICTION_COLUMN].rank(method="average")
        target_ranks = group["target"].rank(method="average")
        if prediction_ranks.nunique(dropna=False) < 2 or target_ranks.nunique(dropna=False) < 2:
            continue
        rank_ic = prediction_ranks.corr(target_ranks)
        if pd.notna(rank_ic):
            daily_rank_ic.append(float(rank_ic))
    if not daily_rank_ic:
        return 0.0
    return float(np.mean(np.asarray(daily_rank_ic, dtype=np.float64)))


class TestComputeMeanDailySpearmanIc:
    def test_matches_reference_with_unsorted_dates_ties_and_nan(self) -> None:
        predictions = np.asarray([0.2, np.nan, 0.2, 1.0, 0.5, 0.5, -1.0], dtype=np.float64)
        targets = np.asarray([1.0, 0.5, 1.0, -1.0, 0.2, np.nan, 0.3], dtype=np.float64)
        dates = np.asarray([
            "2024-01-03",
            "2024-01-02",
            "2024-01-03",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-02",
        ])

        expected = _reference_mean_daily_spearman_ic(predictions, targets, dates)
        actual = compute_mean_daily_spearman_ic(predictions, targets, dates)

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

            expected = _reference_mean_daily_spearman_ic(predictions, targets, dates)
            actual = compute_mean_daily_spearman_ic(predictions, targets, dates)

            assert actual == pytest.approx(expected, abs=1e-12)

    def test_returns_zero_when_no_day_has_valid_cross_section(self) -> None:
        predictions = np.asarray([1.0, 1.0, np.nan], dtype=np.float64)
        targets = np.asarray([0.0, 0.0, 0.1], dtype=np.float64)
        dates = np.asarray(["2024-01-02", "2024-01-02", "2024-01-03"])

        assert compute_mean_daily_spearman_ic(predictions, targets, dates) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
