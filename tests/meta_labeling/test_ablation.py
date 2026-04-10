from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from core.src.meta_model.meta_labeling.ablation import (
    ABLATION_REGIMES,
    METRIC_KEYS,
    compute_regime_metrics,
    format_ablation_table,
    run_ablation_screen,
    select_top_regimes,
)


def _make_ablation_frame(n_dates: int = 10, n_tickers: int = 20) -> pd.DataFrame:
    """Synthetic prediction panel with enough cross-section for quintiles."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    rows: list[dict[str, object]] = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d,
                "ticker": t,
                "dataset_split": "train",
                "primary_prediction": rng.randn() * 0.05,
                "meta_probability": rng.uniform(0.2, 0.8),
                "meta_candidate": 1 if rng.rand() > 0.4 else 0,
                "target_week_hold_5sessions_net_log_return": rng.randn() * 0.02,
                "expected_return_5d": rng.randn() * 0.01,
            })
    return pd.DataFrame(rows)


def test_compute_regime_metrics_returns_all_keys() -> None:
    frame = _make_ablation_frame()
    regime = {"name": "no_meta", "strategy": "no_meta"}
    metrics = compute_regime_metrics(frame, regime)
    for key in METRIC_KEYS:
        assert key in metrics, f"Missing key: {key}"
        assert np.isfinite(metrics[key]), f"Non-finite value for {key}"


def test_run_ablation_screen_covers_all_regimes() -> None:
    frame = _make_ablation_frame()
    results = run_ablation_screen(frame, ABLATION_REGIMES)
    assert len(results) == len(ABLATION_REGIMES)
    for regime in ABLATION_REGIMES:
        assert str(regime["name"]) in results.index


def test_rank_ic_finite_for_all_regimes() -> None:
    frame = _make_ablation_frame()
    results = run_ablation_screen(frame, ABLATION_REGIMES)
    for regime_name in results.index:
        assert np.isfinite(results.loc[regime_name, "rank_ic"]), (
            f"Non-finite rank_ic for {regime_name}"
        )


def test_breadth_positive_for_non_baseline() -> None:
    frame = _make_ablation_frame()
    results = run_ablation_screen(frame, ABLATION_REGIMES)
    for name in ["no_meta", "candidate_only", "rank_blend_050"]:
        assert results.loc[name, "breadth"] > 0, f"Zero breadth for {name}"


def test_select_top_regimes_includes_baselines() -> None:
    frame = _make_ablation_frame()
    results = run_ablation_screen(frame, ABLATION_REGIMES)
    selected = select_top_regimes(results)
    assert "candidate_only" in selected
    assert "no_meta" in selected
    assert len(selected) >= 3


def test_constant_prediction_yields_near_zero_ic() -> None:
    frame = _make_ablation_frame(n_dates=5, n_tickers=30)
    frame["primary_prediction"] = 0.1
    frame["meta_probability"] = 0.5
    regime = {"name": "no_meta", "strategy": "no_meta"}
    metrics = compute_regime_metrics(frame, regime)
    assert abs(metrics["rank_ic"]) < 0.05


def test_format_ablation_table_produces_lines() -> None:
    frame = _make_ablation_frame(n_dates=3, n_tickers=10)
    results = run_ablation_screen(frame, ABLATION_REGIMES)
    table = format_ablation_table(results)
    lines = table.strip().split("\n")
    assert len(lines) == len(ABLATION_REGIMES) + 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
