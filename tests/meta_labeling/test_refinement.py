from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from core.src.meta_model.meta_labeling.refinement import (
    KNOWN_STRATEGIES,
    _apply_binary_gate,
    _apply_candidate_only,
    _apply_no_meta,
    _apply_rank_blend,
    _apply_soft_shifted,
    _cross_sectional_zscore,
    compute_refined_signal,
)

_DATE = "date"
_TICKER = "ticker"
_PRIMARY = "primary_prediction"
_META_PROB = "meta_probability"
_EXP_RET = "expected_return_5d"
_CONFIDENCE = "meta_confidence"
_REFINED = "refined_prediction"
_REFINED_EXP = "refined_expected_return_5d"


def _make_frame(
    primary: list[float],
    meta_prob: list[float],
    *,
    expected: list[float] | None = None,
    dates: list[str] | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    n = len(primary)
    data: dict[str, list[object]] = {
        _DATE: dates or [f"2024-01-0{i + 1}" for i in range(n)],
        _TICKER: tickers or [f"T{i}" for i in range(n)],
        "dataset_split": ["train"] * n,
        _PRIMARY: primary,
        _META_PROB: meta_prob,
    }
    if expected is not None:
        data[_EXP_RET] = expected
    df = pd.DataFrame(data)
    df[_DATE] = pd.to_datetime(df[_DATE])
    return df


# --- binary_gate ---


def test_binary_gate_zeros_below_threshold() -> None:
    primary = np.array([0.5, 0.3, -0.2])
    meta = np.array([0.6, 0.4, 0.8])
    conf, refined, _ = _apply_binary_gate(primary, meta, None)
    assert conf.tolist() == pytest.approx([0.6, 0.0, 0.8])
    assert refined.tolist() == pytest.approx([0.3, 0.0, -0.16])


def test_binary_gate_reproduces_current_behavior() -> None:
    frame = _make_frame(
        primary=[0.5, -0.3],
        meta_prob=[0.75, 0.40],
        expected=[0.10, 0.05],
    )
    result = compute_refined_signal(frame, strategy="binary_gate")
    assert result[_CONFIDENCE].tolist() == pytest.approx([0.75, 0.0])
    assert result[_REFINED].tolist() == pytest.approx([0.375, 0.0])
    assert result[_REFINED_EXP].tolist() == pytest.approx([0.075, 0.0])


# --- candidate_only ---


def test_candidate_only_keeps_positive_primary_only() -> None:
    primary = np.array([0.5, -0.3, 0.0, 0.1])
    meta = np.array([0.9, 0.9, 0.9, 0.2])
    conf, refined, _ = _apply_candidate_only(primary, meta, None)
    assert conf.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])
    assert refined.tolist() == pytest.approx([0.5, 0.0, 0.0, 0.1])


def test_candidate_only_ignores_meta_probability() -> None:
    primary = np.array([0.5, 0.5])
    meta_low = np.array([0.1, 0.1])
    meta_high = np.array([0.9, 0.9])
    _, ref_low, _ = _apply_candidate_only(primary, meta_low, None)
    _, ref_high, _ = _apply_candidate_only(primary, meta_high, None)
    assert ref_low.tolist() == pytest.approx(ref_high.tolist())


# --- soft_shifted ---


def test_soft_shifted_floor_is_zero() -> None:
    primary = np.array([1.0])
    meta = np.array([0.45])
    conf, refined, _ = _apply_soft_shifted(primary, meta, None, floor=0.45)
    assert conf.tolist() == pytest.approx([0.0])
    assert refined.tolist() == pytest.approx([0.0])


def test_soft_shifted_linear_ramp() -> None:
    primary = np.array([1.0])
    meta = np.array([0.725])
    conf, _, _ = _apply_soft_shifted(primary, meta, None, floor=0.45)
    expected_conf = (0.725 - 0.45) / (1.0 - 0.45)
    assert conf.tolist() == pytest.approx([expected_conf])


def test_soft_shifted_at_one_is_one() -> None:
    primary = np.array([1.0])
    meta = np.array([1.0])
    conf, _, _ = _apply_soft_shifted(primary, meta, None, floor=0.45)
    assert conf.tolist() == pytest.approx([1.0])


# --- rank_blend ---


def test_rank_blend_zscore_per_date() -> None:
    primary = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    meta = pd.Series([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    dates = pd.Series(["2024-01-01"] * 3 + ["2024-01-02"] * 3)
    result = _apply_rank_blend(primary, meta, dates, blend_lambda=1.0)
    assert len(result) == 6
    assert np.all(np.isfinite(result))
    day1 = result[:3]
    assert day1.sum() == pytest.approx(0.0, abs=1e-8)


def test_rank_blend_lambda_zero_equals_primary_zscore() -> None:
    primary = pd.Series([0.1, 0.3, 0.5, 0.7])
    meta = pd.Series([0.9, 0.1, 0.5, 0.3])
    dates = pd.Series(["2024-01-01"] * 4)
    blend = _apply_rank_blend(primary, meta, dates, blend_lambda=0.0)
    z_primary = _cross_sectional_zscore(primary, dates)
    assert blend.tolist() == pytest.approx(z_primary.tolist())


def test_rank_blend_passthrough_expected_return() -> None:
    frame = _make_frame(
        primary=[0.1, 0.2, 0.3],
        meta_prob=[0.5, 0.6, 0.4],
        expected=[0.05, 0.10, 0.03],
        dates=["2024-01-01"] * 3,
        tickers=["A", "B", "C"],
    )
    result = compute_refined_signal(frame, strategy="rank_blend", rank_blend_lambda=0.5)
    assert result[_CONFIDENCE].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert result[_REFINED_EXP].tolist() == pytest.approx([0.05, 0.10, 0.03])


# --- no_meta ---


def test_no_meta_passes_primary_through() -> None:
    primary = np.array([0.5, -0.3, 0.0])
    meta = np.array([0.1, 0.9, 0.5])
    expected = np.array([0.02, -0.01, 0.0])
    conf, refined, refined_exp = _apply_no_meta(primary, meta, expected)
    assert conf.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert refined.tolist() == pytest.approx([0.5, -0.3, 0.0])
    assert refined_exp is not None
    assert refined_exp.tolist() == pytest.approx([0.02, -0.01, 0.0])


# --- dispatcher ---


def test_unknown_strategy_raises() -> None:
    frame = _make_frame(primary=[0.1], meta_prob=[0.5])
    with pytest.raises(ValueError, match="Unknown refinement strategy"):
        compute_refined_signal(frame, strategy="magic_strategy")


def test_compute_refined_signal_default_matches_binary_gate() -> None:
    frame = _make_frame(
        primary=[0.5, -0.3],
        meta_prob=[0.75, 0.40],
        expected=[0.10, 0.05],
    )
    default_result = compute_refined_signal(frame)
    explicit_result = compute_refined_signal(frame, strategy="binary_gate")
    assert default_result[_REFINED].tolist() == pytest.approx(
        explicit_result[_REFINED].tolist(),
    )
    assert default_result[_CONFIDENCE].tolist() == pytest.approx(
        explicit_result[_CONFIDENCE].tolist(),
    )


def test_all_known_strategies_are_callable() -> None:
    frame = _make_frame(
        primary=[0.2, -0.1, 0.3],
        meta_prob=[0.6, 0.3, 0.5],
        expected=[0.01, -0.005, 0.02],
        dates=["2024-01-01"] * 3,
        tickers=["A", "B", "C"],
    )
    for strategy in KNOWN_STRATEGIES:
        result = compute_refined_signal(frame, strategy=strategy)
        assert _REFINED in result.columns
        assert _CONFIDENCE in result.columns
        assert _REFINED_EXP in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
