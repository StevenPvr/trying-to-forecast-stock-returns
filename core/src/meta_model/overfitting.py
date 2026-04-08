from __future__ import annotations

"""Overfitting diagnostics: Probabilistic Sharpe Ratio, Deflated Sharpe, PBO."""

from dataclasses import asdict, dataclass
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

_NORMAL = NormalDist()
_EPSILON: float = 1e-12
_EULER_GAMMA: float = 0.5772156649015329


@dataclass(frozen=True)
class OverfittingDiagnostics:
    """Immutable snapshot of overfitting risk metrics for a given strategy."""

    trial_count: int
    pbo: float
    sharpe_ratio: float
    probabilistic_sharpe_ratio: float
    deflated_sharpe_ratio: float
    minimum_track_record_length: float


def compute_sharpe_ratio(returns: pd.Series, *, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (excess return / volatility, zero risk-free rate)."""
    clean_returns = pd.Series(returns, dtype=float).dropna()
    if clean_returns.empty:
        return 0.0
    std = float(clean_returns.std(ddof=0))
    if std <= _EPSILON:
        return 0.0
    return float(clean_returns.mean() / std * np.sqrt(periods_per_year))


def compute_probabilistic_sharpe_ratio(
    returns: pd.Series,
    *,
    benchmark_sharpe: float,
    periods_per_year: int = 252,
) -> float:
    """Probability that the true Sharpe exceeds *benchmark_sharpe* (Bailey & Lopez de Prado)."""
    clean_returns = pd.Series(returns, dtype=float).dropna()
    if len(clean_returns) < 2:
        return 0.0
    sharpe_ratio = compute_sharpe_ratio(clean_returns, periods_per_year=periods_per_year)
    skewness = float(clean_returns.skew())
    kurtosis = float(clean_returns.kurt())
    denominator = 1.0 - skewness * sharpe_ratio + ((kurtosis - 1.0) / 4.0) * sharpe_ratio**2
    if denominator <= _EPSILON:
        return 0.0
    z_score = (
        (sharpe_ratio - benchmark_sharpe)
        * np.sqrt(max(len(clean_returns) - 1, 1))
        / np.sqrt(denominator)
    )
    return float(_NORMAL.cdf(z_score))


def compute_minimum_track_record_length(
    returns: pd.Series,
    *,
    benchmark_sharpe: float,
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
) -> float:
    """Minimum number of observations to reject H0: Sharpe <= *benchmark_sharpe*."""
    clean_returns = pd.Series(returns, dtype=float).dropna()
    if clean_returns.empty:
        return float("inf")
    sharpe_ratio = compute_sharpe_ratio(clean_returns, periods_per_year=periods_per_year)
    if sharpe_ratio <= benchmark_sharpe:
        return float("inf")
    skewness = float(clean_returns.skew())
    kurtosis = float(clean_returns.kurt())
    denominator = (sharpe_ratio - benchmark_sharpe) ** 2
    numerator = max(
        1.0 - skewness * sharpe_ratio + ((kurtosis - 1.0) / 4.0) * sharpe_ratio**2,
        _EPSILON,
    )
    z_value = _NORMAL.inv_cdf(confidence_level)
    return float(max(1.0, 1.0 + numerator * (z_value**2) / max(denominator, _EPSILON)))


def compute_expected_max_sharpe(trial_count: int) -> float:
    """Expected maximum Sharpe under the null across *trial_count* independent trials."""
    if trial_count <= 1:
        return 0.0
    adjusted_trial_count = max(float(trial_count), 2.0)
    z_high = _NORMAL.inv_cdf(1.0 - 1.0 / adjusted_trial_count)
    z_mid = _NORMAL.inv_cdf(1.0 - 1.0 / (adjusted_trial_count * np.e))
    return float((1.0 - _EULER_GAMMA) * z_high + _EULER_GAMMA * z_mid)


def compute_deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    trial_count: int,
    periods_per_year: int = 252,
) -> float:
    """Deflated Sharpe Ratio accounting for multiple testing across *trial_count* trials."""
    benchmark_sharpe = compute_expected_max_sharpe(trial_count)
    return compute_probabilistic_sharpe_ratio(
        returns,
        benchmark_sharpe=benchmark_sharpe,
        periods_per_year=periods_per_year,
    )


def estimate_probability_of_backtest_overfitting(
    trials_frame: pd.DataFrame,
    *,
    fold_column_prefix: str = "fold_",
    score_column_suffix: str = "_daily_rank_ic",
) -> float:
    """Probability of Backtest Overfitting via combinatorial purged cross-validation."""
    fold_columns = [
        column
        for column in trials_frame.columns
        if column.startswith(fold_column_prefix) and column.endswith(score_column_suffix)
    ]
    completed = pd.DataFrame(trials_frame.loc[trials_frame[fold_columns].notna().all(axis=1)].copy())
    if len(fold_columns) < 2 or completed.empty:
        return 0.0
    score_matrix = completed.loc[:, fold_columns].to_numpy(dtype=np.float64)
    fold_indices = range(score_matrix.shape[1])
    train_fold_count = max(1, score_matrix.shape[1] // 2)
    logits: list[float] = []
    for train_subset in combinations(fold_indices, train_fold_count):
        test_subset = tuple(index for index in fold_indices if index not in train_subset)
        train_scores = score_matrix[:, train_subset].mean(axis=1)
        selected_trial_index = int(np.argmax(train_scores))
        test_scores = score_matrix[:, test_subset].mean(axis=1)
        percentile = float((test_scores <= test_scores[selected_trial_index]).mean())
        clipped_percentile = min(max(percentile, _EPSILON), 1.0 - _EPSILON)
        logits.append(float(np.log(clipped_percentile / (1.0 - clipped_percentile))))
    if not logits:
        return 0.0
    return float(np.mean(np.asarray(logits) <= 0.0))


def build_overfitting_diagnostics(
    returns: pd.Series,
    *,
    trial_count: int,
    trials_frame: pd.DataFrame | None = None,
) -> OverfittingDiagnostics:
    """Compute all overfitting diagnostics in a single call."""
    pbo = 0.0 if trials_frame is None else estimate_probability_of_backtest_overfitting(trials_frame)
    sharpe_ratio = compute_sharpe_ratio(returns)
    probabilistic_sharpe_ratio = compute_probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)
    deflated_sharpe_ratio = compute_deflated_sharpe_ratio(returns, trial_count=trial_count)
    minimum_track_record_length = compute_minimum_track_record_length(
        returns,
        benchmark_sharpe=compute_expected_max_sharpe(trial_count),
    )
    return OverfittingDiagnostics(
        trial_count=trial_count,
        pbo=pbo,
        sharpe_ratio=sharpe_ratio,
        probabilistic_sharpe_ratio=probabilistic_sharpe_ratio,
        deflated_sharpe_ratio=deflated_sharpe_ratio,
        minimum_track_record_length=minimum_track_record_length,
    )


def diagnostics_to_payload(diagnostics: OverfittingDiagnostics) -> dict[str, float]:
    """Serialise diagnostics to a flat dict of floats for JSON output."""
    return {key: float(value) for key, value in asdict(diagnostics).items()}
