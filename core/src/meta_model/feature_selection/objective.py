from __future__ import annotations

"""Objective function: fold-level economic scoring and selection gate logic."""

from dataclasses import dataclass

import numpy as np

from core.src.meta_model.feature_selection.config import FeatureSelectionConfig


@dataclass(frozen=True)
class FoldEconomicScore:
    """Economic score for a single cross-validation fold."""

    index: int
    weight: float
    net_pnl_after_costs: float
    alpha_over_benchmark_net: float
    turnover_annualized: float
    max_drawdown: float
    daily_rank_ic_mean: float = 0.0
    daily_rank_ic_ir: float = 0.0
    daily_top_bottom_spread_mean: float = 0.0


@dataclass(frozen=True)
class SubsetEconomicScore:
    """Aggregated economic score across all folds for a feature subset."""

    feature_names: tuple[str, ...]
    objective_score: float
    weighted_net_pnl_after_costs: float
    weighted_alpha_over_benchmark_net: float
    weighted_turnover_annualized: float
    weighted_max_drawdown: float
    positive_fold_share: float
    median_fold_net_pnl: float
    lower_quartile_fold_net_pnl: float
    is_valid: bool
    fold_scores: tuple[FoldEconomicScore, ...]
    pnl_positive_fold_share: float = 0.0
    weighted_daily_rank_ic_mean: float = 0.0
    weighted_daily_rank_ic_ir: float = 0.0
    weighted_daily_top_bottom_spread_mean: float = 0.0


def aggregate_subset_score(
    feature_names: list[str],
    fold_scores: list[FoldEconomicScore],
) -> SubsetEconomicScore:
    """Compute weighted aggregates across folds and return a ``SubsetEconomicScore``."""
    if not fold_scores:
        return SubsetEconomicScore(
            feature_names=tuple(sorted(feature_names)),
            objective_score=0.0,
            weighted_net_pnl_after_costs=0.0,
            weighted_alpha_over_benchmark_net=0.0,
            weighted_turnover_annualized=0.0,
            weighted_max_drawdown=0.0,
            positive_fold_share=0.0,
            median_fold_net_pnl=0.0,
            lower_quartile_fold_net_pnl=0.0,
            is_valid=False,
            fold_scores=tuple(),
        )
    weights = np.asarray([fold.weight for fold in fold_scores], dtype=np.float64)
    fold_net_pnl = np.asarray([fold.net_pnl_after_costs for fold in fold_scores], dtype=np.float64)
    fold_alpha = np.asarray([fold.alpha_over_benchmark_net for fold in fold_scores], dtype=np.float64)
    fold_turnover = np.asarray([fold.turnover_annualized for fold in fold_scores], dtype=np.float64)
    fold_drawdown = np.asarray([fold.max_drawdown for fold in fold_scores], dtype=np.float64)
    fold_rank_ic_mean = np.asarray([fold.daily_rank_ic_mean for fold in fold_scores], dtype=np.float64)
    fold_rank_ic_ir = np.asarray([fold.daily_rank_ic_ir for fold in fold_scores], dtype=np.float64)
    fold_top_bottom_spread = np.asarray(
        [fold.daily_top_bottom_spread_mean for fold in fold_scores],
        dtype=np.float64,
    )
    weighted_net_pnl = _weighted_mean(fold_net_pnl, weights)
    weighted_rank_ic_mean = _weighted_mean(fold_rank_ic_mean, weights)
    return SubsetEconomicScore(
        feature_names=tuple(sorted(feature_names)),
        objective_score=weighted_rank_ic_mean,
        weighted_net_pnl_after_costs=weighted_net_pnl,
        weighted_alpha_over_benchmark_net=_weighted_mean(fold_alpha, weights),
        weighted_turnover_annualized=_weighted_mean(fold_turnover, weights),
        weighted_max_drawdown=_weighted_mean(fold_drawdown, weights),
        positive_fold_share=float(np.mean(fold_rank_ic_mean > 0.0)),
        median_fold_net_pnl=float(np.median(fold_net_pnl)),
        lower_quartile_fold_net_pnl=float(np.quantile(fold_net_pnl, 0.25)),
        is_valid=weighted_rank_ic_mean > 0.0,
        fold_scores=tuple(fold_scores),
        pnl_positive_fold_share=float(np.mean(fold_net_pnl > 0.0)),
        weighted_daily_rank_ic_mean=weighted_rank_ic_mean,
        weighted_daily_rank_ic_ir=_weighted_mean(fold_rank_ic_ir, weights),
        weighted_daily_top_bottom_spread_mean=_weighted_mean(fold_top_bottom_spread, weights),
    )


def is_candidate_move_acceptable(
    candidate_score: SubsetEconomicScore,
    current_score: SubsetEconomicScore | None,
    config: FeatureSelectionConfig,
) -> bool:
    """Return True if *candidate_score* improves on *current_score* within guardrails."""
    if not passes_selection_gates(candidate_score, config):
        return False
    if current_score is None:
        return candidate_score.objective_score > 0.0
    if candidate_score.objective_score <= current_score.objective_score:
        return False
    if candidate_score.weighted_turnover_annualized > (
        current_score.weighted_turnover_annualized * config.turnover_guardrail_multiplier
    ):
        return False
    if candidate_score.weighted_max_drawdown < (
        current_score.weighted_max_drawdown - config.max_drawdown_guardrail_additive
    ):
        return False
    return _exceeds_null_improvement(candidate_score, current_score, config.null_bootstrap_count)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.average(values, weights=weights))


def _exceeds_null_improvement(
    candidate_score: SubsetEconomicScore,
    current_score: SubsetEconomicScore,
    bootstrap_count: int,
) -> bool:
    if bootstrap_count <= 0:
        return True
    candidate_values = np.asarray([fold.daily_rank_ic_mean for fold in candidate_score.fold_scores], dtype=np.float64)
    current_values = np.asarray([fold.daily_rank_ic_mean for fold in current_score.fold_scores], dtype=np.float64)
    if candidate_values.size != current_values.size or candidate_values.size == 0:
        return True
    observed_improvement = float(candidate_score.objective_score - current_score.objective_score)
    fold_deltas = candidate_values - current_values
    if np.allclose(fold_deltas, 0.0):
        return False
    centered_deltas = fold_deltas - float(fold_deltas.mean())
    seed_value = hash(tuple(fold_deltas.tolist())) & 0xFFFF_FFFF
    rng = np.random.default_rng(seed_value)
    null_scores = np.empty(bootstrap_count, dtype=np.float64)
    for i in range(bootstrap_count):
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=centered_deltas.size)
        null_scores[i] = float(np.mean(centered_deltas * signs))
    return observed_improvement > float(np.quantile(null_scores, 0.95))


def passes_selection_gates(
    candidate_score: SubsetEconomicScore,
    config: FeatureSelectionConfig,
) -> bool:
    """Return True if *candidate_score* passes all hard selection gates."""
    return (
        candidate_score.objective_score > 0.0
        and candidate_score.weighted_daily_rank_ic_ir > 0.0
        and candidate_score.weighted_daily_top_bottom_spread_mean > 0.0
        and candidate_score.positive_fold_share >= config.positive_fold_share_min
        and candidate_score.pnl_positive_fold_share >= config.pnl_positive_fold_share_min
        and candidate_score.lower_quartile_fold_net_pnl >= config.lower_quartile_fold_pnl_floor
    )


def passes_sfi_gates(
    candidate_score: SubsetEconomicScore,
    config: FeatureSelectionConfig,
) -> bool:
    """Return True if *candidate_score* passes the lighter SFI-stage gates."""
    return (
        candidate_score.objective_score > 0.0
        and candidate_score.weighted_daily_rank_ic_ir > 0.0
        and candidate_score.positive_fold_share >= config.positive_fold_share_min
    )


__all__ = [
    "FoldEconomicScore",
    "SubsetEconomicScore",
    "aggregate_subset_score",
    "is_candidate_move_acceptable",
    "passes_sfi_gates",
    "passes_selection_gates",
]
