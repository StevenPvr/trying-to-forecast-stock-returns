from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.objective import (
    FoldEconomicScore,
    aggregate_subset_score,
    is_candidate_move_acceptable,
    passes_selection_gates,
)


def _make_fold(
    index: int,
    *,
    weight: float,
    net_pnl_after_costs: float,
    alpha_over_benchmark_net: float,
    turnover_annualized: float,
    max_drawdown: float,
    daily_rank_ic_mean: float | None = None,
    daily_rank_ic_ir: float | None = None,
    daily_top_bottom_spread_mean: float | None = None,
) -> FoldEconomicScore:
    resolved_rank_ic_mean = net_pnl_after_costs if daily_rank_ic_mean is None else daily_rank_ic_mean
    resolved_rank_ic_ir = resolved_rank_ic_mean if daily_rank_ic_ir is None else daily_rank_ic_ir
    resolved_top_bottom_spread = (
        alpha_over_benchmark_net
        if daily_top_bottom_spread_mean is None
        else daily_top_bottom_spread_mean
    )
    return FoldEconomicScore(
        index=index,
        weight=weight,
        net_pnl_after_costs=net_pnl_after_costs,
        alpha_over_benchmark_net=alpha_over_benchmark_net,
        turnover_annualized=turnover_annualized,
        max_drawdown=max_drawdown,
        daily_rank_ic_mean=resolved_rank_ic_mean,
        daily_rank_ic_ir=resolved_rank_ic_ir,
        daily_top_bottom_spread_mean=resolved_top_bottom_spread,
    )


class TestFeatureSelectionObjective:
    def test_aggregate_subset_score_weights_recent_folds(self) -> None:
        score = aggregate_subset_score(
            ["feature_a", "feature_b"],
            [
                _make_fold(
                    1,
                    weight=1.0,
                    net_pnl_after_costs=0.01,
                    alpha_over_benchmark_net=0.01,
                    turnover_annualized=0.50,
                    max_drawdown=-0.05,
                ),
                _make_fold(
                    2,
                    weight=2.0,
                    net_pnl_after_costs=0.04,
                    alpha_over_benchmark_net=0.03,
                    turnover_annualized=0.60,
                    max_drawdown=-0.04,
                ),
            ],
        )

        assert score.weighted_net_pnl_after_costs == pytest.approx(0.03)
        assert score.weighted_daily_rank_ic_mean == pytest.approx(0.03)
        assert score.weighted_alpha_over_benchmark_net == pytest.approx(0.0233333333)
        assert score.positive_fold_share == pytest.approx(1.0)

    def test_is_candidate_move_acceptable_rejects_unstable_candidate(self) -> None:
        config = FeatureSelectionConfig(
            positive_fold_share_min=0.60,
            turnover_guardrail_multiplier=1.15,
            max_drawdown_guardrail_additive=0.05,
            null_bootstrap_count=8,
        )
        current = aggregate_subset_score(
            ["feature_a"],
            [
                _make_fold(
                    1,
                    weight=1.0,
                    net_pnl_after_costs=0.02,
                    alpha_over_benchmark_net=0.01,
                    turnover_annualized=0.50,
                    max_drawdown=-0.05,
                ),
                _make_fold(
                    2,
                    weight=2.0,
                    net_pnl_after_costs=0.03,
                    alpha_over_benchmark_net=0.02,
                    turnover_annualized=0.52,
                    max_drawdown=-0.06,
                ),
            ],
        )
        candidate = aggregate_subset_score(
            ["feature_a", "feature_b"],
            [
                _make_fold(
                    1,
                    weight=1.0,
                    net_pnl_after_costs=-0.01,
                    alpha_over_benchmark_net=-0.01,
                    turnover_annualized=0.80,
                    max_drawdown=-0.20,
                ),
                _make_fold(
                    2,
                    weight=2.0,
                    net_pnl_after_costs=0.06,
                    alpha_over_benchmark_net=0.04,
                    turnover_annualized=0.82,
                    max_drawdown=-0.18,
                ),
            ],
        )

        assert not is_candidate_move_acceptable(candidate, current, config)

    def test_passes_selection_gates_accepts_weak_but_consistent_signal(self) -> None:
        config = FeatureSelectionConfig(
            positive_fold_share_min=0.50,
            lower_quartile_fold_pnl_floor=-0.0005,
            null_bootstrap_count=0,
        )
        candidate = aggregate_subset_score(
            ["feature_a", "feature_b"],
            [
                _make_fold(
                    1,
                    weight=1.0,
                    net_pnl_after_costs=0.0004,
                    alpha_over_benchmark_net=0.0003,
                    turnover_annualized=0.45,
                    max_drawdown=-0.04,
                ),
                _make_fold(
                    2,
                    weight=2.0,
                    net_pnl_after_costs=0.0008,
                    alpha_over_benchmark_net=0.0005,
                    turnover_annualized=0.47,
                    max_drawdown=-0.03,
                ),
            ],
        )

        assert passes_selection_gates(candidate, config)
        assert is_candidate_move_acceptable(candidate, None, config)

    def test_passes_selection_gates_accepts_micro_signal_with_one_small_negative_fold(self) -> None:
        config = FeatureSelectionConfig(
            positive_fold_share_min=0.50,
            lower_quartile_fold_pnl_floor=-0.0005,
            null_bootstrap_count=0,
        )
        candidate = aggregate_subset_score(
            ["feature_a", "feature_b"],
            [
                _make_fold(
                    1,
                    weight=1.0,
                    net_pnl_after_costs=-0.0002,
                    alpha_over_benchmark_net=0.0001,
                    turnover_annualized=0.45,
                    max_drawdown=-0.04,
                ),
                _make_fold(
                    2,
                    weight=2.0,
                    net_pnl_after_costs=0.0006,
                    alpha_over_benchmark_net=0.0004,
                    turnover_annualized=0.47,
                    max_drawdown=-0.03,
                ),
                _make_fold(
                    3,
                    weight=3.0,
                    net_pnl_after_costs=0.0007,
                    alpha_over_benchmark_net=0.0005,
                    turnover_annualized=0.46,
                    max_drawdown=-0.03,
                ),
                _make_fold(
                    4,
                    weight=4.0,
                    net_pnl_after_costs=0.0009,
                    alpha_over_benchmark_net=0.0006,
                    turnover_annualized=0.46,
                    max_drawdown=-0.02,
                ),
            ],
        )

        assert candidate.objective_score > 0.0
        assert candidate.lower_quartile_fold_net_pnl >= -0.0005
        assert passes_selection_gates(candidate, config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
