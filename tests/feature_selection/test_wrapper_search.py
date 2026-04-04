# pyright: reportPrivateUsage=false
from __future__ import annotations

import sys
from threading import Lock
import time
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.grouping import FeatureGroup
from core.src.meta_model.feature_selection.objective import FoldEconomicScore, SubsetEconomicScore
from core.src.meta_model.feature_selection import wrapper_search as wrapper_search_module
from core.src.meta_model.feature_selection.wrapper_search import (
    WrapperSearchResult,
    refine_selected_feature_groups,
    search_feature_groups,
)


def _score_subset(feature_names: list[str]) -> SubsetEconomicScore:
    active = frozenset(feature_names)
    objective_map: dict[frozenset[str], float] = {
        frozenset(): 0.0,
        frozenset({"feature_a"}): 0.01,
        frozenset({"feature_b"}): 0.01,
        frozenset({"feature_c"}): -0.01,
        frozenset({"feature_a", "feature_b"}): 0.08,
        frozenset({"feature_a", "feature_b", "feature_c"}): 0.05,
    }
    objective = objective_map.get(active, -0.02)
    fold_scores = (
        FoldEconomicScore(
            index=1,
            weight=1.0,
            net_pnl_after_costs=objective,
            alpha_over_benchmark_net=max(objective, 0.0),
            turnover_annualized=0.40,
            max_drawdown=-0.05,
            daily_rank_ic_mean=objective,
            daily_rank_ic_ir=objective,
            daily_top_bottom_spread_mean=max(objective, 0.0),
        ),
        FoldEconomicScore(
            index=2,
            weight=2.0,
            net_pnl_after_costs=objective,
            alpha_over_benchmark_net=max(objective, 0.0),
            turnover_annualized=0.42,
            max_drawdown=-0.05,
            daily_rank_ic_mean=objective,
            daily_rank_ic_ir=objective,
            daily_top_bottom_spread_mean=max(objective, 0.0),
        ),
    )
    return SubsetEconomicScore(
        feature_names=tuple(sorted(feature_names)),
        objective_score=objective,
        weighted_net_pnl_after_costs=objective,
        weighted_alpha_over_benchmark_net=max(objective, 0.0),
        weighted_turnover_annualized=0.41,
        weighted_max_drawdown=-0.05,
        positive_fold_share=1.0 if objective > 0.0 else 0.0,
        median_fold_net_pnl=objective,
        lower_quartile_fold_net_pnl=objective,
        is_valid=objective > 0.0,
        fold_scores=fold_scores,
        weighted_daily_rank_ic_mean=objective,
        weighted_daily_rank_ic_ir=objective,
        weighted_daily_top_bottom_spread_mean=max(objective, 0.0),
    )


class TestWrapperSearch:
    def test_search_feature_groups_finds_pairwise_interaction(self) -> None:
        groups = [
            FeatureGroup(group_id="group_a", family="quant", stem="a", feature_names=("feature_a",)),
            FeatureGroup(group_id="group_b", family="quant", stem="b", feature_names=("feature_b",)),
            FeatureGroup(group_id="group_c", family="quant", stem="c", feature_names=("feature_c",)),
        ]
        config = FeatureSelectionConfig(
            search_beam_width=3,
            null_bootstrap_count=0,
            max_active_matrix_gib=1.0,
        )

        result = search_feature_groups(
            root_groups=groups,
            scorer=_score_subset,
            config=config,
            estimated_row_count=100,
        )

        assert result.selected_feature_names == ["feature_a", "feature_b"]
        assert result.best_score.objective_score == pytest.approx(0.08)

    def test_search_feature_groups_limits_pair_seeds_to_top_singletons(self) -> None:
        groups = [
            FeatureGroup(
                group_id=f"group_{index}",
                family="quant",
                stem=f"feature_{index}",
                feature_names=(f"feature_{index}",),
            )
            for index in range(20)
        ]
        call_counter = {"count": 0}

        def counting_scorer(feature_names: list[str]) -> SubsetEconomicScore:
            call_counter["count"] += 1
            return _score_subset(feature_names)

        config = FeatureSelectionConfig(
            search_beam_width=2,
            pair_seed_group_limit=4,
            null_bootstrap_count=0,
            max_active_matrix_gib=1.0,
        )

        search_feature_groups(
            root_groups=groups,
            scorer=counting_scorer,
            config=config,
            estimated_row_count=100,
        )

        assert call_counter["count"] < 100

    def test_search_feature_groups_returns_empty_when_no_subset_passes_gates(self) -> None:
        groups = [
            FeatureGroup(group_id="group_a", family="quant", stem="a", feature_names=("feature_a",)),
            FeatureGroup(group_id="group_b", family="quant", stem="b", feature_names=("feature_b",)),
        ]
        call_counter = {"count": 0}

        def invalid_scorer(feature_names: list[str]) -> SubsetEconomicScore:
            call_counter["count"] += 1
            objective = 0.02 if feature_names else 0.0
            fold_scores = (
                FoldEconomicScore(
                    index=1,
                    weight=1.0,
                    net_pnl_after_costs=objective,
                    alpha_over_benchmark_net=0.0,
                    turnover_annualized=0.40,
                    max_drawdown=-0.05,
                    daily_rank_ic_mean=0.0,
                    daily_rank_ic_ir=0.0,
                    daily_top_bottom_spread_mean=0.0,
                ),
                FoldEconomicScore(
                    index=2,
                    weight=1.0,
                    net_pnl_after_costs=objective,
                    alpha_over_benchmark_net=0.0,
                    turnover_annualized=0.40,
                    max_drawdown=-0.05,
                    daily_rank_ic_mean=0.0,
                    daily_rank_ic_ir=0.0,
                    daily_top_bottom_spread_mean=0.0,
                ),
            )
            return SubsetEconomicScore(
                feature_names=tuple(sorted(feature_names)),
                objective_score=objective,
                weighted_net_pnl_after_costs=objective,
                weighted_alpha_over_benchmark_net=0.0,
                weighted_turnover_annualized=0.40,
                weighted_max_drawdown=-0.05,
                positive_fold_share=1.0 if objective > 0.0 else 0.0,
                median_fold_net_pnl=objective,
                lower_quartile_fold_net_pnl=objective,
                is_valid=objective > 0.0,
                fold_scores=fold_scores,
                weighted_daily_rank_ic_mean=0.0,
                weighted_daily_rank_ic_ir=0.0,
                weighted_daily_top_bottom_spread_mean=0.0,
            )

        result = search_feature_groups(
            root_groups=groups,
            scorer=invalid_scorer,
            config=FeatureSelectionConfig(
                search_beam_width=2,
                null_bootstrap_count=0,
                max_active_matrix_gib=1.0,
            ),
            estimated_row_count=100,
        )

        assert result.selected_group_ids == []
        assert result.selected_feature_names == []
        assert result.best_score.objective_score == pytest.approx(0.0)
        assert call_counter["count"] == 6

    def test_search_feature_groups_discovers_three_way_interaction_without_valid_seed(self) -> None:
        groups = [
            FeatureGroup(group_id="group_a", family="quant", stem="a", feature_names=("feature_a",)),
            FeatureGroup(group_id="group_b", family="quant", stem="b", feature_names=("feature_b",)),
            FeatureGroup(group_id="group_c", family="quant", stem="c", feature_names=("feature_c",)),
            FeatureGroup(group_id="group_d", family="quant", stem="d", feature_names=("feature_d",)),
        ]

        def three_way_scorer(feature_names: list[str]) -> SubsetEconomicScore:
            active = frozenset(feature_names)
            objective_map: dict[frozenset[str], float] = {
                frozenset(): 0.0,
                frozenset({"feature_a"}): -0.01,
                frozenset({"feature_b"}): -0.01,
                frozenset({"feature_c"}): -0.01,
                frozenset({"feature_d"}): -0.02,
                frozenset({"feature_a", "feature_b"}): -0.005,
                frozenset({"feature_a", "feature_c"}): -0.005,
                frozenset({"feature_b", "feature_c"}): -0.005,
                frozenset({"feature_a", "feature_b", "feature_c"}): 0.06,
            }
            objective = objective_map.get(active, -0.03)
            alpha = objective if objective > 0.0 else 0.0
            fold_scores = (
                FoldEconomicScore(
                    index=1,
                    weight=1.0,
                    net_pnl_after_costs=objective,
                    alpha_over_benchmark_net=alpha,
                    turnover_annualized=0.40,
                    max_drawdown=-0.05,
                    daily_rank_ic_mean=objective,
                    daily_rank_ic_ir=objective,
                    daily_top_bottom_spread_mean=alpha,
                ),
                FoldEconomicScore(
                    index=2,
                    weight=1.0,
                    net_pnl_after_costs=objective,
                    alpha_over_benchmark_net=alpha,
                    turnover_annualized=0.40,
                    max_drawdown=-0.05,
                    daily_rank_ic_mean=objective,
                    daily_rank_ic_ir=objective,
                    daily_top_bottom_spread_mean=alpha,
                ),
            )
            return SubsetEconomicScore(
                feature_names=tuple(sorted(feature_names)),
                objective_score=objective,
                weighted_net_pnl_after_costs=objective,
                weighted_alpha_over_benchmark_net=alpha,
                weighted_turnover_annualized=0.40,
                weighted_max_drawdown=-0.05,
                positive_fold_share=1.0 if objective > 0.0 else 0.0,
                median_fold_net_pnl=objective,
                lower_quartile_fold_net_pnl=objective,
                is_valid=objective > 0.0,
                fold_scores=fold_scores,
                weighted_daily_rank_ic_mean=objective,
                weighted_daily_rank_ic_ir=objective,
                weighted_daily_top_bottom_spread_mean=alpha,
            )

        result = search_feature_groups(
            root_groups=groups,
            scorer=three_way_scorer,
            config=FeatureSelectionConfig(
                search_beam_width=2,
                pair_seed_group_limit=3,
                null_bootstrap_count=0,
                max_active_matrix_gib=1.0,
            ),
            estimated_row_count=100,
        )

        assert result.selected_feature_names == ["feature_a", "feature_b", "feature_c"]
        assert result.best_score.objective_score == pytest.approx(0.06)

    def test_search_feature_groups_diversifies_pair_seed_families(self, monkeypatch: pytest.MonkeyPatch) -> None:
        groups = [
            FeatureGroup(group_id="group_a1", family="quant", stem="a1", feature_names=("feature_a1",)),
            FeatureGroup(group_id="group_a2", family="quant", stem="a2", feature_names=("feature_a2",)),
            FeatureGroup(group_id="group_b1", family="macro", stem="b1", feature_names=("feature_b1",)),
        ]

        def diversified_pair_scorer(feature_names: list[str]) -> SubsetEconomicScore:
            active = frozenset(feature_names)
            objective_map: dict[frozenset[str], float] = {
                frozenset(): 0.0,
                frozenset({"feature_a1"}): 0.04,
                frozenset({"feature_a2"}): 0.03,
                frozenset({"feature_b1"}): 0.01,
                frozenset({"feature_a1", "feature_a2"}): -0.01,
                frozenset({"feature_a1", "feature_b1"}): 0.10,
                frozenset({"feature_a2", "feature_b1"}): 0.02,
            }
            objective = objective_map.get(active, -0.02)
            alpha = max(objective, 0.0)
            fold_scores = (
                FoldEconomicScore(
                    index=1,
                    weight=1.0,
                    net_pnl_after_costs=objective,
                    alpha_over_benchmark_net=alpha,
                    turnover_annualized=0.40,
                    max_drawdown=-0.05,
                    daily_rank_ic_mean=objective,
                    daily_rank_ic_ir=objective,
                    daily_top_bottom_spread_mean=alpha,
                ),
                FoldEconomicScore(
                    index=2,
                    weight=1.0,
                    net_pnl_after_costs=objective,
                    alpha_over_benchmark_net=alpha,
                    turnover_annualized=0.40,
                    max_drawdown=-0.05,
                    daily_rank_ic_mean=objective,
                    daily_rank_ic_ir=objective,
                    daily_top_bottom_spread_mean=alpha,
                ),
            )
            return SubsetEconomicScore(
                feature_names=tuple(sorted(feature_names)),
                objective_score=objective,
                weighted_net_pnl_after_costs=objective,
                weighted_alpha_over_benchmark_net=alpha,
                weighted_turnover_annualized=0.40,
                weighted_max_drawdown=-0.05,
                positive_fold_share=1.0 if objective > 0.0 else 0.0,
                median_fold_net_pnl=objective,
                lower_quartile_fold_net_pnl=objective,
                is_valid=objective > 0.0,
                fold_scores=fold_scores,
                weighted_daily_rank_ic_mean=objective,
                weighted_daily_rank_ic_ir=objective,
                weighted_daily_top_bottom_spread_mean=alpha,
            )

        def skip_neighbor_expansion(
            groups: list[FeatureGroup],
            anchor_state: tuple[str, ...],
            anchor_score: SubsetEconomicScore,
            best_state: tuple[str, ...],
            best_score: SubsetEconomicScore,
            singleton_scores: dict[tuple[str, ...], SubsetEconomicScore],
            scorer: Any,
            config: FeatureSelectionConfig,
            estimated_row_count: int,
            search_history: list[dict[str, object]],
        ) -> tuple[tuple[str, ...], SubsetEconomicScore]:
            del groups, anchor_state, anchor_score, singleton_scores, scorer, config, estimated_row_count, search_history
            return best_state, best_score

        monkeypatch.setattr(
            "core.src.meta_model.feature_selection.wrapper_search._expand_search_neighbors",
            skip_neighbor_expansion,
        )

        result = search_feature_groups(
            root_groups=groups,
            scorer=diversified_pair_scorer,
            config=FeatureSelectionConfig(
                search_beam_width=2,
                pair_seed_group_limit=2,
                null_bootstrap_count=0,
                max_active_matrix_gib=1.0,
            ),
            estimated_row_count=100,
        )

        assert result.selected_feature_names == ["feature_a1", "feature_b1"]
        assert result.best_score.objective_score == pytest.approx(0.10)

    def test_search_feature_groups_parallelizes_singleton_scoring(self, monkeypatch: pytest.MonkeyPatch) -> None:
        feature_names = [
            "feature_a",
            "feature_b",
            "feature_c",
            "feature_d",
            "feature_e",
            "feature_f",
        ]
        groups = [
            FeatureGroup(
                group_id=f"group_{feature_name}",
                family="quant",
                stem=feature_name,
                feature_names=(feature_name,),
            )
            for feature_name in feature_names
        ]

        state = {"active": 0, "max_active": 0}
        state_lock = Lock()

        def slow_positive_scorer(feature_names: list[str]) -> SubsetEconomicScore:
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.1)
            with state_lock:
                state["active"] -= 1
            return _score_subset(feature_names)

        def no_pair_seed_states(
            groups: list[FeatureGroup],
            singleton_scores: dict[tuple[str, ...], SubsetEconomicScore],
            estimated_row_count: int,
            config: FeatureSelectionConfig,
        ) -> list[tuple[str, ...]]:
            del groups, singleton_scores, estimated_row_count, config
            return []

        def skip_neighbor_expansion(
            groups: list[FeatureGroup],
            anchor_state: tuple[str, ...],
            anchor_score: SubsetEconomicScore,
            best_state: tuple[str, ...],
            best_score: SubsetEconomicScore,
            singleton_scores: dict[tuple[str, ...], SubsetEconomicScore],
            scorer: Any,
            config: FeatureSelectionConfig,
            estimated_row_count: int,
            search_history: list[dict[str, object]],
        ) -> tuple[tuple[str, ...], SubsetEconomicScore]:
            del groups, anchor_state, anchor_score, singleton_scores, scorer, config, estimated_row_count, search_history
            return best_state, best_score

        monkeypatch.setattr(
            "core.src.meta_model.feature_selection.wrapper_search._build_pair_seed_states",
            no_pair_seed_states,
        )
        monkeypatch.setattr(
            "core.src.meta_model.feature_selection.wrapper_search._expand_search_neighbors",
            skip_neighbor_expansion,
        )
        monkeypatch.setattr(
            "core.src.meta_model.feature_selection.wrapper_search._process_pool_available",
            lambda: False,
        )

        result = search_feature_groups(
            root_groups=groups,
            scorer=slow_positive_scorer,
            config=FeatureSelectionConfig(
                parallel_workers=4,
                state_evaluation_workers=4,
                null_bootstrap_count=0,
                max_active_matrix_gib=1.0,
            ),
            estimated_row_count=100,
        )

        assert state["max_active"] > 1
        assert result.selected_feature_names in (["feature_a"], ["feature_b"])

    def test_resolve_neighbor_evaluation_cap_broadens_exploration_without_accepted_state(self) -> None:
        config = FeatureSelectionConfig(
            search_beam_width=2,
            null_bootstrap_count=0,
            max_active_matrix_gib=1.0,
        )

        exploratory_cap = wrapper_search_module._resolve_neighbor_evaluation_cap(
            200,
            config=config,
            anchor_score=_score_subset(["feature_c"]),
            has_accepted_state=False,
        )
        accepted_cap = wrapper_search_module._resolve_neighbor_evaluation_cap(
            200,
            config=config,
            anchor_score=_score_subset(["feature_a"]),
            has_accepted_state=True,
        )

        assert exploratory_cap == 96
        assert accepted_cap == 8

    def test_refine_selected_feature_groups_disables_parallel_state_workers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root_groups = [
            FeatureGroup(
                group_id="group_a",
                family="quant",
                stem="a",
                feature_names=("feature_a1", "feature_a2"),
            ),
            FeatureGroup(
                group_id="group_b",
                family="quant",
                stem="b",
                feature_names=("feature_b1",),
            ),
        ]
        config = FeatureSelectionConfig(
            parallel_workers=4,
            null_bootstrap_count=0,
            max_active_matrix_gib=1.0,
        )
        current_result = WrapperSearchResult(
            selected_group_ids=["group_a", "group_b"],
            selected_feature_names=["feature_a1", "feature_a2", "feature_b1"],
            best_score=_score_subset(["feature_a1", "feature_a2", "feature_b1"]),
            search_history=[],
        )
        observed_state_workers: list[int | None] = []

        def fake_search_feature_groups(
            *,
            root_groups: list[FeatureGroup],
            scorer: Any,
            config: FeatureSelectionConfig,
            estimated_row_count: int,
        ) -> WrapperSearchResult:
            del root_groups, scorer, estimated_row_count
            observed_state_workers.append(config.state_evaluation_workers)
            return WrapperSearchResult(
                selected_group_ids=["feature_a1"],
                selected_feature_names=["feature_a1", "feature_b1"],
                best_score=_score_subset(["feature_a1", "feature_b1"]),
                search_history=[],
            )

        monkeypatch.setattr(
            "core.src.meta_model.feature_selection.wrapper_search.search_feature_groups",
            fake_search_feature_groups,
        )

        refine_selected_feature_groups(
            root_groups=root_groups,
            scorer=_score_subset,
            config=config,
            estimated_row_count=100,
            current_result=current_result,
        )

        assert observed_state_workers == [1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
