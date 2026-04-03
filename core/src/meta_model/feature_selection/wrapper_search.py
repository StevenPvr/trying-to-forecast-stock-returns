from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import logging
import multiprocessing as mp
import time

from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.grouping import FeatureGroup
from core.src.meta_model.feature_selection.objective import (
    SubsetEconomicScore,
    is_candidate_move_acceptable,
    passes_selection_gates,
)

FeatureSubsetScorer = Callable[[list[str]], SubsetEconomicScore]
LOGGER: logging.Logger = logging.getLogger(__name__)
_process_batch_groups: list[FeatureGroup] | None = None
_process_batch_scorer: FeatureSubsetScorer | None = None


@dataclass(frozen=True)
class WrapperSearchResult:
    selected_group_ids: list[str]
    selected_feature_names: list[str]
    best_score: SubsetEconomicScore
    search_history: list[dict[str, object]]


@dataclass(frozen=True)
class StateEvaluationResult:
    sequence_index: int
    state: tuple[str, ...]
    score: SubsetEconomicScore


def _process_pool_available() -> bool:
    return "fork" in mp.get_all_start_methods()


def _install_process_batch_context(
    groups: list[FeatureGroup],
    scorer: FeatureSubsetScorer,
) -> None:
    global _process_batch_groups, _process_batch_scorer
    _process_batch_groups = groups
    _process_batch_scorer = scorer


def _clear_process_batch_context() -> None:
    global _process_batch_groups, _process_batch_scorer
    _process_batch_groups = None
    _process_batch_scorer = None


def search_feature_groups(
    *,
    root_groups: list[FeatureGroup],
    scorer: FeatureSubsetScorer,
    config: FeatureSelectionConfig,
    estimated_row_count: int,
) -> WrapperSearchResult:
    start_time = time.perf_counter()
    empty_score = scorer([])
    search_history: list[dict[str, object]] = [_build_history_row((), empty_score, False)]
    singleton_states = _build_singleton_states(root_groups, estimated_row_count, config)
    state_worker_count = min(
        config.resolved_state_evaluation_workers(fold_count=config.fold_count),
        len(singleton_states) if singleton_states else 1,
    )
    LOGGER.info(
        "Wrapper search started: root_groups=%d | singleton_states=%d | state_workers=%d | beam_width=%d | pair_seed_group_limit=%d",
        len(root_groups),
        len(singleton_states),
        state_worker_count,
        config.search_beam_width,
        config.pair_seed_group_limit,
    )
    best_state = ()
    best_score = empty_score
    anchor_state = ()
    anchor_score = empty_score
    singleton_scores: dict[tuple[str, ...], SubsetEconomicScore] = {}
    evaluated_scores: dict[tuple[str, ...], SubsetEconomicScore] = {}
    singleton_results = _evaluate_state_batch(
        root_groups,
        singleton_states,
        scorer,
        phase_label="singleton",
        config=config,
    )
    for state_result in singleton_results:
        state_index = state_result.sequence_index
        state = state_result.state
        candidate_score = state_result.score
        singleton_scores[state] = candidate_score
        evaluated_scores[state] = candidate_score
        search_history.append(
            _build_history_row(
                state,
                candidate_score,
                passes_selection_gates(candidate_score, config),
            ),
        )
        if state_index == 1 or state_index == len(singleton_states) or state_index % max(1, config.search_beam_width * 4) == 0:
            LOGGER.info(
                "Wrapper search singleton %d/%d | groups=%d | features=%d | objective=%.6f | passes_gates=%s",
                state_index,
                len(singleton_states),
                len(state),
                len(candidate_score.feature_names),
                candidate_score.objective_score,
                passes_selection_gates(candidate_score, config),
            )
        if is_candidate_move_acceptable(candidate_score, best_score if best_state else None, config):
            best_state = state
            best_score = candidate_score
            anchor_state = state
            anchor_score = candidate_score
            LOGGER.info(
                "Wrapper search accepted singleton | groups=%d | features=%d | objective=%.6f",
                len(state),
                len(candidate_score.feature_names),
                candidate_score.objective_score,
            )
    pair_seed_states = _build_pair_seed_states(
        root_groups,
        singleton_scores,
        estimated_row_count,
        config,
    )
    LOGGER.info("Wrapper search pair seed states=%d", len(pair_seed_states))
    pair_seed_results = _evaluate_state_batch(
        root_groups,
        pair_seed_states,
        scorer,
        phase_label="pair seed",
        config=config,
    )
    for state_result in pair_seed_results:
        state_index = state_result.sequence_index
        state = state_result.state
        candidate_score = state_result.score
        evaluated_scores[state] = candidate_score
        search_history.append(
            _build_history_row(
                state,
                candidate_score,
                passes_selection_gates(candidate_score, config),
            ),
        )
        if state_index == 1 or state_index == len(pair_seed_states) or state_index % max(1, config.search_beam_width * 4) == 0:
            LOGGER.info(
                "Wrapper search pair seed %d/%d | groups=%d | features=%d | objective=%.6f | passes_gates=%s",
                state_index,
                len(pair_seed_states),
                len(state),
                len(candidate_score.feature_names),
                candidate_score.objective_score,
                passes_selection_gates(candidate_score, config),
            )
        if is_candidate_move_acceptable(candidate_score, best_score if best_state else None, config):
            best_state = state
            best_score = candidate_score
            anchor_state = state
            anchor_score = candidate_score
            LOGGER.info(
                "Wrapper search accepted pair seed | groups=%d | features=%d | objective=%.6f",
                len(state),
                len(candidate_score.feature_names),
                candidate_score.objective_score,
            )
    initial_states = [*singleton_states, *pair_seed_states]
    if initial_states and evaluated_scores:
        exploratory_state, exploratory_score = _fallback_best_state(evaluated_scores)
        if best_state:
            anchor_state = best_state
            anchor_score = best_score
        else:
            anchor_state = exploratory_state
            anchor_score = exploratory_score
        LOGGER.info(
            "Wrapper search exploratory anchor | groups=%d | features=%d | objective=%.6f | passes_gates=%s",
            len(anchor_state),
            len(anchor_score.feature_names),
            anchor_score.objective_score,
            passes_selection_gates(anchor_score, config),
        )
    best_state, best_score = _expand_search_neighbors(
        root_groups,
        anchor_state,
        anchor_score,
        best_state,
        best_score,
        singleton_scores,
        scorer,
        config,
        estimated_row_count,
        search_history,
    )
    LOGGER.info(
        "Wrapper search completed: groups=%d | features=%d | objective=%.6f | elapsed=%.2fs",
        len(best_state),
        len(best_score.feature_names),
        best_score.objective_score,
        time.perf_counter() - start_time,
    )
    return WrapperSearchResult(
        selected_group_ids=list(best_state),
        selected_feature_names=_expand_state_features(root_groups, best_state),
        best_score=best_score,
        search_history=search_history,
    )


def refine_selected_feature_groups(
    *,
    root_groups: list[FeatureGroup],
    scorer: FeatureSubsetScorer,
    config: FeatureSelectionConfig,
    estimated_row_count: int,
    current_result: WrapperSearchResult,
) -> WrapperSearchResult:
    start_time = time.perf_counter()
    current_score = current_result.best_score
    current_feature_names = list(current_result.selected_feature_names)
    search_history = list(current_result.search_history)
    selected_groups = [group for group in root_groups if group.group_id in current_result.selected_group_ids]
    for group in selected_groups:
        if len(group.feature_names) <= 1:
            continue
        refinement_config = replace(config, state_evaluation_workers=1)
        LOGGER.info(
            "Wrapper refinement group=%s | group_features=%d | current_selected_features=%d",
            group.group_id,
            len(group.feature_names),
            len(current_feature_names),
        )
        fixed_feature_names = [name for name in current_feature_names if name not in set(group.feature_names)]
        singleton_groups = _build_singleton_groups(group)
        def local_scorer(feature_names: list[str]) -> SubsetEconomicScore:
            combined_feature_names = sorted(set([*fixed_feature_names, *feature_names]))
            return scorer(combined_feature_names)
        local_result = search_feature_groups(
            root_groups=singleton_groups,
            scorer=local_scorer,
            config=refinement_config,
            estimated_row_count=estimated_row_count,
        )
        if is_candidate_move_acceptable(local_result.best_score, current_score, config):
            current_score = local_result.best_score
            current_feature_names = list(current_score.feature_names)
            search_history.extend(local_result.search_history)
            LOGGER.info(
                "Wrapper refinement accepted group=%s | selected_features=%d | objective=%.6f",
                group.group_id,
                len(current_feature_names),
                current_score.objective_score,
            )
    LOGGER.info(
        "Wrapper refinement completed: selected_features=%d | objective=%.6f | elapsed=%.2fs",
        len(current_feature_names),
        current_score.objective_score,
        time.perf_counter() - start_time,
    )
    return WrapperSearchResult(
        selected_group_ids=current_result.selected_group_ids,
        selected_feature_names=sorted(current_feature_names),
        best_score=current_score,
        search_history=search_history,
    )


def _build_singleton_states(
    groups: list[FeatureGroup],
    estimated_row_count: int,
    config: FeatureSelectionConfig,
) -> list[tuple[str, ...]]:
    group_ids = [group.group_id for group in groups]
    return [
        state
        for state in (tuple([group_id]) for group_id in group_ids)
        if _state_fits_memory(groups, state, estimated_row_count, config.max_active_matrix_gib)
    ]


def _build_pair_seed_states(
    groups: list[FeatureGroup],
    singleton_scores: dict[tuple[str, ...], SubsetEconomicScore],
    estimated_row_count: int,
    config: FeatureSelectionConfig,
) -> list[tuple[str, ...]]:
    ranked_singletons = sorted(
        singleton_scores.items(),
        key=lambda item: (
            item[1].objective_score,
            item[1].weighted_net_pnl_after_costs,
            item[0],
        ),
        reverse=True,
    )
    top_group_ids = _select_pair_seed_group_ids(
        groups,
        ranked_singletons,
        pair_seed_group_limit=config.pair_seed_group_limit,
    )
    pair_states = [
        tuple(sorted((left_group_id, right_group_id)))
        for left_index, left_group_id in enumerate(top_group_ids)
        for right_group_id in top_group_ids[left_index + 1:]
    ]
    return [
        state
        for state in pair_states
        if _state_fits_memory(groups, state, estimated_row_count, config.max_active_matrix_gib)
    ]


def _expand_search_neighbors(
    groups: list[FeatureGroup],
    anchor_state: tuple[str, ...],
    anchor_score: SubsetEconomicScore,
    best_state: tuple[str, ...],
    best_score: SubsetEconomicScore,
    singleton_scores: dict[tuple[str, ...], SubsetEconomicScore],
    scorer: FeatureSubsetScorer,
    config: FeatureSelectionConfig,
    estimated_row_count: int,
    search_history: list[dict[str, object]],
) -> tuple[tuple[str, ...], SubsetEconomicScore]:
    current_state = anchor_state
    current_score = anchor_score
    current_best_state = best_state
    current_best_score = best_score
    group_priority = {
        state[0]: score.objective_score
        for state, score in singleton_scores.items()
    }
    iteration_index = 0
    while True:
        iteration_index += 1
        neighbors = _build_neighbor_states(
            groups,
            current_state,
            estimated_row_count,
            config,
            group_priority,
        )
        LOGGER.info(
            "Wrapper search iteration %d | current_groups=%d | current_features=%d | neighbors=%d | evaluating_neighbors=%d | objective=%.6f",
            iteration_index,
            len(current_state),
            len(current_score.feature_names),
            len(neighbors),
            _resolve_neighbor_evaluation_cap(
                len(neighbors),
                config=config,
                anchor_score=current_score,
                has_accepted_state=bool(current_best_state),
            ),
            current_score.objective_score,
        )
        improved_state = current_state
        improved_score = current_score
        accepted_neighbor_found = False
        evaluation_cap = _resolve_neighbor_evaluation_cap(
            len(neighbors),
            config=config,
            anchor_score=current_score,
            has_accepted_state=bool(current_best_state),
        )
        neighbor_results = _evaluate_state_batch(
            groups,
            neighbors[:evaluation_cap],
            scorer,
            phase_label="neighbor",
            config=config,
        )
        for state_result in neighbor_results:
            neighbor_index = state_result.sequence_index
            neighbor_state = state_result.state
            candidate_score = state_result.score
            search_history.append(
                _build_history_row(
                    neighbor_state,
                    candidate_score,
                    passes_selection_gates(candidate_score, config),
                ),
            )
            if neighbor_index == 1 or neighbor_index == evaluation_cap or neighbor_index % max(1, config.search_beam_width * 2) == 0:
                LOGGER.info(
                    "Wrapper search neighbor %d/%d | groups=%d | features=%d | objective=%.6f | passes_gates=%s",
                    neighbor_index,
                    evaluation_cap,
                    len(neighbor_state),
                    len(candidate_score.feature_names),
                    candidate_score.objective_score,
                    passes_selection_gates(candidate_score, config),
                )
            comparison_score = current_best_score if current_best_state else None
            if is_candidate_move_acceptable(candidate_score, comparison_score, config):
                improved_state = neighbor_state
                improved_score = candidate_score
                current_best_state = neighbor_state
                current_best_score = candidate_score
                accepted_neighbor_found = True
                LOGGER.info(
                    "Wrapper search accepted neighbor | groups=%d | features=%d | objective=%.6f",
                    len(neighbor_state),
                    len(candidate_score.feature_names),
                    candidate_score.objective_score,
                )
        if not accepted_neighbor_found or improved_state == current_state:
            LOGGER.info("Wrapper search converged after %d iterations", iteration_index)
            return current_best_state, current_best_score
        current_state = improved_state
        current_score = improved_score


def _select_pair_seed_group_ids(
    groups: list[FeatureGroup],
    ranked_singletons: list[tuple[tuple[str, ...], SubsetEconomicScore]],
    *,
    pair_seed_group_limit: int,
) -> list[str]:
    if not ranked_singletons:
        return []
    limit = max(2, min(pair_seed_group_limit, len(ranked_singletons)))
    group_family_by_id = {group.group_id: group.family for group in groups}
    family_order: list[str] = []
    family_ranked_group_ids: dict[str, list[str]] = {}
    for state, _ in ranked_singletons:
        group_id = state[0]
        group_family = group_family_by_id.get(group_id, "other")
        if group_family not in family_ranked_group_ids:
            family_order.append(group_family)
            family_ranked_group_ids[group_family] = []
        family_ranked_group_ids[group_family].append(group_id)
    diversified_group_ids: list[str] = []
    family_rank = 0
    while len(diversified_group_ids) < limit:
        added_group = False
        for group_family in family_order:
            family_group_ids = family_ranked_group_ids[group_family]
            if family_rank >= len(family_group_ids):
                continue
            diversified_group_ids.append(family_group_ids[family_rank])
            added_group = True
            if len(diversified_group_ids) >= limit:
                break
        if not added_group:
            break
        family_rank += 1
    return diversified_group_ids


def _resolve_neighbor_evaluation_cap(
    neighbor_count: int,
    *,
    config: FeatureSelectionConfig,
    anchor_score: SubsetEconomicScore,
    has_accepted_state: bool,
) -> int:
    base_cap = min(neighbor_count, config.search_beam_width * 4)
    if neighbor_count <= base_cap:
        return neighbor_count
    if has_accepted_state or passes_selection_gates(anchor_score, config):
        return base_cap
    exploratory_cap = max(config.search_beam_width * 16, 96)
    return min(neighbor_count, exploratory_cap)


def _fallback_best_state(
    evaluated_scores: dict[tuple[str, ...], SubsetEconomicScore],
) -> tuple[tuple[str, ...], SubsetEconomicScore]:
    return max(
        evaluated_scores.items(),
        key=lambda item: (item[1].objective_score, len(item[0]), item[0]),
    )


def _evaluate_state_batch(
    groups: list[FeatureGroup],
    states: list[tuple[str, ...]],
    scorer: FeatureSubsetScorer,
    *,
    phase_label: str,
    config: FeatureSelectionConfig,
) -> list[StateEvaluationResult]:
    if not states:
        return []
    worker_count = min(
        config.resolved_state_evaluation_workers(fold_count=config.fold_count),
        len(states),
    )
    if worker_count <= 1:
        return [
            StateEvaluationResult(
                sequence_index=state_index,
                state=state,
                score=scorer(_expand_state_features(groups, state)),
            )
            for state_index, state in enumerate(states, start=1)
        ]
    chunk_size = _resolve_state_chunk_size(len(states), worker_count, config)
    indexed_states = [(state_index, state) for state_index, state in enumerate(states, start=1)]
    indexed_state_chunks = _chunk_indexed_states(indexed_states, chunk_size)
    if _process_pool_available():
        LOGGER.info(
            "Wrapper search %s backend=process | workers=%d | chunk_size=%d",
            phase_label,
            worker_count,
            chunk_size,
        )
        try:
            return _evaluate_state_chunks_in_process_pool(
                groups,
                indexed_state_chunks,
                scorer,
                worker_count=worker_count,
                phase_label=phase_label,
                config=config,
                total_state_count=len(states),
            )
        except (NotImplementedError, OSError, PermissionError) as error:
            LOGGER.warning(
                "Wrapper search %s process backend unavailable (%s); falling back to threads.",
                phase_label,
                error,
            )
    LOGGER.info(
        "Wrapper search %s backend=thread | workers=%d | chunk_size=%d",
        phase_label,
        worker_count,
        chunk_size,
    )
    return _evaluate_state_chunks_in_thread_pool(
        groups,
        indexed_state_chunks,
        scorer,
        worker_count=worker_count,
        phase_label=phase_label,
        config=config,
        total_state_count=len(states),
    )


def _resolve_state_chunk_size(
    state_count: int,
    worker_count: int,
    config: FeatureSelectionConfig,
) -> int:
    target_chunk_count = max(worker_count * max(2, config.search_beam_width), worker_count)
    return max(1, min(16, (state_count + target_chunk_count - 1) // target_chunk_count))


def _chunk_indexed_states(
    indexed_states: list[tuple[int, tuple[str, ...]]],
    chunk_size: int,
) -> list[list[tuple[int, tuple[str, ...]]]]:
    return [
        indexed_states[start_index:start_index + chunk_size]
        for start_index in range(0, len(indexed_states), chunk_size)
    ]


def _evaluate_state_chunks_in_process_pool(
    groups: list[FeatureGroup],
    indexed_state_chunks: list[list[tuple[int, tuple[str, ...]]]],
    scorer: FeatureSubsetScorer,
    *,
    worker_count: int,
    phase_label: str,
    config: FeatureSelectionConfig,
    total_state_count: int,
) -> list[StateEvaluationResult]:
    process_context = mp.get_context("fork")
    _install_process_batch_context(groups, scorer)
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=process_context) as executor:
            futures = [
                executor.submit(_score_state_chunk_in_process, indexed_state_chunk)
                for indexed_state_chunk in indexed_state_chunks
            ]
            return _collect_state_chunk_results(
                futures,
                phase_label=phase_label,
                config=config,
                total_state_count=total_state_count,
            )
    finally:
        _clear_process_batch_context()


def _evaluate_state_chunks_in_thread_pool(
    groups: list[FeatureGroup],
    indexed_state_chunks: list[list[tuple[int, tuple[str, ...]]]],
    scorer: FeatureSubsetScorer,
    *,
    worker_count: int,
    phase_label: str,
    config: FeatureSelectionConfig,
    total_state_count: int,
) -> list[StateEvaluationResult]:
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="wrapper-search") as executor:
        futures = [
            executor.submit(_score_state_chunk, groups, indexed_state_chunk, scorer)
            for indexed_state_chunk in indexed_state_chunks
        ]
        return _collect_state_chunk_results(
            futures,
            phase_label=phase_label,
            config=config,
            total_state_count=total_state_count,
        )


def _collect_state_chunk_results(
    futures: list[Future[list[StateEvaluationResult]]],
    *,
    phase_label: str,
    config: FeatureSelectionConfig,
    total_state_count: int,
) -> list[StateEvaluationResult]:
    ordered_results: dict[int, StateEvaluationResult] = {}
    completed_state_count = 0
    progress_step = max(1, config.search_beam_width * 4)
    next_progress_marker = 1
    for future in as_completed(futures):
        chunk_results = future.result()
        completed_state_count += len(chunk_results)
        for result in chunk_results:
            ordered_results[result.sequence_index] = result
        if completed_state_count >= next_progress_marker or completed_state_count == total_state_count:
            LOGGER.info(
                "Wrapper search %s progress %d/%d",
                phase_label,
                min(completed_state_count, total_state_count),
                total_state_count,
            )
            next_progress_marker = min(total_state_count, next_progress_marker + progress_step)
    return [ordered_results[state_index] for state_index in sorted(ordered_results)]


def _score_state_chunk_in_process(
    indexed_states: list[tuple[int, tuple[str, ...]]],
) -> list[StateEvaluationResult]:
    if _process_batch_groups is None or _process_batch_scorer is None:
        raise RuntimeError("Wrapper search process batch context is not installed.")
    return _score_state_chunk(_process_batch_groups, indexed_states, _process_batch_scorer)


def _score_state_chunk(
    groups: list[FeatureGroup],
    indexed_states: list[tuple[int, tuple[str, ...]]],
    scorer: FeatureSubsetScorer,
) -> list[StateEvaluationResult]:
    return [
        StateEvaluationResult(
            sequence_index=state_index,
            state=state,
            score=scorer(_expand_state_features(groups, state)),
        )
        for state_index, state in indexed_states
    ]


def _build_neighbor_states(
    groups: list[FeatureGroup],
    current_state: tuple[str, ...],
    estimated_row_count: int,
    config: FeatureSelectionConfig,
    group_priority: dict[str, float],
) -> list[tuple[str, ...]]:
    current_ids = set(current_state)
    all_ids = [group.group_id for group in groups]
    additions: set[tuple[str, ...]] = set()
    replacements: set[tuple[str, ...]] = set()
    removals: set[tuple[str, ...]] = set()
    for group_id in all_ids:
        if group_id not in current_ids:
            additions.add(tuple(sorted([*current_ids, group_id])))
    for group_id in current_ids:
        removals.add(tuple(sorted(current_ids.difference({group_id}))))
        for replacement_id in all_ids:
            if replacement_id in current_ids:
                continue
            replaced = current_ids.difference({group_id}).union({replacement_id})
            replacements.add(tuple(sorted(replaced)))
    filtered_neighbors = [
        state
        for state in _rank_neighbor_states(
            additions,
            replacements,
            removals,
            group_priority,
        )
        if state
        and _state_fits_memory(groups, state, estimated_row_count, config.max_active_matrix_gib)
    ]
    return filtered_neighbors


def _rank_neighbor_states(
    additions: set[tuple[str, ...]],
    replacements: set[tuple[str, ...]],
    removals: set[tuple[str, ...]],
    group_priority: dict[str, float],
) -> list[tuple[str, ...]]:
    ordered_states: list[tuple[str, ...]] = []
    for states in (additions, replacements, removals):
        ordered_states.extend(
            sorted(
                states,
                key=lambda state: (
                    -_state_priority(state, group_priority),
                    -len(state),
                    state,
                ),
            ),
        )
    return ordered_states


def _state_priority(state: tuple[str, ...], group_priority: dict[str, float]) -> float:
    return float(sum(group_priority.get(group_id, 0.0) for group_id in state))


def _state_fits_memory(
    groups: list[FeatureGroup],
    state: tuple[str, ...],
    estimated_row_count: int,
    max_active_matrix_gib: float,
) -> bool:
    active_feature_count = len(_expand_state_features(groups, state))
    estimated_bytes = estimated_row_count * active_feature_count * 4
    max_bytes = max_active_matrix_gib * (1024.0 ** 3)
    return estimated_bytes <= max_bytes


def _expand_state_features(groups: list[FeatureGroup], state: tuple[str, ...]) -> list[str]:
    active_group_ids = set(state)
    active_features = [
        feature_name
        for group in groups
        if group.group_id in active_group_ids
        for feature_name in group.feature_names
    ]
    return sorted(set(active_features))


def _build_history_row(
    state: tuple[str, ...],
    score: SubsetEconomicScore,
    passes_gates: bool,
) -> dict[str, object]:
    return {
        "group_ids": list(state),
        "feature_names": list(score.feature_names),
        "objective_score": score.objective_score,
        "weighted_net_pnl_after_costs": score.weighted_net_pnl_after_costs,
        "weighted_alpha_over_benchmark_net": score.weighted_alpha_over_benchmark_net,
        "weighted_turnover_annualized": score.weighted_turnover_annualized,
        "weighted_max_drawdown": score.weighted_max_drawdown,
        "weighted_daily_rank_ic_mean": score.weighted_daily_rank_ic_mean,
        "weighted_daily_rank_ic_ir": score.weighted_daily_rank_ic_ir,
        "weighted_daily_top_bottom_spread_mean": score.weighted_daily_top_bottom_spread_mean,
        "positive_fold_share": score.positive_fold_share,
        "median_fold_net_pnl": score.median_fold_net_pnl,
        "lower_quartile_fold_net_pnl": score.lower_quartile_fold_net_pnl,
        "is_valid": score.is_valid,
        "passes_selection_gates": passes_gates,
    }


def _build_singleton_groups(group: FeatureGroup) -> list[FeatureGroup]:
    return [
        FeatureGroup(
            group_id=f"{group.group_id}:feature:{index}",
            family=group.family,
            stem=group.stem,
            feature_names=(feature_name,),
            level=group.level + 1,
            parent_group_id=group.group_id,
        )
        for index, feature_name in enumerate(group.feature_names, start=1)
    ]


__all__ = [
    "FeatureSubsetScorer",
    "WrapperSearchResult",
    "refine_selected_feature_groups",
    "search_feature_groups",
]
