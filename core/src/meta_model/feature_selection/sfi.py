from __future__ import annotations

from dataclasses import replace
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import logging
import multiprocessing as mp
from collections.abc import Callable
from typing import Any, cast

import pandas as pd

from core.src.meta_model.feature_selection.cache import FeatureSelectionRuntimeCache
from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.cv import SelectionFold
from core.src.meta_model.feature_selection.grouping import infer_feature_family, normalize_feature_stem
from core.src.meta_model.feature_selection.io import FeatureSelectionMetadata
from core.src.meta_model.feature_selection.objective import SubsetEconomicScore
from core.src.meta_model.feature_selection.scoring import BacktestFeatureSubsetScorer

LOGGER: logging.Logger = logging.getLogger(__name__)
SFI_PROGRESS_HEARTBEAT_SECONDS: float = 10.0
_PROCESS_CACHE: FeatureSelectionRuntimeCache | None = None
_PROCESS_SCORER: BacktestFeatureSubsetScorer | None = None
_PROCESS_CONFIG: FeatureSelectionConfig | None = None


def build_sfi_score_frame(
    cache: FeatureSelectionRuntimeCache,
    feature_names: list[str],
    scorer: Callable[[list[str]], SubsetEconomicScore],
    config: FeatureSelectionConfig,
) -> pd.DataFrame:
    worker_count = _resolve_sfi_worker_count(config, len(feature_names))
    LOGGER.info(
        "Feature selection SFI started: features=%d | workers=%d | coverage_threshold=%.2f",
        len(feature_names),
        worker_count,
        config.sfi_min_coverage_fraction,
    )
    if worker_count == 1 or len(feature_names) <= 1:
        rows = [_score_single_feature(cache, scorer, feature_name, config) for feature_name in feature_names]
    elif _can_use_process_pool(cache, scorer):
        try:
            rows = _score_features_parallel_process(
                cast(FeatureSelectionRuntimeCache, cache),
                cast(BacktestFeatureSubsetScorer, scorer),
                feature_names,
                config,
                worker_count,
            )
        except (PermissionError, OSError, ValueError, BrokenProcessPool) as exc:
            LOGGER.warning(
                "Feature selection SFI multiprocessing unavailable; falling back to threaded scoring: %s",
                exc,
            )
            rows = _score_features_parallel(cache, scorer, feature_names, config, worker_count)
    else:
        rows = _score_features_parallel(cache, scorer, feature_names, config, worker_count)
    score_frame = pd.DataFrame(rows).sort_values(
        ["objective_score", "daily_rank_ic_mean", "coverage_fraction", "feature_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    LOGGER.info(
        "Feature selection SFI completed: survivors=%d | under_coverage=%d",
        int(cast(pd.Series, score_frame["passes_sfi"]).sum()),
        int((~cast(pd.Series, score_frame["passes_sfi"])).sum()),
    )
    return score_frame


def _score_features_parallel(
    cache: FeatureSelectionRuntimeCache,
    scorer: Callable[[list[str]], SubsetEconomicScore],
    feature_names: list[str],
    config: FeatureSelectionConfig,
    worker_count: int,
) -> list[dict[str, object]]:
    thread_scorer = _resolve_thread_sfi_scorer(scorer, config, worker_count)
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="feature-sfi") as executor:
        future_map = {
            executor.submit(_score_single_feature, cache, thread_scorer, feature_name, config): feature_name
            for feature_name in feature_names
        }
        LOGGER.info(
            "Feature selection SFI tasks submitted: features=%d | workers=%d | backend=thread",
            len(feature_names),
            worker_count,
        )
        rows_by_name = _collect_feature_rows_from_futures(
            future_map,
            total_count=len(feature_names),
        )
    return [rows_by_name[feature_name] for feature_name in sorted(rows_by_name)]


def _score_features_parallel_process(
    cache: FeatureSelectionRuntimeCache,
    scorer: BacktestFeatureSubsetScorer,
    feature_names: list[str],
    config: FeatureSelectionConfig,
    worker_count: int,
) -> list[dict[str, object]]:
    max_cache_gib = min(0.5, max(0.25, float(cache._max_cache_bytes) / float(1024.0 ** 3 * max(1, worker_count))))
    worker_config = _build_sfi_worker_config(config, worker_count)
    process_context = mp.get_context("spawn")
    LOGGER.info(
        "Feature selection SFI multiprocessing enabled: workers=%d | worker_parallel_workers=%d | worker_cache_gib=%.2f",
        worker_count,
        worker_config.parallel_workers,
        max_cache_gib,
    )
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=process_context,
        initializer=_initialize_sfi_process_worker,
        initargs=(cache._dataset_path, cache._metadata, scorer._folds, worker_config, max_cache_gib),
    ) as executor:
        future_map = {
            executor.submit(_score_single_feature_process, feature_name): feature_name
            for feature_name in feature_names
        }
        LOGGER.info(
            "Feature selection SFI tasks submitted: features=%d | workers=%d | backend=process",
            len(feature_names),
            worker_count,
        )
        rows_by_name = _collect_feature_rows_from_futures(
            future_map,
            total_count=len(feature_names),
        )
    return [rows_by_name[feature_name] for feature_name in sorted(rows_by_name)]


def _can_use_process_pool(
    cache: FeatureSelectionRuntimeCache,
    scorer: Callable[[list[str]], SubsetEconomicScore],
) -> bool:
    return (
        isinstance(cache, FeatureSelectionRuntimeCache)
        and isinstance(scorer, BacktestFeatureSubsetScorer)
        and cache._dataset_path is not None
    )


def _resolve_sfi_worker_count(
    config: FeatureSelectionConfig,
    feature_count: int,
) -> int:
    return min(max(1, config.parallel_workers), max(1, feature_count))


def _build_sfi_worker_config(
    config: FeatureSelectionConfig,
    worker_count: int,
) -> FeatureSelectionConfig:
    per_worker_parallel_workers = max(1, config.parallel_workers // max(1, worker_count))
    return replace(
        config,
        parallel_workers=per_worker_parallel_workers,
        state_evaluation_workers=1,
    )


def _resolve_thread_sfi_scorer(
    scorer: Callable[[list[str]], SubsetEconomicScore],
    config: FeatureSelectionConfig,
    worker_count: int,
) -> Callable[[list[str]], SubsetEconomicScore]:
    if not isinstance(scorer, BacktestFeatureSubsetScorer):
        return scorer
    worker_config = _build_sfi_worker_config(config, worker_count)
    LOGGER.info(
        "Feature selection SFI thread worker scorer configured: outer_workers=%d | inner_parallel_workers=%d",
        worker_count,
        worker_config.parallel_workers,
    )
    return BacktestFeatureSubsetScorer(
        scorer._cache,
        scorer._folds,
        worker_config,
        backtest_config=scorer._backtest_config,
    )


def _collect_feature_rows_from_futures(
    future_map: dict[Future[dict[str, object]], str],
    *,
    total_count: int,
) -> dict[str, dict[str, object]]:
    rows_by_name: dict[str, dict[str, object]] = {}
    completed_count = 0
    pending_futures: set[Future[dict[str, object]]] = set(future_map)
    while pending_futures:
        done_futures, pending_futures = wait(
            pending_futures,
            timeout=SFI_PROGRESS_HEARTBEAT_SECONDS,
            return_when=FIRST_COMPLETED,
        )
        if not done_futures:
            LOGGER.info(
                "Feature selection SFI heartbeat | completed=%d/%d | remaining=%d",
                completed_count,
                total_count,
                len(pending_futures),
            )
            continue
        for future in done_futures:
            row = future.result()
            rows_by_name[str(row["feature_name"])] = row
            completed_count += 1
            LOGGER.info(
                "Feature selection SFI progress %d/%d | latest_feature=%s | latest_objective=%.6f | latest_coverage=%.4f | passes_sfi=%s",
                completed_count,
                total_count,
                str(row["feature_name"]),
                float(cast(float, row["objective_score"])),
                float(cast(float, row["coverage_fraction"])),
                bool(cast(bool, row["passes_sfi"])),
            )
    return rows_by_name


def _initialize_sfi_process_worker(
    dataset_path: Any,
    metadata: FeatureSelectionMetadata,
    folds: list[SelectionFold],
    config: FeatureSelectionConfig,
    max_cache_gib: float,
) -> None:
    global _PROCESS_CACHE, _PROCESS_SCORER, _PROCESS_CONFIG
    resolved_dataset_path = cast(Any, dataset_path)
    _PROCESS_CONFIG = config
    _PROCESS_CACHE = FeatureSelectionRuntimeCache(
        resolved_dataset_path,
        metadata,
        random_seed=config.random_seed,
        max_cache_gib=max_cache_gib,
    )
    _PROCESS_SCORER = BacktestFeatureSubsetScorer(_PROCESS_CACHE, folds, config)


def _score_single_feature_process(feature_name: str) -> dict[str, object]:
    if _PROCESS_CACHE is None or _PROCESS_SCORER is None or _PROCESS_CONFIG is None:
        raise RuntimeError("SFI process worker was not initialized correctly.")
    return _score_single_feature(_PROCESS_CACHE, _PROCESS_SCORER, feature_name, _PROCESS_CONFIG)


def _score_single_feature(
    cache: FeatureSelectionRuntimeCache,
    scorer: Callable[[list[str]], SubsetEconomicScore],
    feature_name: str,
    config: FeatureSelectionConfig,
) -> dict[str, object]:
    coverage_fraction = cache.feature_coverage_fraction(feature_name)
    subset_score = scorer([feature_name]) if coverage_fraction > 0.0 else _empty_subset_score(feature_name)
    passes_coverage = coverage_fraction >= config.sfi_min_coverage_fraction
    passes_sfi = passes_coverage and subset_score.objective_score > 0.0
    drop_reason = _resolve_sfi_drop_reason(passes_coverage, subset_score)
    return {
        "feature_name": feature_name,
        "feature_family": infer_feature_family(feature_name),
        "feature_stem": normalize_feature_stem(feature_name),
        "coverage_fraction": coverage_fraction,
        "objective_score": subset_score.objective_score,
        "daily_rank_ic_mean": subset_score.weighted_daily_rank_ic_mean,
        "daily_rank_ic_ir": subset_score.weighted_daily_rank_ic_ir,
        "daily_top_bottom_spread_mean": subset_score.weighted_daily_top_bottom_spread_mean,
        "positive_fold_share": subset_score.positive_fold_share,
        "passes_coverage": passes_coverage,
        "passes_sfi": passes_sfi,
        "sfi_drop_reason": drop_reason,
    }


def _empty_subset_score(feature_name: str) -> SubsetEconomicScore:
    return SubsetEconomicScore(
        feature_names=(feature_name,),
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
        weighted_daily_rank_ic_mean=0.0,
        weighted_daily_rank_ic_ir=0.0,
        weighted_daily_top_bottom_spread_mean=0.0,
    )


def _resolve_sfi_drop_reason(
    passes_coverage: bool,
    subset_score: SubsetEconomicScore,
) -> str:
    if not passes_coverage:
        return "low_coverage"
    if subset_score.objective_score <= 0.0:
        return "non_positive_sfi"
    return "retained"


__all__ = ["build_sfi_score_frame"]
