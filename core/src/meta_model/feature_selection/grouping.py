from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import math
import re
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from core.src.meta_model.features_engineering.config import (
    CALENDAR_FEATURE_PREFIX,
    COMPANY_FEATURE_PREFIX,
    CROSS_ASSET_FEATURE_PREFIX,
    DEEP_FEATURE_PREFIX,
    MACRO_FEATURE_PREFIX,
    QUANT_FEATURE_PREFIX,
    SENTIMENT_FEATURE_PREFIX,
    TA_FEATURE_PREFIX,
)

WINDOW_SUFFIX_PATTERN = re.compile(r"_(?:lag_)?\d+d?$")
LOGGER: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.src.meta_model.feature_selection.cache import FeatureSelectionRuntimeCache


@dataclass(frozen=True)
class FeatureBucketKey:
    family: str
    stem: str


@dataclass(frozen=True)
class FeatureGroup:
    group_id: str
    family: str
    stem: str
    feature_names: tuple[str, ...]
    level: int = 0
    parent_group_id: str | None = None


@dataclass(frozen=True)
class BucketGroupingResult:
    bucket_index: int
    bucket_key: FeatureBucketKey
    bucket_feature_count: int
    bucket_groups: list[FeatureGroup]
    manifest_rows: list[dict[str, object]]


def infer_feature_family(column_name: str) -> str:
    if column_name.startswith("xtb_"):
        return "broker"
    if column_name.startswith("sector_"):
        return "sector"
    if column_name.startswith("open_"):
        return "open"
    if column_name.startswith("earnings_"):
        return "earnings"
    if column_name.startswith("signal_"):
        return "signal"
    if column_name.startswith(TA_FEATURE_PREFIX):
        return "ta"
    if column_name.startswith(QUANT_FEATURE_PREFIX):
        return "quant"
    if column_name.startswith(DEEP_FEATURE_PREFIX):
        return "deep"
    if column_name.startswith(CALENDAR_FEATURE_PREFIX):
        return "calendar"
    if column_name.startswith(SENTIMENT_FEATURE_PREFIX):
        return "sentiment"
    if column_name.startswith(MACRO_FEATURE_PREFIX):
        return "macro"
    if column_name.startswith(CROSS_ASSET_FEATURE_PREFIX):
        return "cross_asset"
    if column_name.startswith(COMPANY_FEATURE_PREFIX):
        return "company"
    if column_name.startswith("stock_"):
        return "stock"
    return "other"


def normalize_feature_stem(feature_name: str) -> str:
    normalized = WINDOW_SUFFIX_PATTERN.sub("", feature_name)
    return normalized.rstrip("_")


def build_feature_buckets(feature_names: list[str]) -> dict[FeatureBucketKey, list[str]]:
    buckets: dict[FeatureBucketKey, list[str]] = defaultdict(list)
    for feature_name in sorted(feature_names):
        bucket_key = FeatureBucketKey(
            family=infer_feature_family(feature_name),
            stem=normalize_feature_stem(feature_name),
        )
        buckets[bucket_key].append(feature_name)
    return dict(buckets)


def partition_feature_bucket(
    bucket_key: FeatureBucketKey,
    sampled_frame: pd.DataFrame,
    *,
    max_group_size: int,
    level: int = 0,
    parent_group_id: str | None = None,
) -> list[FeatureGroup]:
    raw_feature_names = cast(list[object], sampled_frame.columns.tolist())
    feature_names = [str(feature_name) for feature_name in raw_feature_names]
    if len(feature_names) <= max_group_size:
        return [_build_feature_group(bucket_key, feature_names, 1, level, parent_group_id)]
    correlation_matrix = _build_abs_spearman_matrix(sampled_frame)
    correlation_cutoff = _derive_correlation_cutoff(correlation_matrix)
    components = _build_components(correlation_matrix, correlation_cutoff)
    if len(components) == 1:
        return _partition_large_component(bucket_key, components[0], correlation_matrix, max_group_size, level, parent_group_id)
    return _build_component_groups(bucket_key, components, correlation_matrix, max_group_size, level, parent_group_id)


def build_feature_groups(
    cache: FeatureSelectionRuntimeCache,
    feature_names: list[str],
    *,
    sample_size: int,
    max_group_size: int,
    parallel_workers: int = 1,
) -> tuple[list[FeatureGroup], pd.DataFrame]:
    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be strictly positive.")
    groups: list[FeatureGroup] = []
    manifest_rows: list[dict[str, object]] = []
    buckets = build_feature_buckets(feature_names)
    bucket_items = list(buckets.items())
    worker_count = min(parallel_workers, len(bucket_items)) if bucket_items else 1
    LOGGER.info(
        "Feature grouping started: features=%d | buckets=%d | sample_size=%d | max_group_size=%d | parallel_workers=%d",
        len(feature_names),
        len(buckets),
        sample_size,
        max_group_size,
        worker_count,
    )
    bucket_results = _group_feature_buckets(
        cache,
        bucket_items,
        sample_size=sample_size,
        max_group_size=max_group_size,
        worker_count=worker_count,
    )
    for bucket_result in bucket_results:
        bucket_groups = bucket_result.bucket_groups
        groups.extend(bucket_groups)
        manifest_rows.extend(bucket_result.manifest_rows)
    manifest = pd.DataFrame(manifest_rows).sort_values(["group_id", "feature_name"]).reset_index(drop=True)
    LOGGER.info(
        "Feature grouping completed: groups=%d | manifest_rows=%d | family_breakdown=%s",
        len(groups),
        len(manifest_rows),
        _summarize_group_families(groups),
    )
    return groups, manifest


def _group_feature_buckets(
    cache: FeatureSelectionRuntimeCache,
    bucket_items: list[tuple[FeatureBucketKey, list[str]]],
    *,
    sample_size: int,
    max_group_size: int,
    worker_count: int,
) -> list[BucketGroupingResult]:
    if worker_count <= 1 or len(bucket_items) <= 1:
        sequential_results: list[BucketGroupingResult] = []
        for bucket_index, (bucket_key, bucket_feature_names) in enumerate(bucket_items, start=1):
            bucket_result = _group_single_bucket(
                cache,
                bucket_index,
                bucket_key,
                bucket_feature_names,
                sample_size=sample_size,
                max_group_size=max_group_size,
            )
            _log_bucket_grouping_result(bucket_result, len(bucket_items))
            sequential_results.append(bucket_result)
        return sequential_results
    bucket_results: dict[int, BucketGroupingResult] = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="feature-grouping") as executor:
        futures = [
            executor.submit(
                _group_single_bucket,
                cache,
                bucket_index,
                bucket_key,
                bucket_feature_names,
                sample_size=sample_size,
                max_group_size=max_group_size,
            )
            for bucket_index, (bucket_key, bucket_feature_names) in enumerate(bucket_items, start=1)
        ]
        for future in as_completed(futures):
            bucket_result = future.result()
            _log_bucket_grouping_result(bucket_result, len(bucket_items))
            bucket_results[bucket_result.bucket_index] = bucket_result
    return [bucket_results[bucket_index] for bucket_index in sorted(bucket_results)]


def _group_single_bucket(
    cache: FeatureSelectionRuntimeCache,
    bucket_index: int,
    bucket_key: FeatureBucketKey,
    bucket_feature_names: list[str],
    *,
    sample_size: int,
    max_group_size: int,
) -> BucketGroupingResult:
    sampled_frame = cache.build_sampled_feature_frame(bucket_feature_names, sample_size=sample_size)
    bucket_groups = partition_feature_bucket(
        bucket_key,
        sampled_frame,
        max_group_size=max_group_size,
    )
    return BucketGroupingResult(
        bucket_index=bucket_index,
        bucket_key=bucket_key,
        bucket_feature_count=len(bucket_feature_names),
        bucket_groups=bucket_groups,
        manifest_rows=_build_manifest_rows(bucket_groups),
    )


def _log_bucket_grouping_result(
    bucket_result: BucketGroupingResult,
    bucket_count: int,
) -> None:
    LOGGER.info(
        "Feature grouping bucket %d/%d | family=%s | stem=%s | bucket_features=%d | bucket_groups=%d",
        bucket_result.bucket_index,
        bucket_count,
        bucket_result.bucket_key.family,
        bucket_result.bucket_key.stem,
        bucket_result.bucket_feature_count,
        len(bucket_result.bucket_groups),
    )


def _build_feature_group(
    bucket_key: FeatureBucketKey,
    feature_names: list[str],
    index: int,
    level: int,
    parent_group_id: str | None,
) -> FeatureGroup:
    group_id = f"{bucket_key.family}:{bucket_key.stem}:{level}:{index}"
    return FeatureGroup(
        group_id=group_id,
        family=bucket_key.family,
        stem=bucket_key.stem,
        feature_names=tuple(sorted(feature_names)),
        level=level,
        parent_group_id=parent_group_id,
    )


def _build_abs_spearman_matrix(sampled_frame: pd.DataFrame) -> pd.DataFrame:
    ranked_frame = sampled_frame.rank(axis=0, method="average", na_option="keep")
    raw_corr = ranked_frame.corr(method="pearson").abs().fillna(0.0)
    corr_values = np.array(raw_corr.to_numpy(dtype=np.float64, copy=True), copy=True)
    np.fill_diagonal(corr_values, 1.0)
    return pd.DataFrame(corr_values, index=raw_corr.index, columns=raw_corr.columns)


def _derive_correlation_cutoff(correlation_matrix: pd.DataFrame) -> float:
    correlation_values = correlation_matrix.to_numpy(dtype=np.float64)
    upper_triangle = correlation_values[np.triu_indices_from(correlation_values, k=1)]
    if upper_triangle.size == 0:
        return 1.0
    sorted_values = np.sort(upper_triangle)
    if sorted_values.size == 1:
        return _to_float(sorted_values[0])
    distances = 1.0 - sorted_values
    distance_deltas = np.diff(distances)
    jump_index = int(np.argmax(distance_deltas))
    cutoff_distance = _to_float((distances[jump_index] + distances[jump_index + 1]) / 2.0)
    return _to_float(max(0.0, min(1.0, 1.0 - cutoff_distance)))


def _build_components(
    correlation_matrix: pd.DataFrame,
    correlation_cutoff: float,
) -> list[list[str]]:
    raw_remaining = cast(list[object], correlation_matrix.columns.tolist())
    remaining = [str(feature_name) for feature_name in raw_remaining]
    visited: set[str] = set()
    components: list[list[str]] = []
    for feature_name in remaining:
        if feature_name in visited:
            continue
        stack = [feature_name]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            linked = [
                other
                for other in remaining
                if other not in visited
                and _to_float(correlation_matrix.at[current, other]) >= correlation_cutoff
            ]
            stack.extend(linked)
        components.append(sorted(component))
    return sorted(components, key=lambda component: (len(component), component), reverse=True)


def _build_component_groups(
    bucket_key: FeatureBucketKey,
    components: list[list[str]],
    correlation_matrix: pd.DataFrame,
    max_group_size: int,
    level: int,
    parent_group_id: str | None,
) -> list[FeatureGroup]:
    groups: list[FeatureGroup] = []
    group_index = 1
    for component in components:
        if len(component) <= max_group_size:
            groups.append(_build_feature_group(bucket_key, component, group_index, level, parent_group_id))
            group_index += 1
            continue
        large_groups = _partition_large_component(
            bucket_key,
            component,
            cast(pd.DataFrame, correlation_matrix.loc[component, component]),
            max_group_size,
            level,
            parent_group_id,
            start_index=group_index,
        )
        groups.extend(large_groups)
        group_index += len(large_groups)
    return groups


def _partition_large_component(
    bucket_key: FeatureBucketKey,
    component: list[str],
    correlation_matrix: pd.DataFrame,
    max_group_size: int,
    level: int,
    parent_group_id: str | None,
    *,
    start_index: int = 1,
) -> list[FeatureGroup]:
    chunk_count = int(math.ceil(len(component) / max_group_size))
    seeds = _select_partition_seeds(component, correlation_matrix, chunk_count)
    assignments = {seed: [seed] for seed in seeds}
    capacities = dict.fromkeys(seeds, max_group_size)
    for feature_name in component:
        if feature_name in assignments:
            continue
        ordered_seeds = sorted(
            seeds,
            key=lambda seed: _seed_assignment_sort_key(
                feature_name,
                seed,
                correlation_matrix,
            ),
        )
        target_seed = next(seed for seed in ordered_seeds if len(assignments[seed]) < capacities[seed])
        assignments[target_seed].append(feature_name)
    groups: list[FeatureGroup] = []
    for group_index, seed in enumerate(sorted(assignments), start=start_index):
        groups.append(
            _build_feature_group(
                bucket_key,
                assignments[seed],
                group_index,
                level,
                parent_group_id,
            ),
        )
    return groups


def _select_partition_seeds(
    component: list[str],
    correlation_matrix: pd.DataFrame,
    chunk_count: int,
) -> list[str]:
    if chunk_count <= 1:
        return [sorted(component)[0]]
    seeds = [sorted(component)[0]]
    while len(seeds) < chunk_count:
        next_seed = max(
            [feature_name for feature_name in component if feature_name not in seeds],
            key=lambda feature_name: _candidate_seed_distance(
                feature_name,
                seeds,
                correlation_matrix,
            ),
        )
        seeds.append(next_seed)
    return seeds


def _seed_assignment_sort_key(
    feature_name: str,
    seed: str,
    correlation_matrix: pd.DataFrame,
) -> tuple[float, str]:
    return (-_to_float(correlation_matrix.at[feature_name, seed]), seed)


def _candidate_seed_distance(
    feature_name: str,
    seeds: list[str],
    correlation_matrix: pd.DataFrame,
) -> float:
    return min(1.0 - _to_float(correlation_matrix.at[feature_name, seed]) for seed in seeds)


def _to_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64).item())


def _build_manifest_rows(groups: list[FeatureGroup]) -> list[dict[str, object]]:
    return [
        {
            "group_id": group.group_id,
            "feature_family": group.family,
            "feature_stem": group.stem,
            "group_level": group.level,
            "parent_group_id": group.parent_group_id,
            "feature_name": feature_name,
        }
        for group in groups
        for feature_name in group.feature_names
    ]


def _summarize_group_families(groups: list[FeatureGroup]) -> str:
    family_counts: dict[str, int] = defaultdict(int)
    for group in groups:
        family_counts[group.family] += 1
    return ",".join(
        f"{family}:{family_counts[family]}"
        for family in sorted(family_counts)
    )


__all__ = [
    "FeatureBucketKey",
    "FeatureGroup",
    "build_feature_buckets",
    "build_feature_groups",
    "infer_feature_family",
    "normalize_feature_stem",
    "partition_feature_bucket",
]
