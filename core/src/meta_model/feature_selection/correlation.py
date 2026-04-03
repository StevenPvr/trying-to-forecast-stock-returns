from __future__ import annotations

from collections.abc import Callable
import importlib
import importlib.util
import logging
from typing import Any, Protocol, TypeVar, cast

import numpy as np
import pandas as pd

from core.src.meta_model.feature_selection.cache import FeatureSelectionRuntimeCache
from core.src.meta_model.feature_selection.config import FeatureSelectionConfig

LOGGER: logging.Logger = logging.getLogger(__name__)
_numba_spec = importlib.util.find_spec("numba")
NUMBA_AVAILABLE: bool = _numba_spec is not None
DecoratedFn = TypeVar("DecoratedFn", bound=Callable[..., Any])


class NumbaDecorator(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Callable[[DecoratedFn], DecoratedFn]:
        ...


if _numba_spec is not None:  # pragma: no branch
    _numba = importlib.import_module("numba")
    njit: NumbaDecorator = cast(NumbaDecorator, _numba.njit)
    prange: Callable[..., Any] = cast(Callable[..., Any], _numba.prange)
else:  # pragma: no cover
    def _njit_fallback(*args: Any, **kwargs: Any) -> Callable[[DecoratedFn], DecoratedFn]:
        def decorator(fn: DecoratedFn) -> DecoratedFn:
            return fn
        return decorator

    def _prange_fallback(*args: int) -> range:
        return range(*args)

    njit = _njit_fallback
    prange = _prange_fallback

SFI_RANK_COLUMNS: list[str] = [
    "objective_score",
    "daily_rank_ic_mean",
    "coverage_fraction",
    "feature_name",
]
MetricBuilder = Callable[[pd.DataFrame], pd.DataFrame]


def run_incremental_linear_correlation_pruning(
    cache: FeatureSelectionRuntimeCache,
    sfi_frame: pd.DataFrame,
    config: FeatureSelectionConfig,
) -> tuple[list[str], pd.DataFrame]:
    survivors = _extract_sfi_survivors(sfi_frame)
    sampled = cache.build_sampled_feature_frame(survivors, sample_size=config.group_sample_size)
    LOGGER.info(
        "Feature selection linear pruning started: candidates=%d | sample_rows=%d | threshold=%.2f",
        len(survivors),
        len(sampled),
        config.linear_correlation_threshold,
    )
    pruned_survivors, audit = _run_pruning_stages(
        sampled,
        sfi_frame,
        survivors,
        threshold=config.linear_correlation_threshold,
        stage_name="linear",
        metric_builder=_build_abs_spearman_matrix,
    )
    LOGGER.info(
        "Feature selection linear pruning completed: survivors=%d | dropped=%d",
        len(pruned_survivors),
        max(0, len(survivors) - len(pruned_survivors)),
    )
    return pruned_survivors, audit


def run_incremental_distance_correlation_pruning(
    cache: FeatureSelectionRuntimeCache,
    sfi_frame: pd.DataFrame,
    linear_survivors: list[str],
    config: FeatureSelectionConfig,
) -> tuple[list[str], pd.DataFrame]:
    candidate_features = _rank_features_by_sfi(sfi_frame, linear_survivors)
    sampled = cache.build_sampled_feature_frame(
        candidate_features,
        sample_size=config.distance_correlation_sample_size,
    )
    LOGGER.info(
        "Feature selection distance pruning started: candidates=%d | sample_rows=%d | threshold=%.2f",
        len(candidate_features),
        len(sampled),
        config.distance_correlation_threshold,
    )
    survivors, audit = _run_pruning_stages(
        sampled,
        sfi_frame,
        candidate_features,
        threshold=config.distance_correlation_threshold,
        stage_name="distance",
        metric_builder=_build_distance_correlation_matrix,
    )
    LOGGER.info(
        "Feature selection distance pruning completed: survivors=%d | dropped=%d",
        len(survivors),
        max(0, len(candidate_features) - len(survivors)),
    )
    return survivors, audit


def _extract_sfi_survivors(sfi_frame: pd.DataFrame) -> list[str]:
    passes_mask = cast(pd.Series, sfi_frame["passes_sfi"]).astype(bool)
    survivor_frame = cast(pd.DataFrame, sfi_frame.loc[passes_mask].copy())
    return _rank_features_by_sfi(sfi_frame=survivor_frame, feature_names=list(cast(pd.Series, survivor_frame["feature_name"]).astype(str)))


def _rank_features_by_sfi(sfi_frame: pd.DataFrame, feature_names: list[str]) -> list[str]:
    feature_frame = cast(
        pd.DataFrame,
        sfi_frame.loc[cast(pd.Series, sfi_frame["feature_name"]).isin(feature_names)].copy(),
    )
    sorted_frame = feature_frame.sort_values(
        SFI_RANK_COLUMNS,
        ascending=[False, False, False, True],
    )
    return [str(feature_name) for feature_name in cast(pd.Series, sorted_frame["feature_name"]).tolist()]


def _run_pruning_stages(
    sampled_frame: pd.DataFrame,
    sfi_frame: pd.DataFrame,
    feature_names: list[str],
    *,
    threshold: float,
    stage_name: str,
    metric_builder: MetricBuilder,
) -> tuple[list[str], pd.DataFrame]:
    audit_rows: list[dict[str, object]] = []
    stem_scopes = _build_stem_scopes(sfi_frame, feature_names)
    LOGGER.info(
        "Feature selection %s pruning stage=stem started: features=%d | scopes=%d",
        stage_name,
        len(feature_names),
        len(stem_scopes),
    )
    current = _prune_within_scopes(
        sampled_frame,
        sfi_frame,
        feature_names,
        scopes=stem_scopes,
        threshold=threshold,
        stage_name=stage_name,
        scope_level="stem",
        metric_builder=metric_builder,
        audit_rows=audit_rows,
    )
    LOGGER.info(
        "Feature selection %s pruning stage=stem completed: survivors=%d | preview=%s",
        stage_name,
        len(current),
        _build_feature_preview(current),
    )
    family_scopes = _build_family_scopes(sfi_frame, current)
    LOGGER.info(
        "Feature selection %s pruning stage=family started: features=%d | scopes=%d",
        stage_name,
        len(current),
        len(family_scopes),
    )
    current = _prune_within_scopes(
        sampled_frame,
        sfi_frame,
        current,
        scopes=family_scopes,
        threshold=threshold,
        stage_name=stage_name,
        scope_level="family",
        metric_builder=metric_builder,
        audit_rows=audit_rows,
    )
    LOGGER.info(
        "Feature selection %s pruning stage=family completed: survivors=%d | preview=%s",
        stage_name,
        len(current),
        _build_feature_preview(current),
    )
    LOGGER.info(
        "Feature selection %s pruning stage=global started: features=%d",
        stage_name,
        len(current),
    )
    current = _prune_within_scopes(
        sampled_frame,
        sfi_frame,
        current,
        scopes={"global": list(current)},
        threshold=threshold,
        stage_name=stage_name,
        scope_level="global",
        metric_builder=metric_builder,
        audit_rows=audit_rows,
    )
    LOGGER.info(
        "Feature selection %s pruning stage=global completed: survivors=%d | preview=%s",
        stage_name,
        len(current),
        _build_feature_preview(current),
    )
    audit = pd.DataFrame(audit_rows)
    return current, audit.sort_values(["stage_level", "scope_key", "feature_name"]).reset_index(drop=True)


def _prune_within_scopes(
    sampled_frame: pd.DataFrame,
    sfi_frame: pd.DataFrame,
    feature_names: list[str],
    *,
    scopes: dict[str, list[str]],
    threshold: float,
    stage_name: str,
    scope_level: str,
    metric_builder: MetricBuilder,
    audit_rows: list[dict[str, object]],
) -> list[str]:
    kept_by_scope: dict[str, list[str]] = {}
    completed_scopes = 0
    total_scopes = len(scopes)
    for scope_key in sorted(scopes):
        scope_features = _rank_features_by_sfi(sfi_frame, scopes[scope_key])
        if len(scope_features) <= 1:
            kept_by_scope[scope_key] = scope_features
            audit_rows.extend(_build_singleton_audit_rows(scope_features, stage_name, scope_level, scope_key))
            completed_scopes += 1
            _log_scope_progress(
                stage_name,
                scope_level,
                scope_key,
                completed_scopes,
                total_scopes,
                len(scope_features),
                len(scope_features),
            )
            continue
        LOGGER.info(
            "Feature selection %s pruning stage=%s scope=%s matrix_build_started | features=%d",
            stage_name,
            scope_level,
            scope_key,
            len(scope_features),
        )
        scope_frame = cast(pd.DataFrame, sampled_frame.loc[:, scope_features].copy())
        metric_matrix = metric_builder(scope_frame)
        LOGGER.info(
            "Feature selection %s pruning stage=%s scope=%s matrix_build_completed | features=%d",
            stage_name,
            scope_level,
            scope_key,
            len(scope_features),
        )
        kept_scope_features = _greedy_keep_scope(
            sfi_frame,
            scope_features,
            metric_matrix,
            threshold=threshold,
            stage_name=stage_name,
            scope_level=scope_level,
            scope_key=scope_key,
            audit_rows=audit_rows,
        )
        kept_by_scope[scope_key] = kept_scope_features
        completed_scopes += 1
        _log_scope_progress(
            stage_name,
            scope_level,
            scope_key,
            completed_scopes,
            total_scopes,
            len(scope_features),
            len(kept_scope_features),
        )
    retained: set[str] = set()
    for scope_features in kept_by_scope.values():
        retained.update(scope_features)
    return _rank_features_by_sfi(sfi_frame, [feature_name for feature_name in feature_names if feature_name in retained])


def _build_singleton_audit_rows(
    scope_features: list[str],
    stage_name: str,
    scope_level: str,
    scope_key: str,
) -> list[dict[str, object]]:
    return [
        {
            "stage_name": stage_name,
            "stage_level": scope_level,
            "scope_key": scope_key,
            "feature_name": feature_name,
            "decision": "keep",
            "representative_feature": feature_name,
            "reference_feature": None,
            "metric_value": None,
        }
        for feature_name in scope_features
    ]


def _greedy_keep_scope(
    sfi_frame: pd.DataFrame,
    scope_features: list[str],
    metric_matrix: pd.DataFrame,
    *,
    threshold: float,
    stage_name: str,
    scope_level: str,
    scope_key: str,
    audit_rows: list[dict[str, object]],
) -> list[str]:
    kept: list[str] = []
    for feature_name in scope_features:
        matched_feature, metric_value = _find_matching_representative(metric_matrix, feature_name, kept, threshold)
        if matched_feature is None:
            kept.append(feature_name)
            audit_rows.append(_build_audit_row(stage_name, scope_level, scope_key, feature_name, "keep", feature_name, None, None))
            continue
        audit_rows.append(
            _build_audit_row(
                stage_name,
                scope_level,
                scope_key,
                feature_name,
                "drop",
                matched_feature,
                matched_feature,
                metric_value,
            ),
        )
    return _rank_features_by_sfi(sfi_frame, kept)


def _find_matching_representative(
    metric_matrix: pd.DataFrame,
    feature_name: str,
    kept: list[str],
    threshold: float,
) -> tuple[str | None, float | None]:
    for kept_feature in kept:
        metric_value = float(cast(float, metric_matrix.at[feature_name, kept_feature]))
        if metric_value >= threshold:
            return kept_feature, metric_value
    return None, None


def _build_audit_row(
    stage_name: str,
    scope_level: str,
    scope_key: str,
    feature_name: str,
    decision: str,
    representative_feature: str,
    reference_feature: str | None,
    metric_value: float | None,
) -> dict[str, object]:
    return {
        "stage_name": stage_name,
        "stage_level": scope_level,
        "scope_key": scope_key,
        "feature_name": feature_name,
        "decision": decision,
        "representative_feature": representative_feature,
        "reference_feature": reference_feature,
        "metric_value": metric_value,
    }


def _build_stem_scopes(sfi_frame: pd.DataFrame, feature_names: list[str]) -> dict[str, list[str]]:
    scope_frame = cast(
        pd.DataFrame,
        sfi_frame.loc[cast(pd.Series, sfi_frame["feature_name"]).isin(feature_names), ["feature_name", "feature_stem"]].copy(),
    )
    scopes: dict[str, list[str]] = {}
    for stem_name, group_frame in scope_frame.groupby("feature_stem", sort=True):
        scopes[str(stem_name)] = [str(name) for name in cast(pd.Series, group_frame["feature_name"]).tolist()]
    return scopes


def _build_family_scopes(sfi_frame: pd.DataFrame, feature_names: list[str]) -> dict[str, list[str]]:
    scope_frame = cast(
        pd.DataFrame,
        sfi_frame.loc[cast(pd.Series, sfi_frame["feature_name"]).isin(feature_names), ["feature_name", "feature_family"]].copy(),
    )
    scopes: dict[str, list[str]] = {}
    for family_name, group_frame in scope_frame.groupby("feature_family", sort=True):
        scopes[str(family_name)] = [str(name) for name in cast(pd.Series, group_frame["feature_name"]).tolist()]
    return scopes


def _log_scope_progress(
    stage_name: str,
    scope_level: str,
    scope_key: str,
    completed_scopes: int,
    total_scopes: int,
    scope_feature_count: int,
    kept_feature_count: int,
) -> None:
    LOGGER.info(
        "Feature selection %s pruning stage=%s progress %d/%d | scope=%s | scope_features=%d | kept=%d | dropped=%d",
        stage_name,
        scope_level,
        completed_scopes,
        total_scopes,
        scope_key,
        scope_feature_count,
        kept_feature_count,
        max(0, scope_feature_count - kept_feature_count),
    )


def _build_feature_preview(feature_names: list[str]) -> str:
    return ", ".join(feature_names[:5]) if feature_names else "none"


def _build_abs_spearman_matrix(sampled_frame: pd.DataFrame) -> pd.DataFrame:
    ranked_frame = sampled_frame.rank(axis=0, method="average", na_option="keep")
    raw_corr = ranked_frame.corr(method="pearson").abs().fillna(0.0)
    values = raw_corr.to_numpy(dtype=np.float64, copy=True)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=raw_corr.index, columns=raw_corr.columns)


def _build_distance_correlation_matrix(sampled_frame: pd.DataFrame) -> pd.DataFrame:
    feature_names = [str(column_name) for column_name in cast(list[object], sampled_frame.columns.tolist())]
    values = np.eye(len(feature_names), dtype=np.float64)
    for row_index, feature_name in enumerate(feature_names):
        left = _resolve_distance_input(cast(pd.Series, sampled_frame[feature_name]))
        for column_index in range(row_index + 1, len(feature_names)):
            right_feature = feature_names[column_index]
            right = _resolve_distance_input(cast(pd.Series, sampled_frame[right_feature]))
            metric_value = _distance_correlation_numba_wrapper(left, right)
            values[row_index, column_index] = metric_value
            values[column_index, row_index] = metric_value
    return pd.DataFrame(values, index=feature_names, columns=feature_names)


def _resolve_distance_input(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        fill_value = float(np.median(finite_values)) if finite_values.size > 0 else 0.0
        values = np.where(finite_mask, values, fill_value)
    return values


@njit(cache=True, parallel=True)
def _distance_correlation_numba(left: np.ndarray, right: np.ndarray) -> float:
    n = left.shape[0]
    if n <= 1 or right.shape[0] != n:
        return 0.0
    left_constant = True
    right_constant = True
    left_first = left[0]
    right_first = right[0]
    for index in range(1, n):
        if left[index] != left_first:
            left_constant = False
        if right[index] != right_first:
            right_constant = False
    if left_constant or right_constant:
        return 0.0

    left_distances = np.empty((n, n), dtype=np.float64)
    right_distances = np.empty((n, n), dtype=np.float64)
    left_row_mean = np.empty(n, dtype=np.float64)
    right_row_mean = np.empty(n, dtype=np.float64)
    left_col_mean = np.empty(n, dtype=np.float64)
    right_col_mean = np.empty(n, dtype=np.float64)

    left_total = 0.0
    right_total = 0.0
    for row_index in prange(n):
        left_row_sum = 0.0
        right_row_sum = 0.0
        left_value = left[row_index]
        right_value = right[row_index]
        for column_index in range(n):
            left_distance = abs(left_value - left[column_index])
            right_distance = abs(right_value - right[column_index])
            left_distances[row_index, column_index] = left_distance
            right_distances[row_index, column_index] = right_distance
            left_row_sum += left_distance
            right_row_sum += right_distance
        left_row_mean[row_index] = left_row_sum / n
        right_row_mean[row_index] = right_row_sum / n
        left_total += left_row_sum
        right_total += right_row_sum

    for column_index in prange(n):
        left_col_sum = 0.0
        right_col_sum = 0.0
        for row_index in range(n):
            left_col_sum += left_distances[row_index, column_index]
            right_col_sum += right_distances[row_index, column_index]
        left_col_mean[column_index] = left_col_sum / n
        right_col_mean[column_index] = right_col_sum / n

    left_total_mean = left_total / (n * n)
    right_total_mean = right_total / (n * n)
    covariance = 0.0
    left_variance = 0.0
    right_variance = 0.0
    for row_index in prange(n):
        for column_index in range(n):
            left_centered = (
                left_distances[row_index, column_index]
                - left_row_mean[row_index]
                - left_col_mean[column_index]
                + left_total_mean
            )
            right_centered = (
                right_distances[row_index, column_index]
                - right_row_mean[row_index]
                - right_col_mean[column_index]
                + right_total_mean
            )
            covariance += left_centered * right_centered
            left_variance += left_centered * left_centered
            right_variance += right_centered * right_centered
    covariance /= n * n
    left_variance /= n * n
    right_variance /= n * n
    if left_variance <= 0.0 or right_variance <= 0.0:
        return 0.0
    denominator = np.sqrt(left_variance * right_variance)
    if denominator <= 0.0:
        return 0.0
    ratio = covariance / denominator
    if ratio < 0.0:
        ratio = 0.0
    return np.sqrt(ratio)


def _distance_correlation_numba_wrapper(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.ascontiguousarray(left.astype(np.float64, copy=False))
    right_array = np.ascontiguousarray(right.astype(np.float64, copy=False))
    return float(_distance_correlation_numba(left_array, right_array))


__all__ = [
    "run_incremental_distance_correlation_pruning",
    "run_incremental_linear_correlation_pruning",
]
