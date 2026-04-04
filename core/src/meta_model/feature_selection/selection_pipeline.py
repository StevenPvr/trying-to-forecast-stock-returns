from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import cast

import pandas as pd

from core.src.meta_model.feature_selection.cache import FeatureSelectionRuntimeCache
from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.correlation import (
    run_incremental_distance_correlation_pruning,
    run_incremental_linear_correlation_pruning,
    run_target_distance_correlation_filter,
)
from core.src.meta_model.feature_selection.cv import SelectionFold
from core.src.meta_model.feature_selection.grouping import build_feature_buckets
from core.src.meta_model.feature_selection.scoring import BacktestFeatureSubsetScorer
from core.src.meta_model.feature_selection.sfi import build_sfi_score_frame

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RobustFeatureSelectionResult:
    score_frame: pd.DataFrame
    selected_feature_names: list[str]
    group_manifest: pd.DataFrame
    sfi_scores: pd.DataFrame
    linear_pruning_audit: pd.DataFrame
    distance_correlation_audit: pd.DataFrame
    target_correlation_audit: pd.DataFrame
    summary: dict[str, object]


def run_robust_feature_selection(
    cache: FeatureSelectionRuntimeCache,
    folds: list[SelectionFold],
    feature_names: list[str],
    config: FeatureSelectionConfig,
) -> RobustFeatureSelectionResult:
    LOGGER.info(
        "Feature selection robust pipeline started: candidates=%d | folds=%d",
        len(feature_names),
        len(folds),
    )
    stage_start = time.perf_counter()
    group_manifest = build_group_manifest(feature_names)
    scorer = BacktestFeatureSubsetScorer(cache, folds, config)
    sfi_scores = build_sfi_score_frame(cache, feature_names, scorer, config)
    _log_sfi_stage_summary(sfi_scores)
    LOGGER.info("Feature selection stage timing | stage=sfi | elapsed=%.2fs", time.perf_counter() - stage_start)
    stage_start = time.perf_counter()
    linear_survivors, linear_audit = run_incremental_linear_correlation_pruning(cache, sfi_scores, config)
    _log_survivor_preview("linear pruning", linear_survivors)
    LOGGER.info("Feature selection stage timing | stage=linear_pruning | elapsed=%.2fs", time.perf_counter() - stage_start)
    stage_start = time.perf_counter()
    distance_survivors, distance_audit = run_incremental_distance_correlation_pruning(
        cache,
        sfi_scores,
        linear_survivors,
        config,
    )
    _log_survivor_preview("distance pruning", distance_survivors)
    LOGGER.info("Feature selection stage timing | stage=distance_pruning | elapsed=%.2fs", time.perf_counter() - stage_start)
    stage_start = time.perf_counter()
    target_survivors, target_audit = run_target_distance_correlation_filter(
        cache,
        sfi_scores,
        distance_survivors,
        config,
    )
    _log_survivor_preview("target correlation filter", target_survivors)
    LOGGER.info("Feature selection stage timing | stage=target_correlation_filter | elapsed=%.2fs", time.perf_counter() - stage_start)
    final_selected_names = build_final_candidate_feature_names(sfi_scores, target_survivors)
    if len(final_selected_names) != len(target_survivors):
        LOGGER.info(
            "Feature selection broker feature rescue: target_survivors=%d | final_candidates=%d | rescued=%d | preview=%s",
            len(target_survivors),
            len(final_selected_names),
            len(final_selected_names) - len(target_survivors),
            ", ".join([n for n in final_selected_names if n not in set(target_survivors)][:5]) or "none",
        )
    _log_final_selection_summary(sfi_scores, final_selected_names)
    score_frame = build_feature_score_report(
        sfi_scores,
        linear_survivors,
        distance_survivors,
        target_survivors,
        target_audit,
        final_selected_names,
    )
    summary = build_selection_summary(
        feature_names,
        sfi_scores,
        linear_survivors,
        distance_survivors,
        target_survivors,
        final_selected_names,
        config,
    )
    LOGGER.info(
        "Feature selection robust pipeline completed: sfi_survivors=%d | linear_survivors=%d | distance_survivors=%d | selected=%d",
        int(cast(pd.Series, sfi_scores["passes_sfi"]).sum()),
        len(linear_survivors),
        len(distance_survivors),
        len(final_selected_names),
    )
    return RobustFeatureSelectionResult(
        score_frame=score_frame,
        selected_feature_names=final_selected_names,
        group_manifest=group_manifest,
        sfi_scores=sfi_scores,
        linear_pruning_audit=linear_audit,
        distance_correlation_audit=distance_audit,
        target_correlation_audit=target_audit,
        summary=summary,
    )


def build_group_manifest(feature_names: list[str]) -> pd.DataFrame:
    buckets = build_feature_buckets(feature_names)
    rows: list[dict[str, object]] = []
    for bucket_key, bucket_features in sorted(buckets.items(), key=lambda item: (item[0].family, item[0].stem)):
        group_id = f"{bucket_key.family}:{bucket_key.stem}:0:1"
        for feature_name in bucket_features:
            rows.append(
                {
                    "group_id": group_id,
                    "feature_family": bucket_key.family,
                    "feature_stem": bucket_key.stem,
                    "group_level": 0,
                    "parent_group_id": None,
                    "feature_name": feature_name,
                },
            )
    return pd.DataFrame(rows).sort_values(["group_id", "feature_name"]).reset_index(drop=True)


def build_final_candidate_feature_names(
    sfi_scores: pd.DataFrame,
    target_survivors: list[str],
) -> list[str]:
    """Ordered unique names: target-filter survivors plus rescued broker features."""
    target_survivor_set = set(target_survivors)
    broker_bypass_frame = cast(
        pd.DataFrame,
        sfi_scores.loc[
            (cast(pd.Series, sfi_scores["feature_family"]).astype(str) == "broker")
            & cast(pd.Series, sfi_scores["passes_coverage"]).astype(bool)
            & ~cast(pd.Series, sfi_scores["feature_name"]).isin(target_survivor_set),
        ].copy(),
    )
    broker_bypass_names = [
        str(feature_name)
        for feature_name in cast(pd.Series, broker_bypass_frame.sort_values(
            ["objective_score", "daily_rank_ic_mean", "coverage_fraction", "feature_name"],
            ascending=[False, False, False, True],
        )["feature_name"]).tolist()
    ]
    ordered_names = [*target_survivors, *broker_bypass_names]
    return list(dict.fromkeys(ordered_names))


def build_feature_score_report(
    sfi_scores: pd.DataFrame,
    linear_survivors: list[str],
    distance_survivors: list[str],
    target_survivors: list[str],
    target_correlation_audit: pd.DataFrame,
    final_selected_names: list[str],
) -> pd.DataFrame:
    score_frame = cast(pd.DataFrame, sfi_scores.copy())
    score_frame["selected_linear"] = cast(pd.Series, score_frame["feature_name"]).isin(linear_survivors)
    score_frame["selected_distance"] = cast(pd.Series, score_frame["feature_name"]).isin(distance_survivors)
    score_frame["selected_target_correlation"] = cast(pd.Series, score_frame["feature_name"]).isin(target_survivors)
    target_columns = [
        "feature_name",
        "target_distance_correlation",
    ]
    available_target_columns = [
        column_name
        for column_name in target_columns
        if column_name in target_correlation_audit.columns
    ]
    if len(available_target_columns) > 1:
        score_frame = cast(
            pd.DataFrame,
            score_frame.merge(
                target_correlation_audit.loc[:, available_target_columns],
                on="feature_name",
                how="left",
            ),
        )
    else:
        score_frame["target_distance_correlation"] = pd.NA

    final_set = set(final_selected_names)
    rank_map = {name: rank for rank, name in enumerate(final_selected_names, start=1)}
    name_series = cast(pd.Series, score_frame["feature_name"]).astype(str)
    score_frame["selected"] = name_series.isin(final_set)
    score_frame["selection_rank"] = name_series.map(lambda n: rank_map.get(str(n), 0)).astype(int)
    score_frame["drop_reason"] = _resolve_feature_drop_reasons(score_frame, final_set)
    return score_frame.sort_values(
        ["selected", "selection_rank", "objective_score", "feature_name"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)


def _resolve_feature_drop_reasons(score_frame: pd.DataFrame, final_set: set[str]) -> pd.Series:
    feature_series = cast(pd.Series, score_frame["feature_name"]).astype(str)
    in_final = feature_series.isin(final_set)
    passes_sfi = cast(pd.Series, score_frame["passes_sfi"]).astype(bool)
    sel_lin = cast(pd.Series, score_frame["selected_linear"]).astype(bool)
    sel_dist = cast(pd.Series, score_frame["selected_distance"]).astype(bool)
    sel_tgt = cast(pd.Series, score_frame["selected_target_correlation"]).astype(bool)
    reasons = pd.Series("not_selected", index=score_frame.index, dtype=object)
    reasons.loc[in_final] = "selected"
    pending = ~in_final
    reasons.loc[pending & ~passes_sfi] = "rejected_sfi"
    pending = pending & passes_sfi
    reasons.loc[pending & ~sel_lin] = "rejected_linear_pruning"
    pending = pending & sel_lin
    reasons.loc[pending & ~sel_dist] = "rejected_distance_pruning"
    pending = pending & sel_dist
    reasons.loc[pending & ~sel_tgt] = "low_target_distance_correlation"
    return reasons


def build_selection_summary(
    feature_names: list[str],
    sfi_scores: pd.DataFrame,
    linear_survivors: list[str],
    distance_survivors: list[str],
    target_survivors: list[str],
    selected_feature_names: list[str],
    config: FeatureSelectionConfig,
) -> dict[str, object]:
    passes_sfi_count = int(cast(pd.Series, sfi_scores["passes_sfi"]).sum())
    return {
        "candidate_feature_count": len(feature_names),
        "sfi_survivor_count": passes_sfi_count,
        "linear_survivor_count": len(linear_survivors),
        "distance_survivor_count": len(distance_survivors),
        "target_correlation_survivor_count": len(target_survivors),
        "selected_feature_count": len(selected_feature_names),
        "proxy_xgboost_params": dict(config.proxy_xgboost_params),
        "proxy_training_rounds": config.proxy_training_rounds,
        "random_seed": config.random_seed,
        "sfi_min_coverage_fraction": config.sfi_min_coverage_fraction,
        "linear_correlation_threshold": config.linear_correlation_threshold,
        "distance_correlation_threshold": config.distance_correlation_threshold,
        "target_distance_correlation_threshold": config.target_distance_correlation_threshold,
    }


def _log_sfi_stage_summary(sfi_scores: pd.DataFrame) -> None:
    survivors = cast(pd.DataFrame, sfi_scores.loc[cast(pd.Series, sfi_scores["passes_sfi"]).astype(bool)].copy())
    LOGGER.info(
        "Feature selection SFI summary: survivors=%d | dropped=%d | top_survivors=%s",
        len(survivors),
        len(sfi_scores) - len(survivors),
        _build_score_preview(survivors, score_column="objective_score"),
    )


def _log_survivor_preview(stage_name: str, survivor_names: list[str]) -> None:
    LOGGER.info(
        "Feature selection %s summary: survivors=%d | preview=%s",
        stage_name,
        len(survivor_names),
        ", ".join(survivor_names[:5]) if survivor_names else "none",
    )


def _log_final_selection_summary(sfi_scores: pd.DataFrame, selected_feature_names: list[str]) -> None:
    selected_set = set(selected_feature_names)
    picked = cast(
        pd.DataFrame,
        sfi_scores.loc[cast(pd.Series, sfi_scores["feature_name"]).astype(str).isin(selected_set)].copy(),
    )
    LOGGER.info(
        "Feature selection final summary: selected=%d | top_by_sfi_objective=%s",
        len(selected_feature_names),
        _build_score_preview(picked, score_column="objective_score"),
    )


def _build_score_preview(score_frame: pd.DataFrame, *, score_column: str) -> str:
    if score_frame.empty:
        return "none"
    sorted_frame = score_frame.sort_values([score_column, "feature_name"], ascending=[False, True]).head(5)
    labels: list[str] = []
    for row in cast(list[dict[str, object]], sorted_frame.to_dict(orient="records")):
        labels.append(f"{str(row['feature_name'])}={float(cast(float, row[score_column])):.6f}")
    return ", ".join(labels)


__all__ = [
    "RobustFeatureSelectionResult",
    "build_feature_score_report",
    "build_final_candidate_feature_names",
    "build_group_manifest",
    "build_selection_summary",
    "run_robust_feature_selection",
]
