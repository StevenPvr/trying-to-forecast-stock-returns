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
)
from core.src.meta_model.feature_selection.cv import SelectionFold
from core.src.meta_model.feature_selection.grouping import build_feature_buckets
from core.src.meta_model.feature_selection.mda import MdaSelectionResult, run_mda_selection
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
    mda_group_scores: pd.DataFrame
    mda_final_scores: pd.DataFrame
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
    mda_result = run_mda_selection(cache, folds, sfi_scores, distance_survivors, config)
    _log_mda_stage_summary(mda_result.final_scores, mda_result.selected_feature_names)
    LOGGER.info("Feature selection stage timing | stage=mda | elapsed=%.2fs", time.perf_counter() - stage_start)
    score_frame = build_feature_score_report(
        feature_names,
        sfi_scores,
        linear_survivors,
        distance_survivors,
        mda_result,
    )
    summary = build_selection_summary(
        feature_names,
        sfi_scores,
        linear_survivors,
        distance_survivors,
        mda_result.selected_feature_names,
        config,
    )
    LOGGER.info(
        "Feature selection robust pipeline completed: sfi_survivors=%d | linear_survivors=%d | distance_survivors=%d | selected=%d",
        int(cast(pd.Series, sfi_scores["passes_sfi"]).sum()),
        len(linear_survivors),
        len(distance_survivors),
        len(mda_result.selected_feature_names),
    )
    return RobustFeatureSelectionResult(
        score_frame=score_frame,
        selected_feature_names=mda_result.selected_feature_names,
        group_manifest=group_manifest,
        sfi_scores=sfi_scores,
        linear_pruning_audit=linear_audit,
        distance_correlation_audit=distance_audit,
        mda_group_scores=mda_result.group_scores,
        mda_final_scores=mda_result.final_scores,
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


def build_feature_score_report(
    feature_names: list[str],
    sfi_scores: pd.DataFrame,
    linear_survivors: list[str],
    distance_survivors: list[str],
    mda_result: MdaSelectionResult,
) -> pd.DataFrame:
    score_frame = cast(pd.DataFrame, sfi_scores.copy())
    score_frame["selected_linear"] = cast(pd.Series, score_frame["feature_name"]).isin(linear_survivors)
    score_frame["selected_distance"] = cast(pd.Series, score_frame["feature_name"]).isin(distance_survivors)
    mda_columns = [
        "feature_name",
        "mda_mean_delta_objective",
        "mda_std_delta_objective",
        "mda_fold_positive_share",
        "mda_repeat_count",
        "selected",
        "selection_rank",
        "drop_reason",
    ]
    score_frame = cast(
        pd.DataFrame,
        score_frame.merge(mda_result.final_scores.loc[:, mda_columns], on="feature_name", how="left"),
    )
    score_frame["selected"] = cast(pd.Series, score_frame["selected"]).fillna(False).astype(bool)
    score_frame["selection_rank"] = cast(pd.Series, score_frame["selection_rank"]).fillna(0).astype(int)
    score_frame["drop_reason"] = cast(pd.Series, score_frame["drop_reason"]).fillna("rejected_before_mda").astype(str)
    missing_mda_mask = ~cast(pd.Series, score_frame["feature_name"]).isin(list(cast(pd.Series, mda_result.final_scores["feature_name"]).astype(str)))
    score_frame.loc[missing_mda_mask, "drop_reason"] = "rejected_before_mda"
    return score_frame.sort_values(
        ["selected", "selection_rank", "objective_score", "feature_name"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)


def build_selection_summary(
    feature_names: list[str],
    sfi_scores: pd.DataFrame,
    linear_survivors: list[str],
    distance_survivors: list[str],
    selected_feature_names: list[str],
    config: FeatureSelectionConfig,
) -> dict[str, object]:
    passes_sfi_count = int(cast(pd.Series, sfi_scores["passes_sfi"]).sum())
    return {
        "candidate_feature_count": len(feature_names),
        "sfi_survivor_count": passes_sfi_count,
        "linear_survivor_count": len(linear_survivors),
        "distance_survivor_count": len(distance_survivors),
        "selected_feature_count": len(selected_feature_names),
        "proxy_xgboost_params": dict(config.proxy_xgboost_params),
        "proxy_training_rounds": config.proxy_training_rounds,
        "random_seed": config.random_seed,
        "sfi_min_coverage_fraction": config.sfi_min_coverage_fraction,
        "linear_correlation_threshold": config.linear_correlation_threshold,
        "distance_correlation_threshold": config.distance_correlation_threshold,
        "mda_permutation_repeats": config.mda_permutation_repeats,
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


def _log_mda_stage_summary(final_scores: pd.DataFrame, selected_feature_names: list[str]) -> None:
    LOGGER.info(
        "Feature selection MDA summary: selected=%d | rejected=%d | top_selected=%s",
        len(selected_feature_names),
        len(final_scores) - len(selected_feature_names),
        _build_score_preview(
            cast(pd.DataFrame, final_scores.loc[cast(pd.Series, final_scores["selected"]).astype(bool)].copy()),
            score_column="mda_mean_delta_objective",
        ),
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
    "build_group_manifest",
    "build_selection_summary",
    "run_robust_feature_selection",
]
