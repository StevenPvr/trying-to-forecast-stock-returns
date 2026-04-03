from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from threading import Lock
from typing import cast

import numpy as np
import pandas as pd

from core.src.meta_model.evaluate.backtest import XtbCostConfig
from core.src.meta_model.evaluate.config import BacktestConfig
from core.src.meta_model.feature_selection.cache import FeatureSelectionRuntimeCache
from core.src.meta_model.feature_selection.config import FeatureSelectionConfig
from core.src.meta_model.feature_selection.cv import SelectionFold
from core.src.meta_model.feature_selection.scoring import score_predicted_validation
from core.src.meta_model.model_contract import DATE_COLUMN, MODEL_TARGET_COLUMN, PREDICTION_COLUMN
from core.src.meta_model.model_registry.main import ModelArtifact, ModelSpec, fit_model, predict_model

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MdaSelectionResult:
    final_scores: pd.DataFrame
    group_scores: pd.DataFrame
    selected_feature_names: list[str]


def run_mda_selection(
    cache: FeatureSelectionRuntimeCache,
    folds: list[SelectionFold],
    sfi_frame: pd.DataFrame,
    candidate_feature_names: list[str],
    config: FeatureSelectionConfig,
) -> MdaSelectionResult:
    ranked_candidates = _rank_candidates(sfi_frame, candidate_feature_names)
    train_frame = cache.build_feature_frame(ranked_candidates)
    model_spec = _build_proxy_model_spec(config)
    backtest_config = BacktestConfig(
        top_fraction=config.proxy_top_fraction,
        neutrality_mode=config.proxy_neutrality_mode,
        open_hurdle_bps=config.proxy_open_hurdle_bps,
    )
    cost_config = XtbCostConfig()
    LOGGER.info(
        "Feature selection MDA started: candidates=%d | repeats=%d | folds=%d",
        len(ranked_candidates),
        config.mda_permutation_repeats,
        len(folds),
    )
    feature_rows = _score_feature_mda_rows(
        train_frame,
        folds,
        ranked_candidates,
        model_spec,
        backtest_config,
        cost_config,
        config,
    )
    final_scores = _build_final_score_frame(sfi_frame, feature_rows)
    group_scores = _build_group_scores(final_scores)
    selected_feature_names = [
        str(feature_name)
        for feature_name in cast(
            pd.Series,
            final_scores.loc[cast(pd.Series, final_scores["selected"]).astype(bool), "feature_name"],
        ).tolist()
    ]
    LOGGER.info(
        "Feature selection MDA completed: selected=%d | rejected=%d | top_selected=%s",
        len(selected_feature_names),
        len(final_scores) - len(selected_feature_names),
        _build_mda_preview(final_scores),
    )
    return MdaSelectionResult(
        final_scores=final_scores,
        group_scores=group_scores,
        selected_feature_names=selected_feature_names,
    )


def _rank_candidates(
    sfi_frame: pd.DataFrame,
    candidate_feature_names: list[str],
) -> list[str]:
    candidate_frame = cast(
        pd.DataFrame,
        sfi_frame.loc[cast(pd.Series, sfi_frame["feature_name"]).isin(candidate_feature_names)].copy(),
    )
    sorted_frame = candidate_frame.sort_values(
        ["objective_score", "daily_rank_ic_mean", "coverage_fraction", "feature_name"],
        ascending=[False, False, False, True],
    )
    return [str(feature_name) for feature_name in cast(pd.Series, sorted_frame["feature_name"]).tolist()]


def _build_proxy_model_spec(config: FeatureSelectionConfig) -> ModelSpec:
    params = dict(config.proxy_xgboost_params)
    params.setdefault("nthread", max(1, config.resolved_model_threads_per_worker(fold_count=max(1, config.fold_count))))
    return ModelSpec(
        model_name="xgboost",
        params=params,
        target_column=MODEL_TARGET_COLUMN,
        training_rounds=config.proxy_training_rounds,
    )


def _score_feature_mda_rows(
    train_frame: pd.DataFrame,
    folds: list[SelectionFold],
    candidate_feature_names: list[str],
    model_spec: ModelSpec,
    backtest_config: BacktestConfig,
    cost_config: XtbCostConfig,
    config: FeatureSelectionConfig,
) -> list[dict[str, object]]:
    LOGGER.info("Feature selection MDA fold contexts building started: folds=%d", len(folds))
    fold_contexts = [
        _build_fold_context(
            train_frame,
            fold,
            candidate_feature_names,
            model_spec,
            backtest_config,
            cost_config,
            total_folds=len(folds),
        )
        for fold in folds
    ]
    LOGGER.info("Feature selection MDA fold contexts built: folds=%d", len(fold_contexts))
    worker_count = _resolve_mda_worker_count(config, len(candidate_feature_names))
    if worker_count <= 1 or len(candidate_feature_names) <= 1:
        return _score_feature_mda_rows_sequential(
            fold_contexts,
            candidate_feature_names,
            config,
        )
    LOGGER.info(
        "Feature selection MDA feature scoring parallel enabled: features=%d | workers=%d",
        len(candidate_feature_names),
        worker_count,
    )
    return _score_feature_mda_rows_parallel(
        fold_contexts,
        candidate_feature_names,
        config,
        worker_count=worker_count,
    )


def _resolve_mda_worker_count(
    config: FeatureSelectionConfig,
    candidate_count: int,
) -> int:
    return min(max(1, config.parallel_workers), max(1, candidate_count))


def _score_feature_mda_rows_sequential(
    fold_contexts: list[dict[str, object]],
    candidate_feature_names: list[str],
    config: FeatureSelectionConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature_index, feature_name in enumerate(candidate_feature_names, start=1):
        row = _score_single_feature_mda(feature_name, fold_contexts, config)
        rows.append(row)
        _log_mda_feature_progress(feature_index, len(candidate_feature_names), feature_name, row)
    return rows


def _score_feature_mda_rows_parallel(
    fold_contexts: list[dict[str, object]],
    candidate_feature_names: list[str],
    config: FeatureSelectionConfig,
    *,
    worker_count: int,
) -> list[dict[str, object]]:
    rows_by_feature: dict[str, dict[str, object]] = {}
    completed_count = 0
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="feature-mda") as executor:
        future_map = {
            executor.submit(_score_single_feature_mda, feature_name, fold_contexts, config): feature_name
            for feature_name in candidate_feature_names
        }
        for future in as_completed(future_map):
            feature_name = future_map[future]
            row = future.result()
            rows_by_feature[feature_name] = row
            completed_count += 1
            _log_mda_feature_progress(completed_count, len(candidate_feature_names), feature_name, row)
    return [rows_by_feature[feature_name] for feature_name in candidate_feature_names]


def _log_mda_feature_progress(
    feature_index: int,
    feature_count: int,
    feature_name: str,
    row: dict[str, object],
) -> None:
    LOGGER.info(
        "Feature selection MDA progress %d/%d | feature=%s | delta=%.6f | fold_positive_share=%.4f | selected=%s",
        feature_index,
        feature_count,
        feature_name,
        float(cast(float, row["mda_mean_delta_objective"])),
        float(cast(float, row["mda_fold_positive_share"])),
        bool(cast(bool, row["selected"])),
    )


def _build_fold_context(
    train_frame: pd.DataFrame,
    fold: SelectionFold,
    candidate_feature_names: list[str],
    model_spec: ModelSpec,
    backtest_config: BacktestConfig,
    cost_config: XtbCostConfig,
    *,
    total_folds: int,
) -> dict[str, object]:
    fold_train = pd.DataFrame(train_frame.take(fold.train_indices).reset_index(drop=True))
    fold_validation = pd.DataFrame(train_frame.take(fold.validation_indices).reset_index(drop=True))
    artifact = fit_model(model_spec, fold_train, candidate_feature_names)
    baseline_score = _score_validation_frame(
        artifact,
        fold_validation,
        candidate_feature_names,
        backtest_config,
        cost_config,
    )
    LOGGER.info(
        "Feature selection MDA fold %d/%d baseline: train_rows=%d | validation_rows=%d | baseline_daily_rank_ic=%.6f",
        fold.index + 1,
        total_folds,
        len(fold_train),
        len(fold_validation),
        baseline_score,
    )
    return {
        "fold": fold,
        "artifact": artifact,
        "validation_frame": fold_validation,
        "baseline_score": baseline_score,
        "predict_lock": Lock(),
    }


def _score_validation_frame(
    artifact: ModelArtifact,
    validation_frame: pd.DataFrame,
    candidate_feature_names: list[str],
    backtest_config: BacktestConfig,
    cost_config: XtbCostConfig,
) -> float:
    predicted_validation = validation_frame.copy()
    predicted_validation[PREDICTION_COLUMN] = predict_model(artifact, validation_frame, candidate_feature_names)
    summary = score_predicted_validation(
        predicted_validation,
        backtest_config,
        cost_config,
        target_column=artifact.training_metadata.get("target_column", MODEL_TARGET_COLUMN),
    )
    return float(summary["daily_rank_ic_mean"])


def _score_single_feature_mda(
    feature_name: str,
    fold_contexts: list[dict[str, object]],
    config: FeatureSelectionConfig,
) -> dict[str, object]:
    deltas: list[float] = []
    fold_positive_count = 0
    fold_mean_deltas: list[float] = []
    for fold_context in fold_contexts:
        fold_deltas = _score_fold_feature_mda(feature_name, fold_context, config)
        mean_delta = float(np.mean(np.asarray(fold_deltas, dtype=np.float64))) if fold_deltas else 0.0
        deltas.extend(fold_deltas)
        fold_mean_deltas.append(mean_delta)
        if mean_delta > 0.0:
            fold_positive_count += 1
    delta_array = np.asarray(deltas, dtype=np.float64)
    mda_mean_delta = float(delta_array.mean()) if delta_array.size > 0 else 0.0
    mda_std_delta = float(delta_array.std(ddof=0)) if delta_array.size > 0 else 0.0
    fold_positive_share = float(fold_positive_count / max(1, len(fold_contexts)))
    selected = mda_mean_delta > 0.0 and fold_positive_share >= 0.50
    LOGGER.info(
        "Feature selection MDA feature completed | feature=%s | fold_mean_deltas=%s | overall_delta=%.6f | selected=%s",
        feature_name,
        ",".join(f"{delta:.6f}" for delta in fold_mean_deltas),
        mda_mean_delta,
        selected,
    )
    return {
        "feature_name": feature_name,
        "mda_mean_delta_objective": mda_mean_delta,
        "mda_std_delta_objective": mda_std_delta,
        "mda_fold_positive_share": fold_positive_share,
        "mda_repeat_count": len(deltas),
        "selected": selected,
        "mda_drop_reason": "selected" if selected else "non_positive_mda",
    }


def _score_fold_feature_mda(
    feature_name: str,
    fold_context: dict[str, object],
    config: FeatureSelectionConfig,
) -> list[float]:
    artifact = cast(ModelArtifact, fold_context["artifact"])
    fold = cast(SelectionFold, fold_context["fold"])
    validation_frame = cast(pd.DataFrame, fold_context["validation_frame"])
    baseline_score = float(cast(float, fold_context["baseline_score"]))
    predict_lock = cast(Lock, fold_context["predict_lock"])
    feature_names = list(artifact.feature_names)
    deltas: list[float] = []
    for repeat_index in range(config.mda_permutation_repeats):
        permuted_validation = _permute_feature_within_dates(
            validation_frame,
            feature_name,
            seed=config.random_seed + (fold.index * 10_000) + repeat_index,
        )
        with predict_lock:
            permuted_predictions = _predict_validation_frame(
                artifact,
                permuted_validation,
                feature_names,
            )
        permuted_score = _score_predicted_validation_frame(
            permuted_predictions,
            backtest_config=BacktestConfig(
                top_fraction=config.proxy_top_fraction,
                neutrality_mode=config.proxy_neutrality_mode,
                open_hurdle_bps=config.proxy_open_hurdle_bps,
            ),
            cost_config=XtbCostConfig(),
            target_column=artifact.training_metadata.get("target_column", MODEL_TARGET_COLUMN),
        )
        deltas.append(baseline_score - permuted_score)
    return deltas


def _predict_validation_frame(
    artifact: ModelArtifact,
    validation_frame: pd.DataFrame,
    candidate_feature_names: list[str],
) -> pd.DataFrame:
    predicted_validation = validation_frame.copy()
    predicted_validation[PREDICTION_COLUMN] = predict_model(
        artifact,
        validation_frame,
        candidate_feature_names,
    )
    return predicted_validation


def _score_predicted_validation_frame(
    predicted_validation: pd.DataFrame,
    *,
    backtest_config: BacktestConfig,
    cost_config: XtbCostConfig,
    target_column: str,
) -> float:
    summary = score_predicted_validation(
        predicted_validation,
        backtest_config,
        cost_config,
        target_column=target_column,
    )
    return float(summary["daily_rank_ic_mean"])


def _permute_feature_within_dates(
    validation_frame: pd.DataFrame,
    feature_name: str,
    *,
    seed: int,
) -> pd.DataFrame:
    permuted = validation_frame.copy()
    rng = np.random.default_rng(seed)
    for _, row_indices in cast(pd.Series, permuted.groupby(DATE_COLUMN, sort=True).indices).items():
        integer_indices = np.asarray(row_indices, dtype=np.int64)
        current_values = permuted.iloc[integer_indices][feature_name].to_numpy(copy=True)
        permuted_values = rng.permutation(current_values)
        permuted.iloc[integer_indices, permuted.columns.get_loc(feature_name)] = permuted_values
    return permuted


def _build_final_score_frame(
    sfi_frame: pd.DataFrame,
    feature_rows: list[dict[str, object]],
) -> pd.DataFrame:
    feature_frame = pd.DataFrame(feature_rows)
    merged = cast(pd.DataFrame, sfi_frame.merge(feature_frame, on="feature_name", how="inner"))
    sorted_frame = merged.sort_values(
        ["selected", "mda_mean_delta_objective", "objective_score", "feature_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected_mask = cast(pd.Series, sorted_frame["selected"]).astype(bool)
    sorted_frame["selection_rank"] = 0
    sorted_frame.loc[selected_mask, "selection_rank"] = np.arange(1, int(selected_mask.sum()) + 1, dtype=np.int64)
    sorted_frame["drop_reason"] = cast(pd.Series, sorted_frame["mda_drop_reason"]).astype(str)
    return sorted_frame


def _build_group_scores(final_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = final_scores.groupby(["feature_family", "feature_stem"], dropna=False, sort=True).agg(
        feature_count=("feature_name", "size"),
        selected_count=("selected", "sum"),
        best_mda_mean_delta_objective=("mda_mean_delta_objective", "max"),
        mean_mda_mean_delta_objective=("mda_mean_delta_objective", "mean"),
        best_sfi_objective_score=("objective_score", "max"),
    )
    return grouped.reset_index()


def _build_mda_preview(final_scores: pd.DataFrame) -> str:
    selected_scores = cast(
        pd.DataFrame,
        final_scores.loc[cast(pd.Series, final_scores["selected"]).astype(bool)].copy(),
    )
    if selected_scores.empty:
        return "none"
    sorted_scores = selected_scores.sort_values(
        ["mda_mean_delta_objective", "feature_name"],
        ascending=[False, True],
    ).head(5)
    labels: list[str] = []
    for row in cast(list[dict[str, object]], sorted_scores.to_dict(orient="records")):
        labels.append(f"{str(row['feature_name'])}={float(cast(float, row['mda_mean_delta_objective'])):.6f}")
    return ", ".join(labels)


__all__ = ["MdaSelectionResult", "run_mda_selection"]
