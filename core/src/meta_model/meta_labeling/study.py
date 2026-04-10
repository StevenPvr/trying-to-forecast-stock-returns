from __future__ import annotations

"""Optuna hyperparameter study for the binary meta-labeling XGBoost model."""

import logging
import time

import numpy as np
import pandas as pd

from core.src.meta_model.meta_labeling.config import MetaLabelingConfig
from core.src.meta_model.meta_labeling.features import META_PROBABILITY_COLUMN
from core.src.meta_model.meta_labeling.labels import (
    META_LABEL_COLUMN,
    select_meta_candidate_rows,
)
from core.src.meta_model.meta_labeling.metrics import (
    binary_average_precision,
    binary_logloss,
    binary_mcc,
    binary_roc_auc,
    has_binary_label_support,
)
from core.src.meta_model.meta_labeling.model import fit_meta_model, predict_meta_model
from core.src.meta_model.model_contract import DATE_COLUMN, LABEL_EMBARGO_DAYS
from core.src.meta_model.optimize_parameters.search_space import load_optuna_module

LOGGER: logging.Logger = logging.getLogger(__name__)


def mapping_float(payload: dict[str, object], key: str) -> float:
    raw_value = payload[key]
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, (np.integer, np.floating)):
        return float(raw_value.item())
    if isinstance(raw_value, str):
        return float(raw_value)
    raise TypeError(f"Expected numeric value for '{key}', got {type(raw_value).__name__}.")


def mapping_int(payload: dict[str, object], key: str) -> int:
    raw_value = payload[key]
    if isinstance(raw_value, int):
        return int(raw_value)
    if isinstance(raw_value, np.integer):
        return int(raw_value.item())
    if isinstance(raw_value, float):
        return int(raw_value)
    if isinstance(raw_value, np.floating):
        return int(raw_value.item())
    if isinstance(raw_value, str):
        return int(raw_value)
    raise TypeError(f"Expected integer value for '{key}', got {type(raw_value).__name__}.")


def meta_params_to_xgboost(
    learning_rate: float,
    max_depth: int,
    min_child_weight: float,
    *,
    random_seed: int,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    gamma: float = 0.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    max_bin: int = 256,
    scale_pos_weight: float | None = None,
) -> dict[str, float | int | str]:
    params: dict[str, float | int | str] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "min_child_weight": float(min_child_weight),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "gamma": float(gamma),
        "lambda": float(reg_lambda),
        "alpha": float(reg_alpha),
        "max_bin": int(max_bin),
        "seed": int(random_seed),
        "verbosity": 0,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return params


def estimate_scale_pos_weight(frame: pd.DataFrame) -> float:
    labels = pd.to_numeric(frame[META_LABEL_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    positive_count = float(np.sum(labels > 0.5))
    negative_count = float(np.sum(labels <= 0.5))
    if positive_count <= 0.0 or negative_count <= 0.0:
        return 1.0
    return max(1.0, negative_count / positive_count)


def build_meta_folds(
    train_tail: pd.DataFrame,
    *,
    fold_count: int,
) -> list[tuple[pd.Timestamp, list[pd.Timestamp]]]:
    unique_dates = pd.Index(pd.to_datetime(train_tail[DATE_COLUMN]).drop_duplicates().sort_values())
    date_chunks = [
        list(chunk.tolist())
        for chunk in np.array_split(unique_dates.to_numpy(), fold_count)
        if len(chunk) > 0
    ]
    folds: list[tuple[pd.Timestamp, list[pd.Timestamp]]] = []
    for fold_dates in date_chunks:
        validation_start = pd.Timestamp(fold_dates[0])
        historical_dates = unique_dates[unique_dates < validation_start]
        if len(historical_dates) <= LABEL_EMBARGO_DAYS:
            continue
        train_cutoff = pd.Timestamp(historical_dates[-LABEL_EMBARGO_DAYS - 1])
        folds.append((train_cutoff, [pd.Timestamp(value) for value in fold_dates]))
    LOGGER.info(
        "Meta folds built: requested=%d | usable=%d | unique_dates=%d",
        fold_count,
        len(folds),
        len(unique_dates),
    )
    return folds


def _suggest_meta_xgboost_params(
    trial: object,
    *,
    runtime: MetaLabelingConfig,
    scale_pos_weight: float,
) -> dict[str, float | int | str]:
    suggest_float = getattr(trial, "suggest_float")
    suggest_int = getattr(trial, "suggest_int")
    return meta_params_to_xgboost(
        learning_rate=float(suggest_float("learning_rate", 0.01, 0.15, log=True)),
        max_depth=int(suggest_int("max_depth", 3, 8)),
        min_child_weight=float(suggest_float("min_child_weight", 0.5, 128.0, log=True)),
        random_seed=runtime.random_seed,
        subsample=float(suggest_float("subsample", 0.60, 1.0)),
        colsample_bytree=float(suggest_float("colsample_bytree", 0.40, 1.0)),
        gamma=float(suggest_float("gamma", 1e-6, 10.0, log=True)),
        reg_lambda=float(suggest_float("lambda", 1e-3, 100.0, log=True)),
        reg_alpha=float(suggest_float("alpha", 1e-5, 25.0, log=True)),
        max_bin=int(suggest_int("max_bin", 64, 256)),
        scale_pos_weight=scale_pos_weight,
    )


def _score_meta_params(
    train_tail: pd.DataFrame,
    *,
    folds: list[tuple[pd.Timestamp, list[pd.Timestamp]]],
    meta_feature_columns: list[str],
    runtime: MetaLabelingConfig,
    params: dict[str, float | int | str],
) -> tuple[float, float, float, float, int]:
    fold_mccs: list[float] = []
    fold_pr_aucs: list[float] = []
    fold_roc_aucs: list[float] = []
    fold_loglosses: list[float] = []
    selected_rounds: list[int] = []
    for train_cutoff, fold_dates in folds:
        fold_train = pd.DataFrame(train_tail.loc[train_tail[DATE_COLUMN] <= train_cutoff].copy())
        fold_val = pd.DataFrame(train_tail.loc[train_tail[DATE_COLUMN].isin(fold_dates)].copy())
        fold_train_candidates = select_meta_candidate_rows(fold_train)
        fold_val_candidates = select_meta_candidate_rows(fold_val)
        if (
            fold_train_candidates.empty
            or fold_val_candidates.empty
            or not has_binary_label_support(fold_train_candidates)
            or not has_binary_label_support(fold_val_candidates)
        ):
            continue
        artifact = fit_meta_model(
            fold_train_candidates,
            meta_feature_columns,
            label_column=META_LABEL_COLUMN,
            params=params,
            training_rounds=runtime.boost_rounds,
            early_stopping_rounds=runtime.early_stopping_rounds,
            early_stopping_validation_fraction=runtime.early_stopping_validation_fraction,
            minimum_training_rounds=runtime.minimum_training_rounds,
        )
        scored = pd.DataFrame(fold_val.copy())
        scored[META_PROBABILITY_COLUMN] = 0.0
        probabilities = predict_meta_model(artifact, fold_val_candidates)
        scored.loc[fold_val_candidates.index, META_PROBABILITY_COLUMN] = probabilities
        scored_candidates = select_meta_candidate_rows(scored)
        fold_mccs.append(
            binary_mcc(
                scored_candidates,
                threshold=runtime.meta_decision_threshold,
                use_balanced_class_weights=runtime.meta_mcc_use_balanced_class_weights,
            ),
        )
        fold_pr_aucs.append(binary_average_precision(scored_candidates))
        fold_roc_aucs.append(binary_roc_auc(scored_candidates))
        fold_loglosses.append(binary_logloss(scored_candidates))
        selected_rounds.append(int(artifact.training_rounds))
    mean_mcc = float(np.mean(np.asarray(fold_mccs, dtype=np.float64))) if fold_mccs else -1.0
    mean_pr_auc = float(np.mean(np.asarray(fold_pr_aucs, dtype=np.float64))) if fold_pr_aucs else 0.0
    mean_roc_auc = float(np.mean(np.asarray(fold_roc_aucs, dtype=np.float64))) if fold_roc_aucs else float("nan")
    mean_logloss = float(np.mean(np.asarray(fold_loglosses, dtype=np.float64))) if fold_loglosses else float("inf")
    selected_training_rounds = int(
        max(
            runtime.minimum_training_rounds,
            round(float(np.median(np.asarray(selected_rounds, dtype=np.float64)))),
        ),
    ) if selected_rounds else int(runtime.minimum_training_rounds)
    return mean_mcc, mean_pr_auc, mean_roc_auc, mean_logloss, selected_training_rounds


def _study_to_ledger(study: object) -> pd.DataFrame:
    trials = list(getattr(study, "trials"))
    rows: list[dict[str, object]] = []
    for trial in trials:
        params = dict(getattr(trial, "params"))
        user_attrs = dict(getattr(trial, "user_attrs"))
        rows.append({
            "trial_index": int(getattr(trial, "number")),
            "learning_rate": float(params["learning_rate"]),
            "max_depth": int(params["max_depth"]),
            "min_child_weight": float(params["min_child_weight"]),
            "subsample": float(params["subsample"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "gamma": float(params["gamma"]),
            "lambda": float(params["lambda"]),
            "alpha": float(params["alpha"]),
            "max_bin": int(params["max_bin"]),
            "objective_score": float(getattr(trial, "value")),
            "mean_mcc": float(user_attrs.get("mean_mcc", float("nan"))),
            "mean_pr_auc": float(user_attrs.get("mean_pr_auc", float("nan"))),
            "mean_roc_auc": float(user_attrs.get("mean_roc_auc", float("nan"))),
            "mean_logloss": float(user_attrs.get("mean_logloss", float("inf"))),
            "selected_training_rounds": int(user_attrs.get("selected_training_rounds", 0)),
        })
    return pd.DataFrame(rows).sort_values("trial_index").reset_index(drop=True)


def run_meta_optuna_study(
    train_tail: pd.DataFrame,
    *,
    meta_feature_columns: list[str],
    runtime: MetaLabelingConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    folds = build_meta_folds(train_tail, fold_count=runtime.fold_count)
    optuna = load_optuna_module()
    candidate_train_tail = select_meta_candidate_rows(train_tail)
    scale_pos_weight = estimate_scale_pos_weight(candidate_train_tail)
    sampler = optuna.samplers.TPESampler(
        seed=runtime.random_seed,
        n_startup_trials=runtime.meta_optuna_startup_trials,
    )
    LOGGER.info(
        "Meta Optuna study started: trials=%d | folds=%d | boost_rounds=%d | sampler=TPESampler | scale_pos_weight=%.4f | decision_threshold=%.2f | weighted_mcc=%s",
        runtime.meta_trial_count,
        len(folds),
        runtime.boost_rounds,
        scale_pos_weight,
        runtime.meta_decision_threshold,
        "on" if runtime.meta_mcc_use_balanced_class_weights else "off",
    )
    study_started_at: float = time.perf_counter()

    def objective(trial: object) -> float:
        params = _suggest_meta_xgboost_params(
            trial,
            runtime=runtime,
            scale_pos_weight=scale_pos_weight,
        )
        mean_mcc, mean_pr_auc, mean_roc_auc, mean_logloss, selected_training_rounds = _score_meta_params(
            train_tail,
            folds=folds,
            meta_feature_columns=meta_feature_columns,
            runtime=runtime,
            params=params,
        )
        set_user_attr = getattr(trial, "set_user_attr")
        set_user_attr("mean_mcc", mean_mcc)
        set_user_attr("mean_pr_auc", mean_pr_auc)
        set_user_attr("mean_roc_auc", mean_roc_auc)
        set_user_attr("mean_logloss", mean_logloss)
        set_user_attr("selected_training_rounds", int(selected_training_rounds))
        LOGGER.info(
            "Meta Optuna trial %d completed: mean_mcc=%.6f | mean_pr_auc=%.6f | mean_roc_auc=%.6f | mean_logloss=%.6f | rounds=%d | lr=%.4f | depth=%d",
            int(getattr(trial, "number")),
            mean_mcc,
            mean_pr_auc,
            mean_roc_auc,
            mean_logloss,
            int(selected_training_rounds),
            float(params["learning_rate"]),
            int(params["max_depth"]),
        )
        return mean_mcc

    study = optuna.create_study(
        study_name="meta_labeling_binary_xgboost",
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(
        objective,
        n_trials=runtime.meta_trial_count,
        n_jobs=1,
        show_progress_bar=False,
    )
    ledger = _study_to_ledger(study)
    best_trial = getattr(study, "best_trial")
    best_row = ledger.sort_values("objective_score", ascending=False).reset_index(drop=True).iloc[0]
    best_params: dict[str, object] = {
        "trial_index": int(getattr(best_trial, "number")),
        "learning_rate": float(getattr(best_trial, "params")["learning_rate"]),
        "max_depth": int(getattr(best_trial, "params")["max_depth"]),
        "min_child_weight": float(getattr(best_trial, "params")["min_child_weight"]),
        "subsample": float(getattr(best_trial, "params")["subsample"]),
        "colsample_bytree": float(getattr(best_trial, "params")["colsample_bytree"]),
        "gamma": float(getattr(best_trial, "params")["gamma"]),
        "lambda": float(getattr(best_trial, "params")["lambda"]),
        "alpha": float(getattr(best_trial, "params")["alpha"]),
        "max_bin": int(getattr(best_trial, "params")["max_bin"]),
        "scale_pos_weight": float(scale_pos_weight),
        "selected_training_rounds": int(getattr(best_trial, "user_attrs")["selected_training_rounds"]),
    }
    LOGGER.info(
        "Meta Optuna study completed: best_mcc=%.6f | best_pr_auc=%.6f | best_roc_auc=%.6f | best_logloss=%.6f | best_rounds=%d | best_lr=%.4f | best_depth=%d | elapsed=%.1fs",
        float(np.asarray(best_row["objective_score"], dtype=np.float64).item()),
        float(np.asarray(best_row["mean_pr_auc"], dtype=np.float64).item()),
        float(np.asarray(best_row["mean_roc_auc"], dtype=np.float64).item()),
        float(np.asarray(best_row["mean_logloss"], dtype=np.float64).item()),
        int(np.asarray(best_row["selected_training_rounds"], dtype=np.int64).item()),
        float(np.asarray(best_row["learning_rate"], dtype=np.float64).item()),
        int(np.asarray(best_row["max_depth"], dtype=np.int64).item()),
        time.perf_counter() - study_started_at,
    )
    return ledger, best_params
