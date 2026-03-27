from __future__ import annotations

from pathlib import Path

from core.src.meta_model.data.paths import DATA_DIR

SECONDARY_MODEL_DATA_DIR: Path = DATA_DIR / "secondary_model"
SECONDARY_FEATURE_SELECTION_DIR: Path = SECONDARY_MODEL_DATA_DIR / "feature_selection"
SECONDARY_DATA_PREPROCESSING_DIR: Path = SECONDARY_MODEL_DATA_DIR / "data_preprocessing"
SECONDARY_OPTIMIZE_PARAMETERS_DIR: Path = SECONDARY_MODEL_DATA_DIR / "optimize_parameters"
SECONDARY_OOS_PREDICTIONS_DIR: Path = SECONDARY_MODEL_DATA_DIR / "oos_predictions"

SECONDARY_FEATURE_CORR_PCA_OUTPUT_PARQUET: Path = (
    SECONDARY_FEATURE_SELECTION_DIR / "dataset_features_corr_pca_secondary.parquet"
)
SECONDARY_FEATURE_CORR_PCA_OUTPUT_SAMPLE_CSV: Path = (
    SECONDARY_FEATURE_SELECTION_DIR / "dataset_features_corr_pca_secondary_sample_5pct.csv"
)
SECONDARY_FEATURE_CORR_PCA_MAPPING_JSON: Path = (
    SECONDARY_FEATURE_SELECTION_DIR / "feature_corr_pca_secondary_mapping.json"
)

SECONDARY_GREEDY_FORWARD_SELECTION_SCORES_PARQUET: Path = (
    SECONDARY_FEATURE_SELECTION_DIR / "feature_greedy_forward_selection_secondary_scores.parquet"
)
SECONDARY_GREEDY_FORWARD_SELECTION_SCORES_CSV: Path = (
    SECONDARY_FEATURE_SELECTION_DIR / "feature_greedy_forward_selection_secondary_scores.csv"
)
SECONDARY_GREEDY_FORWARD_SELECTION_SELECTED_FEATURES_PARQUET: Path = (
    SECONDARY_FEATURE_SELECTION_DIR / "feature_greedy_forward_selection_secondary_selected.parquet"
)
SECONDARY_GREEDY_FORWARD_SELECTION_SELECTED_FEATURES_CSV: Path = (
    SECONDARY_FEATURE_SELECTION_DIR / "feature_greedy_forward_selection_secondary_selected.csv"
)
SECONDARY_GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_PARQUET: Path = (
    SECONDARY_FEATURE_SELECTION_DIR / "dataset_features_greedy_forward_secondary_selected.parquet"
)
SECONDARY_GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_CSV: Path = (
    SECONDARY_FEATURE_SELECTION_DIR / "dataset_features_greedy_forward_secondary_selected_sample_5pct.csv"
)


def build_secondary_feature_selection_target_dir(target_name: str) -> Path:
    return SECONDARY_FEATURE_SELECTION_DIR / target_name


def build_secondary_feature_corr_pca_output_parquet(target_name: str) -> Path:
    return build_secondary_feature_selection_target_dir(target_name) / "dataset_features_corr_pca.parquet"


def build_secondary_feature_corr_pca_output_sample_csv(target_name: str) -> Path:
    return build_secondary_feature_selection_target_dir(target_name) / "dataset_features_corr_pca_sample_5pct.csv"


def build_secondary_feature_corr_pca_mapping_json(target_name: str) -> Path:
    return build_secondary_feature_selection_target_dir(target_name) / "feature_corr_pca_mapping.json"


def build_secondary_greedy_scores_parquet(target_name: str) -> Path:
    return build_secondary_feature_selection_target_dir(target_name) / "feature_greedy_forward_selection_scores.parquet"


def build_secondary_greedy_scores_csv(target_name: str) -> Path:
    return build_secondary_feature_selection_target_dir(target_name) / "feature_greedy_forward_selection_scores.csv"


def build_secondary_greedy_selected_features_parquet(target_name: str) -> Path:
    return build_secondary_feature_selection_target_dir(target_name) / "feature_greedy_forward_selection_selected.parquet"


def build_secondary_greedy_selected_features_csv(target_name: str) -> Path:
    return build_secondary_feature_selection_target_dir(target_name) / "feature_greedy_forward_selection_selected.csv"


def build_secondary_greedy_filtered_features_parquet(target_name: str) -> Path:
    return build_secondary_feature_selection_target_dir(target_name) / "dataset_features_greedy_forward_selected.parquet"


def build_secondary_greedy_filtered_features_csv(target_name: str) -> Path:
    return build_secondary_feature_selection_target_dir(target_name) / "dataset_features_greedy_forward_selected_sample_5pct.csv"


def build_secondary_preprocessing_target_dir(target_name: str) -> Path:
    return SECONDARY_DATA_PREPROCESSING_DIR / target_name


def build_secondary_optuna_trials_parquet(target_name: str) -> Path:
    return build_secondary_optimize_parameters_target_dir(target_name) / "xgboost_optuna_trials.parquet"


def build_secondary_optuna_trials_csv(target_name: str) -> Path:
    return build_secondary_optimize_parameters_target_dir(target_name) / "xgboost_optuna_trials.csv"


def build_secondary_best_params_json(target_name: str) -> Path:
    return build_secondary_optimize_parameters_target_dir(target_name) / "xgboost_best_params.json"


def build_secondary_preprocessed_dataset_parquet(target_name: str) -> Path:
    return build_secondary_preprocessing_target_dir(target_name) / "dataset_preprocessed.parquet"


def build_secondary_optimize_parameters_target_dir(target_name: str) -> Path:
    return SECONDARY_OPTIMIZE_PARAMETERS_DIR / target_name


def build_secondary_xgboost_optuna_trials_parquet(target_name: str) -> Path:
    return build_secondary_optuna_trials_parquet(target_name)


def build_secondary_xgboost_optuna_trials_csv(target_name: str) -> Path:
    return build_secondary_optuna_trials_csv(target_name)


def build_secondary_xgboost_best_params_json(target_name: str) -> Path:
    return build_secondary_best_params_json(target_name)


SECONDARY_OOS_PREDICTIONS_PARQUET: Path = (
    SECONDARY_OOS_PREDICTIONS_DIR / "dataset_oos_predictions.parquet"
)
SECONDARY_OOS_PREDICTIONS_SAMPLE_CSV: Path = (
    SECONDARY_OOS_PREDICTIONS_DIR / "dataset_oos_predictions_sample_5pct.csv"
)


def build_secondary_oos_predictions_target_dir(target_name: str) -> Path:
    return SECONDARY_OOS_PREDICTIONS_DIR / target_name


def build_secondary_target_oos_predictions_parquet(target_name: str) -> Path:
    return build_secondary_oos_predictions_target_dir(target_name) / "dataset_oos_predictions.parquet"


def build_secondary_target_oos_predictions_csv(target_name: str) -> Path:
    return build_secondary_oos_predictions_target_dir(target_name) / "dataset_oos_predictions_sample_5pct.csv"
