from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
CORE_DIR: Path = PROJECT_ROOT / "core"
DATA_DIR: Path = CORE_DIR / "data"
REFERENCE_DATA_DIR: Path = DATA_DIR / "reference"

# --- Data Fetching ---
DATA_FETCHING_DIR: Path = DATA_DIR / "data_fetching"
OUTPUT_PARQUET: Path = DATA_FETCHING_DIR / "sp500_ohlcv_2004_2025.parquet"
OUTPUT_SAMPLE_CSV: Path = DATA_FETCHING_DIR / "sp500_ohlcv_2004_2025_sample_5pct.csv"
MERGED_OUTPUT_PARQUET: Path = DATA_FETCHING_DIR / "dataset_2004_2025.parquet"
MERGED_OUTPUT_SAMPLE_CSV: Path = DATA_FETCHING_DIR / "dataset_2004_2025_sample_5pct.csv"
UNIVERSE_COMPANIES_XLSX: Path = DATA_FETCHING_DIR / "sp500_universe_companies.xlsx"
MEMBERSHIP_HISTORY_CSV: Path = REFERENCE_DATA_DIR / "sp500_membership_history.csv"
FUNDAMENTALS_HISTORY_CSV: Path = REFERENCE_DATA_DIR / "sp500_fundamentals_history.csv"

# --- Data Cleaning ---
DATA_CLEANING_DIR: Path = DATA_DIR / "data_cleaning"
CLEANED_OUTPUT_PARQUET: Path = DATA_CLEANING_DIR / "dataset_cleaned_2004_2025.parquet"
CLEANED_OUTPUT_SAMPLE_CSV: Path = DATA_CLEANING_DIR / "dataset_cleaned_2004_2025_sample_5pct.csv"
OUTLIER_PLOTS_DIR: Path = DATA_CLEANING_DIR / "outlier_plots"

# --- Data Preprocessing ---
DATA_PREPROCESSING_DIR: Path = DATA_DIR / "data_preprocessing"
PREPROCESSED_OUTPUT_PARQUET: Path = (
    DATA_PREPROCESSING_DIR / "dataset_preprocessed_2009_2025.parquet"
)
PREPROCESSED_OUTPUT_SAMPLE_CSV: Path = (
    DATA_PREPROCESSING_DIR / "dataset_preprocessed_2009_2025_sample_5pct.csv"
)
PREPROCESSED_TRAIN_PARQUET: Path = DATA_PREPROCESSING_DIR / "dataset_preprocessed_train.parquet"
PREPROCESSED_TRAIN_SAMPLE_CSV: Path = (
    DATA_PREPROCESSING_DIR / "dataset_preprocessed_train_sample_5pct.csv"
)
PREPROCESSED_VAL_PARQUET: Path = DATA_PREPROCESSING_DIR / "dataset_preprocessed_val.parquet"
PREPROCESSED_VAL_SAMPLE_CSV: Path = (
    DATA_PREPROCESSING_DIR / "dataset_preprocessed_val_sample_5pct.csv"
)
PREPROCESSED_TEST_PARQUET: Path = DATA_PREPROCESSING_DIR / "dataset_preprocessed_test.parquet"
PREPROCESSED_TEST_SAMPLE_CSV: Path = (
    DATA_PREPROCESSING_DIR / "dataset_preprocessed_test_sample_5pct.csv"
)

# --- Features Engineering ---
DATA_FEATURES_ENGINEERING_DIR: Path = DATA_DIR / "features_engineering"
FEATURES_OUTPUT_PARQUET: Path = (
    DATA_FEATURES_ENGINEERING_DIR / "dataset_features_2004_2025.parquet"
)
FEATURES_OUTPUT_SAMPLE_CSV: Path = (
    DATA_FEATURES_ENGINEERING_DIR / "dataset_features_2004_2025_sample_5pct.csv"
)

# --- Feature Correlation + Kernel PCA ---
DATA_FEATURE_CORR_PCA_DIR: Path = DATA_DIR / "feature_corr_pca"
FEATURE_CORR_PCA_OUTPUT_PARQUET: Path = (
    DATA_FEATURE_CORR_PCA_DIR / "dataset_features_corr_pca_2004_2025.parquet"
)
FEATURE_CORR_PCA_OUTPUT_SAMPLE_CSV: Path = (
    DATA_FEATURE_CORR_PCA_DIR / "dataset_features_corr_pca_2004_2025_sample_5pct.csv"
)
FEATURE_CORR_PCA_MAPPING_JSON: Path = (
    DATA_FEATURE_CORR_PCA_DIR / "feature_corr_pca_mapping.json"
)

# --- Feature Selection ---
DATA_FEATURE_SELECTION_DIR: Path = DATA_DIR / "feature_selection"
FEATURE_SELECTION_SFI_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_sfi_scores.parquet"
)
FEATURE_SELECTION_SFI_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_sfi_scores.csv"
)
FEATURE_SELECTION_UNDERPERFORMING_FEATURES_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_sfi_underperformers.parquet"
)
FEATURE_SELECTION_UNDERPERFORMING_FEATURES_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_sfi_underperformers.csv"
)
GREEDY_FORWARD_SELECTION_SCORES_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_greedy_forward_selection_scores.parquet"
)
GREEDY_FORWARD_SELECTION_SCORES_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_greedy_forward_selection_scores.csv"
)
GREEDY_FORWARD_SELECTION_SELECTED_FEATURES_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_greedy_forward_selection_selected.parquet"
)
GREEDY_FORWARD_SELECTION_SELECTED_FEATURES_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_greedy_forward_selection_selected.csv"
)
GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "dataset_features_greedy_forward_selected.parquet"
)
GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "dataset_features_greedy_forward_selected_sample_5pct.csv"
)

# --- Feature MDA ---
FEATURE_MDA_GROUP_SCORES_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_mda_group_scores.parquet"
)
FEATURE_MDA_GROUP_SCORES_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_mda_group_scores.csv"
)
FEATURE_MDA_FINAL_SCORES_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_mda_final_scores.parquet"
)
FEATURE_MDA_FINAL_SCORES_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_mda_final_scores.csv"
)
FEATURE_MDA_SELECTED_FEATURES_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_mda_selected.parquet"
)
FEATURE_MDA_SELECTED_FEATURES_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_mda_selected.csv"
)
FEATURE_MDA_FILTERED_FEATURES_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "dataset_features_mda_selected.parquet"
)
FEATURE_MDA_FILTERED_FEATURES_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "dataset_features_mda_selected_sample_5pct.csv"
)
DEEP_FEATURE_MDA_FILTERED_FEATURES_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "dataset_deep_features_mda_selected.parquet"
)
DEEP_FEATURE_MDA_FILTERED_FEATURES_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "dataset_deep_features_mda_selected_sample_5pct.csv"
)

# --- Deep Feature Engineering ---
DATA_DEEP_FEATURE_ENGINEERING_DIR: Path = DATA_DIR / "deep_feature_engineering"
DEEP_FEATURES_OUTPUT_PARQUET: Path = (
    DATA_DEEP_FEATURE_ENGINEERING_DIR / "dataset_deep_features.parquet"
)
DEEP_FEATURES_OUTPUT_SAMPLE_CSV: Path = (
    DATA_DEEP_FEATURE_ENGINEERING_DIR / "dataset_deep_features_sample_5pct.csv"
)

# --- Accounting Feature Engineering ---
DATA_ACCOUNTING_FEATURE_ENGINEERING_DIR: Path = DATA_DIR / "accounting_feature_engineering"
ACCOUNTING_HISTORY_PARQUET: Path = (
    DATA_ACCOUNTING_FEATURE_ENGINEERING_DIR / "accounting_history.parquet"
)
ACCOUNTING_HISTORY_SAMPLE_CSV: Path = (
    DATA_ACCOUNTING_FEATURE_ENGINEERING_DIR / "accounting_history_sample_5pct.csv"
)
ACCOUNTING_FEATURES_OUTPUT_PARQUET: Path = (
    DATA_ACCOUNTING_FEATURE_ENGINEERING_DIR / "dataset_accounting_features.parquet"
)
ACCOUNTING_FEATURES_OUTPUT_SAMPLE_CSV: Path = (
    DATA_ACCOUNTING_FEATURE_ENGINEERING_DIR / "dataset_accounting_features_sample_5pct.csv"
)

# --- Parameter Optimization ---
DATA_OPTIMIZE_PARAMETERS_DIR: Path = DATA_DIR / "optimize_parameters"
XGBOOST_OPTUNA_TRIALS_PARQUET: Path = (
    DATA_OPTIMIZE_PARAMETERS_DIR / "xgboost_optuna_trials.parquet"
)
XGBOOST_OPTUNA_TRIALS_CSV: Path = (
    DATA_OPTIMIZE_PARAMETERS_DIR / "xgboost_optuna_trials.csv"
)
XGBOOST_OPTUNA_BEST_PARAMS_JSON: Path = (
    DATA_OPTIMIZE_PARAMETERS_DIR / "xgboost_best_params.json"
)

# --- Evaluate / Backtest ---
DATA_EVALUATE_DIR: Path = DATA_DIR / "evaluate"
EVALUATE_TEST_PREDICTIONS_PARQUET: Path = (
    DATA_EVALUATE_DIR / "test_predictions.parquet"
)
EVALUATE_TEST_PREDICTIONS_CSV: Path = (
    DATA_EVALUATE_DIR / "test_predictions_sample_5pct.csv"
)
EVALUATE_BACKTEST_TRADES_PARQUET: Path = (
    DATA_EVALUATE_DIR / "backtest_trades.parquet"
)
EVALUATE_BACKTEST_TRADES_CSV: Path = (
    DATA_EVALUATE_DIR / "backtest_trades_sample_5pct.csv"
)
EVALUATE_BACKTEST_DAILY_PARQUET: Path = (
    DATA_EVALUATE_DIR / "backtest_daily.parquet"
)
EVALUATE_BACKTEST_DAILY_CSV: Path = (
    DATA_EVALUATE_DIR / "backtest_daily_sample_5pct.csv"
)
EVALUATE_BACKTEST_SUMMARY_JSON: Path = (
    DATA_EVALUATE_DIR / "backtest_summary.json"
)
