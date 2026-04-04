from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
CORE_DIR: Path = PROJECT_ROOT / "core"
DATA_DIR: Path = CORE_DIR / "data"
REFERENCE_DATA_DIR: Path = DATA_DIR / "reference"
REFERENCE_WRDS_DIR: Path = REFERENCE_DATA_DIR / "wrds"
XTB_REFERENCE_DIR: Path = REFERENCE_DATA_DIR / "xtb"
REFERENCE_EARNINGS_HISTORY_CSV: Path = REFERENCE_DATA_DIR / "sp500_earnings_history.csv"
XTB_INSTRUMENT_SPECS_REFERENCE_JSON: Path = (
    XTB_REFERENCE_DIR / "xtb_instrument_specs.json"
)

# --- Data Fetching ---
DATA_FETCHING_DIR: Path = DATA_DIR / "data_fetching"
OUTPUT_PARQUET: Path = DATA_FETCHING_DIR / "sp500_ohlcv_2004_2025.parquet"
OUTPUT_SAMPLE_CSV: Path = DATA_FETCHING_DIR / "sp500_ohlcv_2004_2025_sample_5pct.csv"
MERGED_OUTPUT_PARQUET: Path = DATA_FETCHING_DIR / "dataset_2004_2025.parquet"
MERGED_OUTPUT_SAMPLE_CSV: Path = DATA_FETCHING_DIR / "dataset_2004_2025_sample_5pct.csv"
UNIVERSE_COMPANIES_XLSX: Path = DATA_FETCHING_DIR / "sp500_universe_companies.xlsx"
MEMBERSHIP_HISTORY_CSV: Path = REFERENCE_DATA_DIR / "sp500_membership_history.csv"
FUNDAMENTALS_HISTORY_CSV: Path = REFERENCE_DATA_DIR / "sp500_fundamentals_history.csv"
WRDS_FUNDQ_EXTRACT_CSV: Path = REFERENCE_WRDS_DIR / "compustat_fundq_extract.csv"

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
PREPROCESSED_FEATURE_REGISTRY_PARQUET: Path = (
    DATA_PREPROCESSING_DIR / "feature_registry.parquet"
)
PREPROCESSED_FEATURE_REGISTRY_JSON: Path = (
    DATA_PREPROCESSING_DIR / "feature_registry.json"
)
PREPROCESSED_FEATURE_SCHEMA_MANIFEST_JSON: Path = (
    DATA_PREPROCESSING_DIR / "feature_schema_manifest.json"
)
PREPROCESSED_RESEARCH_LABEL_PANEL_PARQUET: Path = (
    DATA_PREPROCESSING_DIR / "research_label_panel.parquet"
)

# --- Features Engineering ---
DATA_FEATURES_ENGINEERING_DIR: Path = DATA_DIR / "features_engineering"
FEATURES_OUTPUT_PARQUET: Path = (
    DATA_FEATURES_ENGINEERING_DIR / "dataset_features_2004_2025.parquet"
)
FEATURES_OUTPUT_SAMPLE_CSV: Path = (
    DATA_FEATURES_ENGINEERING_DIR / "dataset_features_2004_2025_sample_5pct.csv"
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
FEATURE_SELECTION_STABILITY_SCORES_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_stability_scores.parquet"
)
FEATURE_SELECTION_STABILITY_SCORES_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_stability_scores.csv"
)
FEATURE_SELECTION_SELECTED_FEATURES_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_stability_selected.parquet"
)
FEATURE_SELECTION_SELECTED_FEATURES_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_stability_selected.csv"
)
FEATURE_SELECTION_FILTERED_DATASET_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "dataset_preprocessed_feature_selected.parquet"
)
FEATURE_SELECTION_FILTERED_DATASET_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "dataset_preprocessed_feature_selected_sample_5pct.csv"
)
FEATURE_SELECTION_FEATURE_REGISTRY_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_registry.parquet"
)
FEATURE_SELECTION_FEATURE_REGISTRY_JSON: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_registry.json"
)
FEATURE_SELECTION_INPUT_INVENTORY_JSON: Path = (
    DATA_FEATURE_SELECTION_DIR / "input_inventory.json"
)
FEATURE_SELECTION_SCHEMA_MANIFEST_JSON: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_schema_manifest.json"
)
FEATURE_SELECTION_GROUP_MANIFEST_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_group_manifest.parquet"
)
FEATURE_SELECTION_GROUP_MANIFEST_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_group_manifest.csv"
)
FEATURE_SELECTION_WRAPPER_SEARCH_PARQUET: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_wrapper_search.parquet"
)
FEATURE_SELECTION_WRAPPER_SEARCH_CSV: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_wrapper_search.csv"
)
FEATURE_SELECTION_SUMMARY_JSON: Path = (
    DATA_FEATURE_SELECTION_DIR / "feature_selection_summary.json"
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
OPTIMIZATION_TRIAL_LEDGER_PARQUET: Path = (
    DATA_OPTIMIZE_PARAMETERS_DIR / "trial_ledger.parquet"
)
OPTIMIZATION_TRIAL_LEDGER_CSV: Path = (
    DATA_OPTIMIZE_PARAMETERS_DIR / "trial_ledger.csv"
)
OPTIMIZATION_OVERFITTING_REPORT_JSON: Path = (
    DATA_OPTIMIZE_PARAMETERS_DIR / "overfitting_report.json"
)

# --- Broker XTB ---
DATA_BROKER_XTB_DIR: Path = DATA_DIR / "broker_xtb"
XTB_TRADABLE_UNIVERSE_PARQUET: Path = (
    DATA_BROKER_XTB_DIR / "xtb_tradable_universe.parquet"
)
XTB_TRADABLE_UNIVERSE_CSV: Path = (
    DATA_BROKER_XTB_DIR / "xtb_tradable_universe.csv"
)
XTB_SPECS_SNAPSHOT_JSON: Path = (
    DATA_BROKER_XTB_DIR / "xtb_specs_snapshot.json"
)
XTB_SWAP_SNAPSHOT_JSON: Path = (
    DATA_BROKER_XTB_DIR / "xtb_swap_snapshot.json"
)
XTB_MARGIN_SNAPSHOT_JSON: Path = (
    DATA_BROKER_XTB_DIR / "xtb_margin_snapshot.json"
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
EVALUATE_MODEL_LEADERBOARD_JSON: Path = (
    DATA_EVALUATE_DIR / "model_leaderboard.json"
)
EVALUATE_MANUAL_ORDERS_CSV: Path = DATA_EVALUATE_DIR / "manual_orders.csv"
EVALUATE_MANUAL_WATCHLIST_CSV: Path = DATA_EVALUATE_DIR / "manual_watchlist.csv"
EVALUATE_EXECUTION_CHECKLIST_JSON: Path = (
    DATA_EVALUATE_DIR / "execution_checklist.json"
)
EVALUATE_POST_TRADE_RECONCILIATION_PARQUET: Path = (
    DATA_EVALUATE_DIR / "post_trade_reconciliation.parquet"
)
EVALUATE_OVERFITTING_REPORT_JSON: Path = (
    DATA_EVALUATE_DIR / "overfitting_report.json"
)
