# Artefacts et datasets

## Répertoires principaux

Tous les chemins centraux sont déclarés dans [core/src/meta_model/data/paths.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/paths.py).

## Data fetching

- `core/data/data_fetching/sp500_ohlcv_2004_2025.parquet`
  - OHLCV brut
- `core/data/data_fetching/dataset_2004_2025.parquet`
  - dataset fusionné marché/macro/calendrier/sentiment/cross-asset/univers

Le fetching canonique exige désormais:

- `core/data/reference/sp500_membership_history.csv`
- `core/data/reference/sp500_fundamentals_history.csv`
- `core/data/reference/xtb/xtb_instrument_specs.json`

Source fondamentale prioritaire:

1. WRDS direct via `ID_WRDS` / `PASSWORD_WRDS`
2. `core/data/reference/wrds/compustat_fundq_extract.csv`
3. fallback open source bootstrap

En mode production, le fetching ne retombe plus silencieusement sur le S&P 500 complet: il intersecte l’univers PIT avec les symboles stock CFD explicitement présents dans le snapshot XTB et sauvegarde aussi les snapshots broker et l’univers tradable courant.

État vérifié localement:

- `sp500_membership_history.csv`: `910` tickers historiques
- `sp500_fundamentals_history.csv`: `517` tickers XTB couverts
- `dataset_2004_2025.parquet`: `1 634 851` lignes x `111` colonnes

## Data cleaning

- `core/data/data_cleaning/dataset_cleaned_2004_2025.parquet`
  - dataset nettoyé prêt pour feature engineering

## Features engineering

- `core/data/features_engineering/dataset_features_2004_2025.parquet`
  - features tabulaires complètes + lags
- `core/data/features_engineering/dataset_features_2004_2025_sample_5pct.csv`
  - échantillon CSV de debug

Note pratique:

- dans l’état actuel du workspace, `features_engineering` n’a pas encore été rerun jusqu’au bout sur les nouveaux artefacts régénérés, donc ces deux fichiers peuvent être absents tant que l’étape n’a pas été rejouée.

## Data preprocessing

- `core/data/data_preprocessing/dataset_preprocessed_2009_2025.parquet`
  - dataset principal du méta-modèle
- `core/data/data_preprocessing/dataset_preprocessed_train.parquet`
- `core/data/data_preprocessing/dataset_preprocessed_val.parquet`
- `core/data/data_preprocessing/dataset_preprocessed_test.parquet`
- `core/data/data_preprocessing/feature_registry.parquet`
- `core/data/data_preprocessing/feature_registry.json`
- `core/data/data_preprocessing/feature_schema_manifest.json`
- `core/data/data_preprocessing/research_label_panel.parquet`

### Colonnes critiques du dataset préprocessé

- `date`
- `ticker`
- `dataset_split`
- `target_main`
- `target_intraday_open_to_close_log_return`
- `target_intraday_open_to_close_net_log_return`
- `benchmark_intraday_open_to_close_net_log_return`
- `target_intraday_open_to_close_excess_log_return`
- `target_intraday_open_to_close_sector_residual_log_return`
- `target_intraday_open_to_close_net_cs_zscore`
- `target_intraday_open_to_close_net_cs_rank`
- `target_overnight_close_to_next_open_net_log_return`
- `target_short_hold_1d_to_2d_net_log_return`
- `target_medium_hold_3d_to_5d_log_return`
- `target_medium_hold_3d_to_5d_net_log_return`

## Feature selection

- `core/data/feature_selection/feature_stability_scores.parquet`
- `core/data/feature_selection/feature_stability_scores.csv`
- `core/data/feature_selection/feature_stability_selected.parquet`
- `core/data/feature_selection/feature_stability_selected.csv`
- `core/data/feature_selection/dataset_preprocessed_feature_selected.parquet`
- `core/data/feature_selection/dataset_preprocessed_feature_selected_sample_5pct.csv`
- `core/data/feature_selection/feature_registry.parquet`
- `core/data/feature_selection/feature_registry.json`
- `core/data/feature_selection/feature_schema_manifest.json`

Le dataset filtré est désormais la source canonique pour `optimize_parameters` et `evaluate`.

## Optimize parameters

- `core/data/optimize_parameters/xgboost_optuna_trials.parquet`
- `core/data/optimize_parameters/xgboost_optuna_trials.csv`
- `core/data/optimize_parameters/xgboost_best_params.json`
- `core/data/optimize_parameters/trial_ledger.parquet`
- `core/data/optimize_parameters/trial_ledger.csv`
- `core/data/optimize_parameters/overfitting_report.json`

Les trials stockent maintenant des colonnes du type:

- `fold_1_daily_rank_ic`
- `fold_1_daily_rank_ic_full_window`
- `fold_1_daily_rank_ic_window_std`
- `mean_daily_rank_ic`
- `std_daily_rank_ic`
- `objective_score`

## Evaluate

- `core/data/evaluate/test_predictions.parquet`
  - prédictions sur les dates d’exécution
  - inclut aussi `signal_date`
  - inclut désormais `model_name`
- `core/data/evaluate/backtest_trades.parquet`
  - inclut désormais `model_name`
- `core/data/evaluate/backtest_daily.parquet`
  - inclut désormais `model_name`, `benchmark_return`, `turnover`, `gross_exposure`, `net_exposure`
- `core/data/evaluate/backtest_summary.json`
- `core/data/evaluate/model_leaderboard.json`
  - comparaison des modèles promouvables
- `core/data/evaluate/manual_orders.csv`
- `core/data/evaluate/manual_watchlist.csv`
- `core/data/evaluate/execution_checklist.json`
- `core/data/evaluate/post_trade_reconciliation.parquet`
- `core/data/evaluate/overfitting_report.json`

## Broker XTB

- `core/data/broker_xtb/xtb_tradable_universe.parquet`
- `core/data/broker_xtb/xtb_specs_snapshot.json`
- `core/data/broker_xtb/xtb_swap_snapshot.json`
- `core/data/broker_xtb/xtb_margin_snapshot.json`

### Contrat des prédictions

Le fichier de prédictions contient désormais deux notions temporelles:

- `signal_date`
  - date où le modèle voit les features
- `date`
  - date où la prédiction est envoyée au backtest, donc date d’exécution

## Ce qu’il faut retenir

Si tu dois relire rapidement les artefacts qui gouvernent réellement le modèle aujourd’hui:

1. `dataset_preprocessed_feature_selected.parquet`
2. `feature_stability_selected.parquet`
3. `xgboost_best_params.json`
4. `model_leaderboard.json`
5. `backtest_summary.json`
