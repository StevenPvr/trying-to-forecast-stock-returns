# Architecture du repo

## Objectif

Le repo vise à construire un pipeline de recherche pour la prédiction cross-sectionnelle des stock returns et leur traduction en signal exploitable.

La structure active est désormais recentrée sur une seule couche canonique:

- `meta_model` pour le pipeline complet de recherche, d’optimisation et d’évaluation,
- un périmètre de tests aligné uniquement sur les modules réellement supportés.

## Arborescence utile

### Cœur du pipeline principal

- [core/src/meta_model/data](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data)
  - fetching, cleaning, preprocessing, paths, constants, registre de features
- [core/src/meta_model/features_engineering](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/features_engineering)
  - features techniques, quant, deep, lags, post-processing
- [core/src/meta_model/feature_selection](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/feature_selection)
  - scoring train-only, stabilité temporelle, pruning de redondance, dataset filtré canonique
- [core/src/meta_model/optimize_parameters](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/optimize_parameters)
  - dataset bundle, CV walk-forward, objective, Optuna, sélection one-standard-error
- [core/src/meta_model/model_registry](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/model_registry)
  - modèles comparés sur le même protocole: ridge, elastic_net, composite factoriel, xgboost, lightgbm
- [core/src/meta_model/evaluate](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/evaluate)
  - entraînement walk-forward multi-modèles, backtest, promotion, persistance des outputs
- [core/src/meta_model/model_contract.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/model_contract.py)
  - contrat partagé des colonnes, du target et de l’alignement temporel
- [core/src/meta_model/research_metrics.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/research_metrics.py)
  - métriques de recherche orientées signal

### Tests

Le miroir de tests utile pour le pipeline actif se trouve sous:

- [tests/data/data_preprocessing](/Users/steven/Programmation/prevision-sp500/tests/data/data_preprocessing)
- [tests/features_engineering](/Users/steven/Programmation/prevision-sp500/tests/features_engineering)
- [tests/feature_selection](/Users/steven/Programmation/prevision-sp500/tests/feature_selection)
- [tests/optimize_parameters](/Users/steven/Programmation/prevision-sp500/tests/optimize_parameters)
- [tests/evaluate](/Users/steven/Programmation/prevision-sp500/tests/evaluate)

## Pipeline effectif

### 0. Bootstrap broker et références

- Entrées:
  - document officiel XTB
  - historique S&P 500 public
  - WRDS direct ou extract local
- Sorties:
  - `core/data/reference/xtb/xtb_instrument_specs.json`
  - `core/data/reference/sp500_membership_history.csv`
  - `core/data/reference/sp500_fundamentals_history.csv`

Le fetching canonique suppose ces trois artefacts déjà présents.

### 1. Data fetching

- Entrée: sources externes marché, macro, calendrier, sentiment, cross-asset, univers PIT S&P 500, snapshot broker XTB
- Sortie principale: `core/data/data_fetching/dataset_2004_2025.parquet`

### 2. Data cleaning

- Entrée: dataset fetché
- Sortie principale: `core/data/data_cleaning/dataset_cleaned_2004_2025.parquet`

### 3. Feature engineering

- Entrée: dataset nettoyé
- Sortie principale: `core/data/features_engineering/dataset_features_2004_2025.parquet`

### 4. Data preprocessing

- Entrée: `core/data/features_engineering/dataset_features_2004_2025.parquet`
- Rôle: construire les labels broker-aware intraday/overnight/short-hold, les splits, la politique d’imputation et le registre canonique des features
- Sortie principale: `core/data/data_preprocessing/dataset_preprocessed_2009_2025.parquet`
- Artefacts de contrat:
  - `core/data/data_preprocessing/feature_registry.parquet`
  - `core/data/data_preprocessing/feature_schema_manifest.json`

### 5. Feature selection

- Entrée: `core/data/data_preprocessing/dataset_preprocessed_2009_2025.parquet`
- Rôle: scorer toutes les features sur `train`, mesurer leur stabilité par blocs temporels, puis éliminer les redondances avant l’opti
- Sortie principale: `core/data/feature_selection/dataset_preprocessed_feature_selected.parquet`
- Artefacts de contrat:
  - `core/data/feature_selection/feature_registry.parquet`
  - `core/data/feature_selection/feature_schema_manifest.json`

### 6. Hyperparameter optimization

- Entrée: dataset filtré par feature selection
- Sorties:
  - `core/data/optimize_parameters/xgboost_optuna_trials.parquet`
  - `core/data/optimize_parameters/xgboost_best_params.json`

### 7. Evaluate / backtest

- Entrées: dataset filtré + meilleurs paramètres
- Sorties:
  - `core/data/evaluate/test_predictions.parquet`
  - `core/data/evaluate/backtest_trades.parquet`
  - `core/data/evaluate/backtest_daily.parquet`
  - `core/data/evaluate/backtest_summary.json`
  - `core/data/evaluate/model_leaderboard.json`

## Ce qui est canonique vs ce qui est encore incomplet

### Canonique

- `model_contract.py`
- target d’entraînement cross-sectionnel standardisé
- objective `daily rank IC`
- exécution décalée d’un jour dans `evaluate/training.py`
- diagnostics de signal dans `research_metrics.py`
- registre de modèles avec promotion benchmark-relative
- moteur portefeuille avec benchmark, turnover, exposition et beta réalisés

### Expérimental ou optionnel

- extensions futures de baselines modèles,
- sophistication portefeuille au-delà du moteur actuel,
- overlays de régime et overlays risque séparés du score alpha central.

## Lecture conseillée

Pour comprendre rapidement le repo sans se perdre:

1. [docs/meta-model-research-contract.md](/Users/steven/Programmation/prevision-sp500/docs/meta-model-research-contract.md)
2. [core/src/meta_model/model_contract.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/model_contract.py)
3. [core/src/meta_model/data/data_preprocessing/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/data_preprocessing/main.py)
4. [core/src/meta_model/feature_selection/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/feature_selection/main.py)
5. [core/src/meta_model/optimize_parameters/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/optimize_parameters/main.py)
6. [core/src/meta_model/evaluate/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/evaluate/main.py)
