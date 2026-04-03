# Contrat de recherche du meta_model

## Pourquoi ce document existe

Le repo a longtemps mélangé plusieurs définitions implicites du problème:

- target brut close-to-close,
- objectif RMSE,
- exécution ambiguë le même jour,
- prédictions secondaires fusionnées sans contrat clair.

Le contrat actuel vise à rendre le pipeline plus falsifiable et plus proche d’un vrai problème d’alpha.

## Contrat central

Le contrat partagé vit dans [core/src/meta_model/model_contract.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/model_contract.py).

### Colonnes de base

- `date`
  - date d’exécution après décalage dans les prédictions de backtest
- `signal_date`
  - date de génération du signal avant décalage
- `ticker`
- `dataset_split`

### Targets produits par le preprocessing

Le preprocessing dans [core/src/meta_model/data/data_preprocessing/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/data_preprocessing/main.py) construit plusieurs cibles broker-aware à partir de `stock_open_price` et `stock_close_price`.

- `target_main`
  - label intraday brut `t+1 open -> t+1 close`
  - utile pour la réalisation économique du backtest
- `target_intraday_open_to_close_net_log_return`
  - label primaire net des coûts broker estimés
- `benchmark_intraday_open_to_close_net_log_return`
  - benchmark cross-sectionnel du label primaire net
- `target_intraday_open_to_close_excess_log_return`
  - version excédentaire vs benchmark cross-sectionnel
- `target_intraday_open_to_close_sector_residual_log_return`
  - version residualisée secteur du label primaire net
- `target_intraday_open_to_close_net_cs_zscore`
  - target d’entraînement par défaut
- `target_intraday_open_to_close_net_cs_rank`
  - version rankée utile en analyse
- `target_overnight_close_to_next_open_net_log_return`
  - label secondaire de gap overnight
- `target_short_hold_1d_to_2d_net_log_return`
  - label secondaire short swing
- `target_medium_hold_3d_to_5d_log_return`
  - label brut expérimental moyen terme
- `target_medium_hold_3d_to_5d_net_log_return`
  - label net expérimental moyen terme

## Alignement temporel

### Convention actuelle

- `execution_lag_days = 1`
- `hold_period_days = 0` pour le primaire intraday

Conséquence:

- les features observées à la date de signal `t` ne sont pas monétisées à `t`,
- le modèle primaire est évalué sur un rendement `t+1 open -> t+1 close`,
- l’embargo de label utilisé pour l’entraînement walk-forward vaut `1` jour pour le primaire.

### Où cela est implémenté

- création des targets: [core/src/meta_model/data/data_preprocessing/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/data_preprocessing/main.py)
- embargo d’entraînement: [core/src/meta_model/evaluate/training.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/evaluate/training.py)
- décalage de la date de prédiction vers la date d’exécution: [core/src/meta_model/evaluate/training.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/evaluate/training.py)

## Politique d’imputation et registre de features

Le forward-fill n’est plus illimité ni uniquement piloté par des heuristiques locales.

Le contrat passe désormais par [core/src/meta_model/data/registry.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/registry.py), qui assigne à chaque feature:

- une famille,
- une source,
- un `availability_lag_sessions`,
- un `safe_ffill_max_days`,
- une politique de missingness,
- un statut `enabled_for_alpha_model`.

Familles principales:

- `ta_`, `quant_`, `deep_`, `cross_asset_`
  - `ffill` limité à 5 jours
- `calendar_`
  - `ffill` limité à 1 jour
- `sentiment_`
  - `ffill` limité à 10 jours
- `macro_`
  - `ffill` limité à 21 jours
- `company_`
  - `ffill` limité à 63 jours

Les artefacts de contrat produits au preprocessing sont:

- `core/data/data_preprocessing/feature_registry.parquet`
- `core/data/data_preprocessing/feature_registry.json`
- `core/data/data_preprocessing/feature_schema_manifest.json`

## Feature Selection

La sélection de features est désormais une étape explicite entre `data_preprocessing` et `optimize_parameters`.

Implémentation:

- scoring univarié uniquement sur le split `train`
- mesure de stabilité sur plusieurs blocs temporels chronologiques
- score final = utilité absolue moins pénalité d’instabilité
- pruning de redondance par corrélation avant de fixer le sous-ensemble final

Le point d’entrée est [core/src/meta_model/feature_selection/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/feature_selection/main.py).

Conséquence importante:

- `optimize_parameters` et `evaluate` lisent désormais par défaut le dataset filtré `dataset_preprocessed_feature_selected.parquet`
- la sélection est faite sans utiliser les splits `val` et `test`
- la réduction de dimension est enfin séparée proprement du tuning modèle
- un manifest de schéma empêche désormais une dérive silencieuse entre dataset sélectionné, opti et évaluation

## Optimisation

### Objectif actuel

Le tuning dans [core/src/meta_model/optimize_parameters/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/optimize_parameters/main.py) utilise maintenant:

- un custom metric `daily_rank_ic`,
- un objectif agrégé qui maximise l’IC moyen,
- une pénalisation de l’instabilité entre folds,
- une pénalisation de l’instabilité entre fenêtres train,
- une pénalisation de complexité.
- un ledger de trials et un rapport d’overfitting avec `PBO`.

La métrique partagée est implémentée dans [core/src/meta_model/research_metrics.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/research_metrics.py).

### Pourquoi le modèle reste un régressseur

Le modèle reste un XGBoost `reg:squarederror`, mais:

- la target d’entraînement est cross-sectionnellement standardisée,
- l’early stopping est piloté par `daily_rank_ic`,
- la sélection d’hyperparamètres est pilotée par l’objectif rank-IC et non plus par le RMSE.

## Évaluation

Le backtest dans [core/src/meta_model/evaluate/backtest.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/evaluate/backtest.py) utilise:

- la date d’exécution déjà décalée,
- le realized return brut `target_forward_log_return_5d`,
- les coûts transactionnels et de financement existants,
- un benchmark cross-sectionnel égal-pondéré,
- le turnover et les expositions journalières,
- une variante de neutralité pilotée par `neutrality_mode`.

Le pipeline d’évaluation compare maintenant plusieurs modèles sur le même protocole:

- `ridge`
- `elastic_net`
- `factor_composite`
- `xgboost`
- `lightgbm` si disponible

La promotion finale du modèle se fait par tri sur:

1. `alpha_over_benchmark_net`
2. `daily_rank_ic_ir`
3. `sharpe_ratio`

Le pipeline d’évaluation enrichit maintenant le résumé avec:

- `daily_rank_ic_mean`
- `daily_rank_ic_std`
- `daily_rank_ic_ir`
- `daily_linear_ic_mean`
- `daily_top_bottom_spread_mean`
- `daily_top_bottom_spread_std`
- `annualized_benchmark_return`
- `alpha_over_benchmark_net`
- `calmar_ratio`
- `turnover_annualized`
- `average_gross_exposure`
- `average_net_exposure`
- `realized_beta`
- `capacity_binding_share`
- `margin_headroom`
- `deflated_sharpe_ratio`
- `pbo`

## Ce que ce contrat ne résout pas encore

- solveur d’allocation plus riche pour neutralisation secteur / industrie / bêta,
- benchmark implémentable explicite,
- régression facteurs publique type FF5,
- quotas ou contraintes explicites par famille de features,
- snapshots broker officiels plus riches que le provider par défaut.
