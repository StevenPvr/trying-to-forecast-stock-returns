# Registre de features

## Pourquoi ce document existe

Le pipeline ne traite plus les features comme une simple liste de colonnes numériques. Chaque feature canonique possède désormais un contrat explicite produit et persisté dans [core/src/meta_model/data/registry.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/registry.py).

Ce registre sert à trois choses:

- gouverner l’imputation au preprocessing,
- tracer quelles colonnes sont autorisées dans le score alpha,
- figer un schéma vérifiable entre preprocessing, feature selection, optimisation et évaluation.

## Colonnes du registre

Chaque registre contient au minimum:

- `feature_name`
- `family`
- `source`
- `availability_lag_sessions`
- `safe_ffill_max_days`
- `missing_policy`
- `is_date_level`
- `is_cross_sectional`
- `enabled_for_alpha_model`

## Règles d’inférence actuelles

Les règles sont heuristiques mais centralisées:

- `macro_`
  - `availability_lag_sessions = 1`
  - `safe_ffill_max_days = 21`
  - `is_date_level = True`
- `sentiment_`
  - `availability_lag_sessions = 1`
  - `safe_ffill_max_days = 10`
- `calendar_`
  - `availability_lag_sessions = 1`
  - `safe_ffill_max_days = 1`
- `cross_asset_`
  - `availability_lag_sessions = 1`
  - `safe_ffill_max_days = 5`
- `company_`
  - `availability_lag_sessions = 1`
  - `safe_ffill_max_days = 63`
- `ta_`, `quant_`, `deep_`
  - `availability_lag_sessions = 0`
  - `safe_ffill_max_days = 5`
- `stock_`
  - pas de forward-fill par défaut
- `pred_*`
  - `missing_policy = "disallow"`
  - `enabled_for_alpha_model = False`

Les suffixes de type `_lag_5d` augmentent automatiquement `availability_lag_sessions`.

## Artefacts produits

### Au preprocessing

- `core/data/data_preprocessing/feature_registry.parquet`
- `core/data/data_preprocessing/feature_registry.json`
- `core/data/data_preprocessing/feature_schema_manifest.json`

### Après feature selection

- `core/data/feature_selection/feature_registry.parquet`
- `core/data/feature_selection/feature_registry.json`
- `core/data/feature_selection/feature_schema_manifest.json`

## Manifest de schéma

Le manifest contient:

- la liste ordonnée des features attendues,
- un `feature_schema_hash` dérivé de cette liste.

Ce hash est vérifié:

- dans [core/src/meta_model/optimize_parameters/dataset.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/optimize_parameters/dataset.py)
- dans [core/src/meta_model/evaluate/dataset.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/evaluate/dataset.py)

Effet recherché:

- empêcher qu’un dataset filtré et un modèle final utilisent silencieusement deux ensembles de features différents,
- autoriser quand même des datasets synthétiques de test hors pipeline canonique s’ils n’ont pas de manifest compagnon.

## Ce que le registre ne fait pas encore

- quotas par famille de features,
- priorisation économique explicite par feature,
- métadonnées de publication plus fines que la famille/source,
- séparation complète entre features de score alpha et features de risque overlay.
