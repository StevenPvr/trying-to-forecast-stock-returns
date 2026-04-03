# Tests et qualité

## Configuration pytest

Le repo utilise maintenant `pytest` avec `--import-mode=importlib` dans [pyproject.toml](/Users/steven/Programmation/prevision-sp500/pyproject.toml).

Pourquoi:

- le dépôt contient de nombreux `test_main.py`,
- le mode d’import classique créait des collisions de modules,
- la collecte devenait instable même quand les tests eux-mêmes étaient valides.

## Suites validées pendant le redressement du PRD

La suite complète du repo actif passe maintenant:

```bash
.venv/bin/pytest tests -q
```

Résultat observé:

- `348 passed`

En complément, les suites suivantes ont été utilisées comme batterie ciblée pendant le redressement:

- [tests/data/data_preprocessing/test_main.py](/Users/steven/Programmation/prevision-sp500/tests/data/data_preprocessing/test_main.py)
- [tests/evaluate/test_main.py](/Users/steven/Programmation/prevision-sp500/tests/evaluate/test_main.py)
- [tests/feature_selection/test_main.py](/Users/steven/Programmation/prevision-sp500/tests/feature_selection/test_main.py)
- [tests/optimize_parameters/test_main.py](/Users/steven/Programmation/prevision-sp500/tests/optimize_parameters/test_main.py)
- [tests/features_engineering/test_main.py](/Users/steven/Programmation/prevision-sp500/tests/features_engineering/test_main.py)
- [tests/broker_xtb/test_main.py](/Users/steven/Programmation/prevision-sp500/tests/broker_xtb/test_main.py)
- [tests/overfitting/test_main.py](/Users/steven/Programmation/prevision-sp500/tests/overfitting/test_main.py)
- [tests/data/data_reference/test_reference_pipeline.py](/Users/steven/Programmation/prevision-sp500/tests/data/data_reference/test_reference_pipeline.py)
- [tests/data/data_reference/test_wrds_provider.py](/Users/steven/Programmation/prevision-sp500/tests/data/data_reference/test_wrds_provider.py)

Commande:

```bash
.venv/bin/pytest \
  tests/data/data_preprocessing/test_main.py \
  tests/evaluate/test_main.py \
  tests/feature_selection/test_main.py \
  tests/optimize_parameters/test_main.py \
  tests/features_engineering/test_main.py -q
```

- `62 passed`

## Vérifications live rejouées localement

En plus de `pytest`, les commandes suivantes ont été relancées avec succès sur les artefacts locaux:

```bash
.venv/bin/python core/src/meta_model/broker_xtb/main.py
.venv/bin/python core/src/meta_model/data/data_reference/main.py
.venv/bin/python core/src/meta_model/launch/main.py
.venv/bin/python core/src/meta_model/data/data_fetching/main.py
.venv/bin/python core/src/meta_model/data/data_cleaning/main.py
```

Résultats observés:

- `launch/main.py` retourne `is_ready = true`
- `data_reference/main.py` génère `517` tickers fondamentaux XTB
- `data_fetching/main.py` produit `1 634 851` lignes x `111` colonnes
- `data_cleaning/main.py` produit `1 634 851` lignes x `109` colonnes

## Invariants méthodologiques désormais couverts

### data_preprocessing

- construction du target forward-looking
- construction du target intraday broker-aware avec `execution_lag_days = 1`
- présence des nouvelles cibles dérivées
- non-exclusion automatique de la période Covid
- politique d’imputation compatible avec le registre de features
- persistance du registre et du manifest de schéma

### evaluate

- embargo de label cohérent avec le décalage d’exécution
- sélection top/bottom
- coût broker-aware par instrument
- effet sur l’équité quotidienne
- exports d’exécution manuelle et métriques de capacité/marge

### optimize_parameters

- CV walk-forward
- objective économique orientée `daily rank IC`
- persistance des colonnes de trial
- sélection one-standard-error
- ledger de trials et rapport `PBO`

### broker_xtb

- résolution des specs stock/index
- coût et marge broker-aware
- univers tradable filtré par spread
- bundle opérationnel `manual_orders` / `manual_watchlist`

### feature_selection

- scoring des features uniquement sur le split `train`
- stabilité mesurée par folds temporels chronologiques
- pruning de redondance avant l’opti et l’évaluation
- recâblage par défaut de l’opti et de l’éval vers le dataset sélectionné
- invariant `selected_features == filtered_dataset_features`
- persistance du manifest de schéma

### features_engineering

- fallback séquentiel si `ProcessPoolExecutor` n’est pas disponible
- lagging cohérent des features tabulaires actives

### data_reference / WRDS

- mapping WRDS `comp.security -> gvkey -> fundq`
- fallback bootstrap pour les tickers XTB non résolus par WRDS
- couverture finale `517 / 517` tickers XTB au niveau du fichier fondamental de référence

### Nettoyage du legacy

- suppression des suites dédiées à `feature_corr_pca`
- suppression des suites dédiées à `feature_selection_lag/greedy_forward_selection`
- suppression complète de la couche `secondary_model`
- suppression des suites orphelines `accounting_feature_engineering`
- suppression des suites orphelines `deep_feature_engineering`

## Dette de qualité restante

- renommer progressivement les `test_main.py` en fichiers uniques plus explicites,
- ajouter des tests de contrat plus proches de la recherche:
  - non-utilisation des colonnes `target_*` comme features
  - intégration d’un benchmark facteurs public plus riche
- rejouer toute la chaîne de `features_engineering` à `evaluate` sur les artefacts fraîchement régénérés pour une validation runtime complète, pas seulement un passage des tests
