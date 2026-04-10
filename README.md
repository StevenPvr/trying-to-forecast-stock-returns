# prevision-sp500

Pipeline de recherche broker-aware pour la prédiction cross-sectionnelle des stock returns, leur raffinage par méta-labeling, puis la construction d’un book CFD XTB discipliné.

Le contrat de recherche du `meta_model` a été recentré autour de principes plus crédibles:

- targets broker-aware alignés sur une exécution différée d’un jour,
- target d’entraînement cross-sectionnel standardisé,
- embargo temporel cohérent avec la réalisation du label,
- optimisation orientée `daily rank IC` avec diagnostics anti-overfitting,
- évaluation enrichie avec diagnostics de signal, contraintes portefeuille et exports d’exécution manuelle.

## Lecture rapide

- Vue d’ensemble du repo: [docs/architecture.md](/Users/steven/Programmation/prevision-sp500/docs/architecture.md)
- Contrat de recherche du méta-modèle: [docs/meta-model-research-contract.md](/Users/steven/Programmation/prevision-sp500/docs/meta-model-research-contract.md)
- Registre de features et manifests de schéma: [docs/feature-registry.md](/Users/steven/Programmation/prevision-sp500/docs/feature-registry.md)
- Registre de modèles et promotion: [docs/model-registry.md](/Users/steven/Programmation/prevision-sp500/docs/model-registry.md)
- Vue d’ensemble du sous-système `meta_model`: [core/src/meta_model/README.md](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/README.md)
- Moteur portefeuille et backtest: [docs/portfolio-engine.md](/Users/steven/Programmation/prevision-sp500/docs/portfolio-engine.md)
- Artefacts et datasets produits: [docs/data-artifacts.md](/Users/steven/Programmation/prevision-sp500/docs/data-artifacts.md)
- Runbook de lancement: [docs/launch-runbook.md](/Users/steven/Programmation/prevision-sp500/docs/launch-runbook.md)
- Tests, qualité et limites connues: [docs/testing-and-quality.md](/Users/steven/Programmation/prevision-sp500/docs/testing-and-quality.md)
- PRD de redressement: [urgent-todo.md](/Users/steven/Programmation/prevision-sp500/urgent-todo.md)

## Pipeline principal actuel

Le pipeline du `meta_model` est orchestré par des `main.py` et suit désormais un seul chemin canonique.

1. `broker_xtb`
   - [core/src/meta_model/broker_xtb/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/broker_xtb/main.py)
   - Régénère le snapshot officiel XTB et alimente la référence broker stricte utilisée par le reste du pipeline.

2. `data_reference`
   - [core/src/meta_model/data/data_reference/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/data_reference/main.py)
   - Régénère les références PIT du pipeline:
     - historique de membership S&P 500,
     - fondamentaux PIT filtrés sur l’univers XTB,
     - priorité `WRDS direct -> extract WRDS -> bootstrap open source`.

3. `launch`
   - [core/src/meta_model/launch/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/launch/main.py)
   - Vérifie la readiness minimale de lancement:
     - présence des références obligatoires,
     - snapshot XTB strict,
     - disponibilité `lightgbm`,
     - présence de `FRED_API_KEY`.

4. `data_fetching`
   - [core/src/meta_model/data/data_fetching/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/data_fetching/main.py)
   - Construit le panel brut marché/macro/calendrier/sentiment/cross-asset sur un univers PIT déjà intersecté avec les stock CFDs XTB explicitement présents dans le snapshot broker.

5. `data_cleaning`
   - [core/src/meta_model/data/data_cleaning/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/data_cleaning/main.py)
   - Nettoie le panel avant feature engineering.

6. `features_engineering`
   - [core/src/meta_model/features_engineering/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/features_engineering/main.py)
   - Produit les features tabulaires et leurs lags.

7. `data_preprocessing`
   - [core/src/meta_model/data/data_preprocessing/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/data_preprocessing/main.py)
   - Repart directement du dataset de features canonique.
   - Construit un panel de labels broker-aware avec primaire `intraday_open_to_close`, assigne les splits, applique une politique d’imputation pilotée par registre et fige un contrat de colonnes traçable.

8. `feature_selection`
   - [core/src/meta_model/feature_selection/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/feature_selection/main.py)
   - Score les features uniquement sur le split `train`, mesure leur stabilité par folds temporels, puis prune la redondance avant de produire le dataset canonique pour le modeling.

9. `optimize_parameters`
   - [core/src/meta_model/optimize_parameters/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/optimize_parameters/main.py)
   - Tune XGBoost sur le dataset filtré, avec un objectif de `daily rank IC` robuste, un ledger de trials et un rapport d’overfitting.

10. `meta_labeling`
   - [core/src/meta_model/meta_labeling/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/meta_labeling/main.py)
   - Produit les prédictions strictement OOS du modèle primaire sur le train post-burn puis sur la validation, entraîne un classifieur benchmark-relative, et fige le signal raffiné consommé par l’aval.

11. `portfolio_optimization`
   - [core/src/meta_model/portfolio_optimization/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/portfolio_optimization/main.py)
   - Tune le solveur MIQP sur la validation en consommant le signal raffiné (`primary + meta`) et fige les paramètres portefeuille.

12. `model_registry`
   - [core/src/meta_model/model_registry/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/model_registry/main.py)
   - Fournit les modèles comparés sur le même protocole: `ridge`, `elastic_net`, `factor_composite`, `xgboost`, et `lightgbm`.
   - Ce module est un registre partagé consommé par `optimize_parameters` et `evaluate`, pas une étape CLI distincte à lancer séparément.

13. `evaluate`
   - [core/src/meta_model/evaluate/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/evaluate/main.py)
   - Rejoue le primaire, le méta-modèle et le solveur MIQP sur le split `test`, puis produit les métriques signal, portefeuille et les exports d’exécution.

14. `broker_xtb_bridge`
   - [core/src/meta_model/broker_xtb](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/broker_xtb)
   - Produit les snapshots broker, l’univers tradable XTB, les estimations de coûts/marge, et les exports `manual_orders.csv` / `manual_watchlist.csv` / `execution_checklist.json`.

## Contrat de modélisation

Le contrat central partagé vit dans [core/src/meta_model/model_contract.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/model_contract.py).

Points clés:

- `signal_date` = date à laquelle les features sont observées.
- `execution_lag_days = 1` = les prédictions sont évaluées et backtestées à `t+1`.
- `hold_period_days = 0` par défaut pour le primaire intraday.
- `target_main` = label intraday brut `t+1 open -> t+1 close`.
- `target_week_hold_net_cs_zscore` = target primaire du modèle de régression.
- le méta-modèle apprend ensuite `1{target_week_hold_excess_log_return > 0}` sur des prédictions OOS du primaire.
- le preprocessing produit aussi des labels `overnight`, `short_hold_1d_to_2d` et `medium_hold_3d_to_5d`.
- la promotion finale compare `ridge`, `elastic_net`, `factor_composite`, `xgboost`, et éventuellement `lightgbm`.
- la promotion finale est bloquée si les diagnostics `PBO` / `DSR` échouent.

## Chaîne d’exécution complète

La chaîne canonique complète, dans l’ordre exact, est la suivante:

```bash
.venv/bin/python core/src/meta_model/broker_xtb/main.py
.venv/bin/python core/src/meta_model/data/data_reference/main.py
.venv/bin/python core/src/meta_model/launch/main.py
.venv/bin/python core/src/meta_model/data/data_fetching/main.py
.venv/bin/python core/src/meta_model/data/data_cleaning/main.py
.venv/bin/python core/src/meta_model/features_engineering/main.py
.venv/bin/python core/src/meta_model/data/data_preprocessing/main.py
.venv/bin/python core/src/meta_model/feature_selection/main.py
.venv/bin/python core/src/meta_model/optimize_parameters/main.py
.venv/bin/python core/src/meta_model/meta_labeling/main.py
.venv/bin/python core/src/meta_model/portfolio_optimization/main.py
.venv/bin/python core/src/meta_model/evaluate/main.py
```

Rôle de chaque étape:

1. `broker_xtb/main.py`
   - régénère le snapshot broker XTB officiel
2. `data_reference/main.py`
   - régénère les références PIT S&P 500 + fondamentaux WRDS/XTB
3. `launch/main.py`
   - vérifie la readiness minimale de lancement
4. `data_fetching/main.py`
   - construit le dataset marché + date-level XTB-first
5. `data_cleaning/main.py`
   - nettoie le dataset fetché
6. `features_engineering/main.py`
   - construit les features tabulaires et leurs lags
7. `data_preprocessing/main.py`
   - construit les labels broker-aware, splits, imputation et registre
8. `feature_selection/main.py`
   - sélectionne les features stables train-only
9. `optimize_parameters/main.py`
   - tune les modèles et produit le ledger / rapport anti-overfitting
10. `meta_labeling/main.py`
   - produit le panel OOS du primaire, entraîne le classifieur méta et fige le signal raffiné
11. `portfolio_optimization/main.py`
   - tune le solveur portefeuille MIQP sur le signal raffiné
12. `evaluate/main.py`
   - rejoue la chaîne complète sur `test` et produit les exports d’exécution manuelle

## Commandes utiles

Pré-requis canoniques avant un run `data_fetching` XTB-first:

- `core/data/reference/sp500_membership_history.csv`
- `core/data/reference/sp500_fundamentals_history.csv`
- `core/data/reference/xtb/xtb_instrument_specs.json`
- `FRED_API_KEY` dans `.env`

Variables optionnelles mais recommandées:

- `TIINGO_API_KEY` pour le fallback prix,
- `ID_WRDS` et `PASSWORD_WRDS` pour la génération directe des fondamentaux WRDS.

Régénérer le snapshot XTB officiel:

```bash
.venv/bin/python core/src/meta_model/broker_xtb/main.py
```

Régénérer les références PIT:

```bash
.venv/bin/python core/src/meta_model/data/data_reference/main.py
```

Vérifier la readiness de lancement:

```bash
.venv/bin/python core/src/meta_model/launch/main.py
```

Lancer uniquement le cœur de la chaîne une fois le bootstrap déjà à jour:

```bash
.venv/bin/python core/src/meta_model/data/data_fetching/main.py
.venv/bin/python core/src/meta_model/data/data_cleaning/main.py
.venv/bin/python core/src/meta_model/features_engineering/main.py
.venv/bin/python core/src/meta_model/data/data_preprocessing/main.py
.venv/bin/python core/src/meta_model/feature_selection/main.py
.venv/bin/python core/src/meta_model/optimize_parameters/main.py
.venv/bin/python core/src/meta_model/meta_labeling/main.py
.venv/bin/python core/src/meta_model/portfolio_optimization/main.py
.venv/bin/python core/src/meta_model/evaluate/main.py
```

Exécuter toute la suite de tests:

```bash
.venv/bin/pytest tests -q
```

Le repo utilise `pytest --import-mode=importlib` dans [pyproject.toml](/Users/steven/Programmation/prevision-sp500/pyproject.toml), ce qui évite les collisions entre multiples `test_main.py`.

## État actuel vérifié

Éléments vérifiés localement dans cet état du repo:

- `launch/main.py` passe avec `is_ready = true`,
- `broker_xtb/main.py` régénère `1420` `stock_cfd` et `4` `index_cfd`,
- `data_reference/main.py` régénère:
  - `sp500_membership_history.csv` avec `910` tickers historiques,
  - `sp500_fundamentals_history.csv` avec `517` tickers XTB couverts,
- `data_fetching/main.py` s’exécute jusqu’au bout et produit:
  - `core/data/data_fetching/dataset_2004_2025.parquet`
  - `1 634 851` lignes
  - `111` colonnes
- `data_cleaning/main.py` s’exécute jusqu’au bout et produit:
  - `core/data/data_cleaning/dataset_cleaned_2004_2025.parquet`
  - `1 634 851` lignes
  - `109` colonnes

Points encore ouverts avant de dire que toute la chaîne est validée de bout en bout dans cet environnement précis:

- `features_engineering` n’a pas encore été revérifié jusqu’à sa sortie finale dans ce tour,
- `data_preprocessing`, `feature_selection`, `optimize_parameters` et `evaluate` restent couverts par les tests, mais pas encore relancés complètement sur les nouveaux artefacts frais de `core/data/`,
- certains anciens symboles XTB/Wikipedia n’ont pas de couverture prix satisfaisante chez les providers et sont donc éliminés au `data_fetching`.

Conclusion honnête: le repo est **prêt à être lancé** au sens opérationnel du bootstrap et des deux premiers étages validés en réel, mais la validation complète de la chaîne jusqu’à `evaluate` n’a pas encore été rejouée intégralement sur les artefacts fraîchement régénérés.
