# Meta Model Pipeline

Le pipeline canonique `meta_model` suit maintenant une chaîne stricte, pensée pour rester comparable d’une run à l’autre tout en séparant clairement:

- l’optimisation du modèle primaire,
- le raffinage du signal par méta-labeling,
- l’optimisation portefeuille,
- l’évaluation économique finale.

## Ordre de lancement canonique

1. `core/src/meta_model/broker_xtb/main.py`
2. `core/src/meta_model/data/data_reference/main.py`
3. `core/src/meta_model/launch/main.py`
4. `core/src/meta_model/data/data_fetching/main.py`
5. `core/src/meta_model/data/data_cleaning/main.py`
6. `core/src/meta_model/features_engineering/main.py`
7. `core/src/meta_model/data/data_preprocessing/main.py`
8. `core/src/meta_model/feature_selection/main.py`
9. `core/src/meta_model/optimize_parameters/main.py`
10. `core/src/meta_model/meta_labeling/main.py`
11. `core/src/meta_model/portfolio_optimization/main.py`
12. `core/src/meta_model/evaluate/main.py`

## Logique de la chaîne aval

### 1. Modèle primaire

`optimize_parameters` choisit les meilleurs hyperparamètres du modèle primaire sur un protocole walk-forward interne au train. Le modèle primaire reste une régression cross-sectionnelle.

### 2. Méta-labeling

`meta_labeling` construit ensuite un panneau de prédictions strictement OOS du modèle primaire:

- burn sur les premiers `20%` des dates train,
- retrain quotidien causal du primaire sur `train` uniquement,
- prédictions OOS sur le reste du train puis sur `val`.

Ce stage entraîne ensuite un classifieur méta qui apprend si le trade a une probabilité élevée de battre le benchmark sur l’horizon `5 sessions`.

### 3. Portefeuille

`portfolio_optimization` ne refit plus le signal. Il consomme les panneaux produits par `meta_labeling`, calibre le signal primaire en rendement espéré, puis le pondère par la confiance du méta-modèle avant de tuner les paramètres du solveur MIQP.

### 4. Évaluation finale

`evaluate` rejoue ensuite la chaîne complète sur `test`:

- prédiction quotidienne du primaire,
- scoring quotidien du méta-modèle,
- raffinage du signal,
- allocation MIQP,
- backtest final.

## Artefacts aval à connaître

### `core/data/meta_labeling/`

- `primary_oos_panel_train_tail.parquet`
- `primary_oos_panel_val.parquet`
- `meta_train_oof_predictions.parquet`
- `meta_val_predictions.parquet`
- `meta_best_params.json`
- `meta_model.json`

### `core/data/portfolio_optimization/`

- `portfolio_best_params.json`
- `portfolio_trial_ledger.parquet`
- `portfolio_validation_summary.json`

### `core/data/evaluate/`

- `test_predictions.parquet`
- `backtest_trades.parquet`
- `backtest_daily.parquet`
- `backtest_summary.json`

## Règle importante

Le split `val` reste réservé à l’optimisation portefeuille.  
Le méta-modèle est entièrement fit/tuné **à l’intérieur du train post-burn**, ce qui évite de surconsommer `val` et garde la comparaison des runs plus propre.
