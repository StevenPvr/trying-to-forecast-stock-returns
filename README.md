# Prevision S&P 500

## Pipeline du méta-modèle

Le pipeline principal du projet est maintenant celui du `meta_model`.

L'enchaînement correct est :

1. `data_fetching`
   - script : [core/src/meta_model/data/data_fetching/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/data_fetching/main.py)
   - rôle : récupère et fusionne les données marché, macro, calendrier, sentiment, cross-asset et univers S&P 500
   - sortie principale : `core/data/data_fetching/dataset_2004_2025.parquet`

2. `data_cleaning`
   - script : [core/src/meta_model/data/data_cleaning/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/data_cleaning/main.py)
   - rôle : nettoie le dataset fetché
   - entrée : sortie de `data_fetching`
   - sortie principale : `core/data/data_cleaning/dataset_cleaned_2004_2025.parquet`

3. `features_engineering`
   - script : [core/src/meta_model/features_engineering/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/features_engineering/main.py)
   - rôle : calcule toutes les features du méta-modèle, y compris les features `deep_*` et les lags
   - entrée : dataset nettoyé
   - sortie principale : `core/data/features_engineering/dataset_features_2004_2025.parquet`

4. `feature_corr_pca`
   - script : [core/src/meta_model/feature_corr_pca/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/feature_corr_pca/main.py)
   - rôle : détecte les groupes de features très corrélées sur le `train`, applique le `Kernel PCA` sur ces groupes et produit le mapping JSON associé
   - entrée : sortie de `features_engineering`
   - sortie principale : `core/data/feature_corr_pca/dataset_features_corr_pca_2004_2025.parquet`

5. `greedy_forward_selection`
   - script : [core/src/meta_model/feature_selection_lag/greedy_forward_selection/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/feature_selection_lag/greedy_forward_selection/main.py)
   - rôle : sélectionne un sous-ensemble de features à partir de la sortie `feature_corr_pca`
   - entrée : sortie de `feature_corr_pca`
   - sortie principale : `core/data/feature_selection/dataset_features_greedy_forward_selected.parquet`

6. `data_preprocessing`
   - script : [core/src/meta_model/data/data_preprocessing/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/data/data_preprocessing/main.py)
   - rôle : crée `target_main`, assigne les splits `train/val/test`, forward-fill les features par ticker, retire les lignes/colonnes invalides et prune certaines corrélations résiduelles
   - entrée : sortie de `greedy_forward_selection`
   - sortie principale : `core/data/data_preprocessing/dataset_preprocessed_2009_2025.parquet`
   - sorties annexes :
     - `dataset_preprocessed_train.parquet`
     - `dataset_preprocessed_val.parquet`
     - `dataset_preprocessed_test.parquet`

7. `optimize_parameters`
   - script : [core/src/meta_model/optimize_parameters/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/optimize_parameters/main.py)
   - rôle : optimise les hyperparamètres XGBoost sur la sortie préprocessée
   - entrée : sortie de `data_preprocessing`
   - sortie principale : `core/data/optimize_parameters/xgboost_best_params.json`

8. `evaluate`
   - script : [core/src/meta_model/evaluate/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/evaluate/main.py)
   - rôle : recharge le dataset préprocessé et les meilleurs hyperparamètres, entraîne le modèle final en walk-forward daily retrain, produit les prédictions de test et le backtest
   - entrées :
     - sortie de `data_preprocessing`
     - sortie de `optimize_parameters`
   - sorties principales :
     - `core/data/evaluate/test_predictions.parquet`
     - `core/data/evaluate/backtest_trades.parquet`
     - `core/data/evaluate/backtest_daily.parquet`
     - `core/data/evaluate/backtest_summary.json`

## Résumé court

Le flux exact du méta-modèle est :

`data_fetching -> data_cleaning -> features_engineering -> feature_corr_pca -> greedy_forward_selection -> data_preprocessing -> optimize_parameters -> evaluate`

## Important

- `evaluate` est uniquement le pipeline de backtest du méta-modèle.
- Les `secondary_model` ne passent pas par `evaluate`.
- Les `secondary_model` branchent actuellement à partir de la sortie de `features_engineering`, via [core/src/secondary_model/data/data_preprocessing/main.py](/Users/steven/Programmation/prevision-sp500/core/src/secondary_model/data/data_preprocessing/main.py).
