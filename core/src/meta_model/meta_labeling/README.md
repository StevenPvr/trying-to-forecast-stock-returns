# Meta Labeling

## Objectif

Cette étape ajoute une deuxième couche de décision au-dessus du modèle primaire.

Le modèle primaire continue à produire un score d’alpha cross-sectionnel. Le méta-modèle apprend ensuite **quand ce score a de bonnes chances de battre le benchmark** sur l’horizon `5 sessions`.

Le résultat n’est pas un nouveau backtest autonome. C’est un **signal raffiné** qui sera ensuite consommé par `portfolio_optimization` puis `evaluate`.

## Contrat temporel

Le stage reste strictement causal:

- aucun split `test` n’est chargé,
- burn sur les premiers `20%` des dates uniques du split `train`,
- à partir de là, le modèle primaire est réentraîné chaque jour avec les meilleurs hyperparamètres figés,
- seules les lignes dont le label est déjà mature à la date de décision peuvent entrer dans l’entraînement du primaire.

Les prédictions ainsi produites sont donc réellement **out-of-sample** au sens temporel.

## Label méta

Le label du classifieur est volontairement simple et directement économique:

- `meta_label = 1` si `target_week_hold_excess_log_return > 0`
- `0` sinon

Autrement dit, le méta-modèle répond à la question:

> “Ce trade a-t-il de bonnes chances de battre le benchmark sur l’horizon visé ?”

## Features du méta-modèle

Le classifieur consomme:

- les features déjà retenues par `feature_selection`,
- plus les dérivées du score primaire:
  - `primary_prediction`
  - `primary_prediction_rank_cs`
  - `primary_prediction_zscore_cs`
  - `primary_prediction_abs`
  - `primary_prediction_sign`

## Sorties

### Panels produits

- `primary_oos_panel_train_tail.parquet`
- `primary_oos_panel_val.parquet`
- `meta_train_oof_predictions.parquet`
- `meta_val_predictions.parquet`

### Artefacts figés

- `meta_best_params.json`
- `meta_model.json`

## Comment le signal est raffiné

Le méta-modèle produit une probabilité calibrée `meta_probability`.

On en déduit:

- `meta_confidence = max(0, 2 * meta_probability - 1)`

Puis:

- `refined_prediction = primary_prediction * meta_confidence`
- `refined_expected_return_5d = primary_expected_return_5d * meta_confidence`

Cette règle garde le signe et l’amplitude du primaire quand la confiance est forte, mais coupe naturellement les trades peu crédibles.

## Place dans le pipeline

L’ordre canonique aval est:

1. `optimize_parameters`
2. `meta_labeling`
3. `portfolio_optimization`
4. `evaluate`

Le stage portefeuille n’a plus à reconstruire seul le signal prédictif. Il part directement des sorties figées de ce dossier.
