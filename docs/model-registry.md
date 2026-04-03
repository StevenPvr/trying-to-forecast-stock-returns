# Registre de modèles

## Rôle

Le repo ne suppose plus qu’un seul modèle complexe mérite d’être évalué. Le registre de modèles dans [core/src/meta_model/model_registry/main.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/model_registry/main.py) force la comparaison sur un protocole unique.

## Modèles actifs

- `ridge`
  - baseline linéaire régularisée
  - utile pour savoir si la non-linéarité apporte vraiment quelque chose
- `elastic_net`
  - baseline linéaire sparse / régularisée
  - utile pour vérifier si une sélection implicite plus agressive améliore le ranking sans exploser l’instabilité
- `factor_composite`
  - baseline non entraînée
  - score cross-sectionnel à partir de familles simples: momentum, reversal, volatilité, size, liquidité
- `xgboost`
  - modèle non linéaire sélectionné par le pipeline d’optimisation
- `lightgbm`
  - activé seulement si la dépendance est disponible dans l’environnement

## Contrat commun

Chaque modèle passe par:

- `ModelSpec`
- `ModelArtifact`
- `fit_model(spec, train_frame, feature_columns)`
- `predict_model(artifact, frame, feature_columns)`

## Règle de promotion

Le pipeline d’évaluation classe les modèles avec l’ordre suivant:

1. `alpha_over_benchmark_net`
2. `daily_rank_ic_ir`
3. `calmar_ratio`

En plus de ce tri, un modèle doit passer les garde-fous:

- `PBO <= 0.20`
- `deflated_sharpe_ratio >= 0.10`
- `alpha_over_benchmark_net > 0`
- `daily_rank_ic_ir > 0`

Cela évite de promouvoir un modèle juste parce qu’il a un meilleur score statistique mais un moins bon portefeuille.

## Limites actuelles

- pas encore de régression facteurs publique type FF5 dans la boucle de promotion
- le `factor_composite` reste une baseline déterministe simple, pas un modèle de recherche final
