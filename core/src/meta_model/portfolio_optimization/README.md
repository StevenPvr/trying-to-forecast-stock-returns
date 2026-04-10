# Portfolio Optimization

## Objectif

Le moteur portefeuille canonique du repo est maintenant un **solveur MIQP** orienté exécution réelle retail XTB:

- pas d’actions fractionnaires
- budget cash exact
- exposition visée au plus près de `100%`
- positions existantes conservées jusqu’à leur sortie naturelle
- contraintes de concentration et de liquidité respectées

Le but n’est pas d’avoir un solveur “joli sur le papier”, mais un moteur qui prenne des décisions plausibles avec un petit capital réel, par exemple `1 000 EUR`.

Depuis l’ajout du stage `meta_labeling`, le solveur ne consomme plus uniquement le score primaire:

- le modèle primaire reste la source d’alpha,
- le méta-modèle fournit une probabilité de réussite benchmark-relative,
- le portefeuille optimise sur un **rendement espéré raffiné**:
  - `refined_expected_return_5d = primary_expected_return_5d * meta_confidence`

## Pourquoi un MIQP

Un solveur continu classique est insuffisant dans ce contexte:

- il propose des poids théoriques impossibles à exécuter quand une action vaut `300 EUR`, `700 EUR` ou `1 200 EUR`
- il laisse souvent trop de cash résiduel après arrondi
- il ne traite pas correctement l’indivisibilité des actions

Le problème réel est naturellement **mixte-entier**:

- `q_i` = nombre entier d’actions achetées
- `cash` = cash restant
- exposition finale = portefeuille existant + nouvelles actions entières

Le repo résout donc directement ce problème, au lieu de produire un portefeuille continu puis de l’arrondir après coup.

## Contrat temporel

Le contrat de trading reste volontairement simple et causal:

- le modèle ML prédit toujours un horizon `5 sessions`
- chaque jour, le solveur optimise uniquement la **nouvelle tranche d’entrée**
- les positions déjà ouvertes ne sont pas rebalancées artificiellement
- une position ouverte sort à sa date de sortie prévue, puis libère le cash

Autrement dit, le moteur portefeuille améliore la décision d’entrée quotidienne sans casser le contrat de labels du modèle.

## Formulation mathématique

### Variables

Pour chaque action candidate `i`:

- `q_i ∈ Z_+` : nombre entier d’actions à acheter
- `z_i ∈ {0, 1}` : activation de la ligne
- `cash >= 0` : cash résiduel

Grandeurs dérivées:

- `notional_i = price_i * q_i`
- `w_new_i = notional_i / equity`
- `w_total_i = w_existing_i + w_new_i`

### Objectif primaire

Le solveur maximise une utilité quadratique:

`mu^T w_total - lambda_risk * w_total' Sigma w_total - lambda_turnover * sum(w_new) - lambda_cost * sum(cost_proxy_i * w_new_i)`

Intuition:

- `mu` pousse vers les actifs au meilleur rendement espéré
- `Sigma` pénalise les portefeuilles trop concentrés en risque
- `lambda_turnover` évite de consommer du capital pour des micro-ajustements inutiles
- `lambda_cost` pénalise les ordres dont l’edge net est trop faible

### Exposition proche de 100%

Le solveur ne force pas bêtement `cash = 0`.

Il utilise une résolution **lexicographique en 2 passes**:

1. maximiser l’objectif primaire
2. imposer que cet objectif reste à moins de `primary_objective_tolerance_bps` de l’optimum
3. minimiser le cash résiduel

Cette stratégie est importante en trading réel:

- on n’achète pas une ligne médiocre juste pour “remplir”
- mais on évite aussi de laisser trop de cash quand de bonnes allocations entières existent

## Contraintes

Le solveur impose simultanément:

- budget exact: `sum(notional_i + cost_proxy_i * notional_i) + cash = cash_available`
- long-only strict
- pas de fractions
- minimum d’ordre XTB via `min_lot_shares`
- cap de ligne via `max_position_weight`
- cap sectoriel via `max_sector_weight`
- cap global via `gross_cap_fraction`
- cap de liquidité via `adv_cap_shares`
- filtre net-of-cost: pas d’entrée si l’edge net attendu est négatif après buffer

## Préfiltrage quotidien

Le solveur exact ne tourne pas sur “tout l’univers brut”.

Avant la résolution:

- on ne garde que les titres tradables
- on calcule `expected_return_5d`
- on retire les titres sans edge net positif
- on limite ensuite le pool à `miqp_candidate_pool_size`

Ce préfiltrage est indispensable en production:

- il réduit le temps de résolution
- il garde le solveur stable
- il n’introduit pas de règle fixe arbitraire sur le nombre final de lignes

Le nombre de positions finales reste endogène au solveur.

## Causalité et anti-leakage

Le stage `portfolio_optimization` reste strictement **train-only** pour tout ce qui est appris:

- calibration alpha fit sur OOF train
- covariance fit sur train
- tuning des hyperparamètres portefeuille sur train

Ensuite:

- `evaluate` recharge les paramètres gelés
- aucune re-optimisation de paramètres n’a lieu sur le test

## Solveur et runtime

Backend canonique:

- **SCIP**
- binding Python: **PySCIPOpt**

Paramètres runtime principaux:

- `miqp_time_limit_seconds`
- `miqp_relative_gap`
- `miqp_candidate_pool_size`
- `miqp_primary_objective_tolerance_bps`

Le solveur accepte les statuts pratiques de prod:

- `optimal`
- `timelimit`
- `gaplimit`

Ces statuts restent exploitables tant que la solution courante est cohérente.

## Flux complet

### Pendant `portfolio_optimization`

1. charger les panneaux figés produits par `meta_labeling`
2. calibrer le score primaire en `primary_expected_return_5d`
3. appliquer la confiance du méta-modèle pour obtenir `refined_expected_return_5d`
4. fitter la covariance sur train
5. lancer la recherche d’hyperparamètres avec le moteur MIQP sur tous les cœurs disponibles par défaut
6. figer les meilleurs paramètres dans `portfolio_best_params.json`

### Pendant `evaluate`

1. charger les paramètres portefeuille gelés
2. produire le score primaire quotidien
3. scorer le méta-modèle quotidien
4. convertir le score primaire en rendement espéré puis le raffiner
5. appeler le solveur MIQP à chaque date
6. écrire les diagnostics, les allocations, les trades et le résumé

## Artefacts clés

Dans `core/data/portfolio_optimization/`:

- `portfolio_best_params.json`
- `portfolio_trial_ledger.parquet`
- `portfolio_train_cv_daily.parquet`
- `portfolio_train_cv_allocations.parquet`
- `portfolio_validation_daily.parquet`
- `portfolio_validation_allocations.parquet`
- `portfolio_validation_summary.json`
- `risk_covariance.parquet`

Dans `core/data/evaluate/`:

- `portfolio_target_allocations.parquet`
- `portfolio_optimizer_daily.parquet`
- `portfolio_optimizer_summary.json`
- `backtest_trades.parquet`
- `backtest_daily.parquet`
- `backtest_summary.json`

## Comment lire les diagnostics

Colonnes utiles au quotidien:

- `solver_status`
- `candidate_count`
- `tradable_count`
- `cash_weight`
- `cash_amount_eur`
- `solve_time_seconds`
- `mip_gap`
- `line_count_new`
- `integer_shares_bought_total`

Si `cash_weight` reste élevé, les causes les plus probables sont:

- actions candidates trop chères pour le budget restant
- contraintes secteur ou poids par ligne trop strictes
- edge net insuffisant après coûts et buffer
- univers de candidats trop réduit

## Limites actuelles

- la covariance reste historique shrinkée, pas factorielle
- la résolution se fait sur la tranche d’entrée quotidienne, pas sur un rebalance global du portefeuille ouvert
- le prix d’exécution utilisé par le backtest reste l’`open` du jour d’exécution

Ces limites sont assumées: elles gardent le moteur cohérent avec le pipeline actuel et avec une exécution réaliste.
