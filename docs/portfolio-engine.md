# Moteur portefeuille

## Vue d’ensemble

Le moteur portefeuille canonique vit dans [`core/src/meta_model/evaluate/backtest.py`](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/evaluate/backtest.py) et s’appuie sur le solveur MIQP de [`core/src/meta_model/portfolio_optimization/solver.py`](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/portfolio_optimization/solver.py).

Le repo ne repose plus sur une heuristique `top-k` ni sur un solveur QP continu. La construction portefeuille est désormais **entière, sous contrainte, et orientée exécution réelle**.

## Ce que le moteur fait chaque jour

À chaque date d’exécution:

1. clôturer les positions arrivées à maturité
2. mettre à jour le cash disponible
3. convertir le score primaire en `primary_expected_return_5d`
4. raffiner ce score avec la confiance du méta-modèle
5. filtrer les titres sans edge net positif
6. résoudre le problème MIQP sur les candidats restants
7. ouvrir des positions **entières** seulement
8. enregistrer le diagnostic solveur, les allocations et le PnL quotidien

## Principes de conception

- **Long-only strict**
- **Pas de fractions d’actions**
- **Budget cash exact**
- **Exposition visée au plus près de 100%**, sans acheter des lignes destructrices juste pour remplir
- **Positions existantes figées** jusqu’à leur sortie naturelle
- **Causalité stricte**: les objets appris restent fit train-only

## Pourquoi ce choix

Avec un petit capital retail, un solveur continu n’est pas suffisant:

- il produit des poids irréalisables
- il laisse trop de cash après arrondi
- il sous-estime l’impact de l’indivisibilité des actions

Le MIQP traite directement le vrai problème de trading:

- combien d’actions entières acheter
- sur quels titres
- sous budget, coûts, caps et contraintes de risque

## Contraintes actives

- cap brut global via `gross_cap_fraction`
- cap de ligne via `max_position_weight`
- cap sectoriel via `max_sector_weight`
- cap de liquidité via `adv_cap_shares`
- minimum d’ordre XTB
- coût broker proxy
- buffer net-of-cost via `no_trade_buffer_bps`

## Diagnostics produits

### Quotidiens

- `solver_status`
- `candidate_count`
- `tradable_count`
- `cash_weight`
- `cash_amount_eur`
- `solve_time_seconds`
- `mip_gap`
- `line_count_new`
- `integer_shares_bought_total`
- `expected_portfolio_return`
- `expected_portfolio_volatility`

### Résumé backtest

- `total_return`
- `annualized_return`
- `alpha_over_benchmark_net`
- `sharpe_ratio`
- `calmar_ratio`
- `max_drawdown`
- `turnover_annualized`
- `average_gross_exposure`
- `transaction_cost_amount_total`

## Référence pédagogique

Pour le détail complet de la formulation, du solveur, du flux train-only et des artefacts, voir [`core/src/meta_model/portfolio_optimization/README.md`](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/portfolio_optimization/README.md).
