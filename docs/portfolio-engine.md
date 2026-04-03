# Moteur portefeuille

## Rôle

Le moteur portefeuille dans [core/src/meta_model/evaluate/backtest.py](/Users/steven/Programmation/prevision-sp500/core/src/meta_model/evaluate/backtest.py) transforme un signal cross-sectionnel en portefeuille quotidien broker-aware, puis prépare un pont manuel vers xStation.

## Étapes quotidiennes

À chaque date d’exécution:

1. appliquer le financement journalier sur les trades ouverts
2. clôturer les trades arrivés à maturité
3. construire les candidats long/short à partir des prédictions
4. filtrer les trades sous hurdle net-of-cost
5. allouer les nouvelles positions sous contraintes d’ADV, de marge et de neutralité
6. enregistrer rendement réalisé, benchmark, turnover et expositions

## Contraintes actives

- cap nominal par nom via `action_cap_fraction`
- cap brut global via `gross_cap_fraction`
- cap ADV via `adv_participation_limit`
- coût broker estimé par instrument/date
- marge requise et headroom
- hurdle d’ouverture `open_hurdle_bps`
- financement journalier dérivé des specs broker

## Modes de neutralité

- `dollar_neutral`
  - budgets long et short égaux
- `sector_neutral`
  - ne garde que les secteurs présents des deux côtés et répartit les budgets par secteur
- `sector_beta_neutral`
  - même logique sectorielle, puis budgets long/short ajustés avec les bêtas moyens des candidats

## Métriques produites

### Quotidiennes

- `benchmark_return`
- `turnover`
- `gross_exposure`
- `net_exposure`
- `realized_return`
- `capacity_binding_share`
- `margin_headroom`

### Résumé

- `annualized_return`
- `annualized_benchmark_return`
- `alpha_over_benchmark_net`
- `sharpe_ratio`
- `calmar_ratio`
- `max_drawdown`
- `turnover_annualized`
- `average_gross_exposure`
- `average_net_exposure`
- `realized_beta`
- `net_pnl`
- `capacity_binding_share`
- `margin_headroom`

## Limites actuelles

- benchmark encore égal-pondéré, pas SPY/FF5 complet
- neutralité déterministe, pas solveur d’optimisation quadratique
- les snapshots broker sont encore basés sur un provider par défaut si aucun snapshot officiel n’est présent
