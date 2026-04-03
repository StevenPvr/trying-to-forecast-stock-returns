# Launch Runbook

## Objectif

Ce runbook décrit le minimum nécessaire pour lancer le pipeline canonique `meta_model` en mode XTB-first.

## Pré-requis

Les trois fichiers suivants doivent exister:

- `core/data/reference/sp500_membership_history.csv`
- `core/data/reference/sp500_fundamentals_history.csv`
- `core/data/reference/xtb/xtb_instrument_specs.json`

La variable d’environnement suivante doit aussi être disponible pour le fetching canonique strict:

- `FRED_API_KEY`

Variables complémentaires:

- `TIINGO_API_KEY`
  - recommandée pour récupérer une partie des tickers que `yfinance` et `stooq` ne couvrent pas proprement
- `ID_WRDS`
- `PASSWORD_WRDS`
  - recommandées pour produire les fondamentaux PIT directement depuis WRDS

## Régénérer les références S&P 500

Les deux CSV de référence peuvent être régénérés via:

```bash
.venv/bin/python core/src/meta_model/data/data_reference/main.py
```

Cette commande:

- télécharge l’historique public des constituants S&P 500,
- compresse cet historique en intervalles `start_date` / `end_date`,
- intersecte ensuite cet historique avec l’univers `stock_cfd` explicite du snapshot XTB,
- choisit automatiquement la meilleure source de fondamentaux disponible,
- génère un fichier de fondamentaux pour les tickers XTB tradables uniquement.

Priorité réelle des sources fondamentales:

1. WRDS direct via `ID_WRDS` / `PASSWORD_WRDS`
2. `core/data/reference/wrds/compustat_fundq_extract.csv`
3. fallback open source bootstrap

## Mode WRDS recommandé

Le mode recommandé est désormais **WRDS direct** via `.env`.

Le fallback par extract reste supporté si tu veux geler un snapshot local. Pose alors un extract quarterly Compustat dans:

- `core/data/reference/wrds/compustat_fundq_extract.csv`

Colonnes minimales attendues:

- `tic`
- `datadate`
- `rdq`
- `prccq`
- `cshoq`
- `epspxq`
- `ceqq`
- `actq`
- `lctq`
- `niq`
- `saleq`

Le générateur calcule ensuite:

- `market_cap`
- `trailing_p_e`
- `price_to_book`
- `book_value`
- `current_ratio`
- `profit_margins`
- `return_on_equity`
- `revenue_growth`

Si cet extract WRDS est présent, il est utilisé quand le mode direct WRDS n’est pas disponible.

## Régénérer le snapshot XTB

Le snapshot broker peut être régénéré via:

```bash
.venv/bin/python core/src/meta_model/broker_xtb/main.py
```

Cette commande télécharge le document officiel XTB, extrait les stock CFDs US, exclut les lignes `CLOSE ONLY`, puis réécrit `xtb_instrument_specs.json`.

## Vérifier la readiness de lancement

```bash
.venv/bin/python core/src/meta_model/launch/main.py
```

Cette commande:

- vérifie la présence des fichiers de référence obligatoires,
- charge le snapshot XTB en mode strict,
- compte les instruments `stock_cfd` et `index_cfd`,
- vérifie que `lightgbm` est disponible,
- vérifie que `FRED_API_KEY` est exportée.

Si la commande termine avec un code non nul, le pipeline n’est pas prêt à être lancé.

État vérifié localement au 28 mars 2026:

- `is_ready = true`
- `stock_cfd_count = 1420`
- `index_cfd_count = 4`

## Pipeline canonique

Ordre d’exécution:

1. `.venv/bin/python core/src/meta_model/broker_xtb/main.py`
2. `.venv/bin/python core/src/meta_model/data/data_reference/main.py`
3. `.venv/bin/python core/src/meta_model/launch/main.py`
4. `.venv/bin/python core/src/meta_model/data/data_fetching/main.py`
5. `.venv/bin/python core/src/meta_model/data/data_cleaning/main.py`
6. `.venv/bin/python core/src/meta_model/features_engineering/main.py`
7. `.venv/bin/python core/src/meta_model/data/data_preprocessing/main.py`
8. `.venv/bin/python core/src/meta_model/feature_selection/main.py`
9. `.venv/bin/python core/src/meta_model/optimize_parameters/main.py`
10. `.venv/bin/python core/src/meta_model/evaluate/main.py`

## Notes opérationnelles importantes

- `data_fetching/main.py` loggue encore des erreurs `possibly delisted; no timezone found` sur certains anciens symboles.
- Ce bruit vient surtout de `yfinance`; le pipeline tente ensuite `stooq`, puis `Tiingo`.
- Les tickers restant trop incomplets après ces fallbacks sont éliminés par le filtre NaN du fetching.
- Dans l’état vérifié localement, `data_fetching` termine et produit un parquet non vide:
  - `core/data/data_fetching/dataset_2004_2025.parquet`
  - `1 634 851` lignes
  - `111` colonnes
- `data_cleaning` termine aussi et produit:
  - `core/data/data_cleaning/dataset_cleaned_2004_2025.parquet`
  - `1 634 851` lignes
  - `109` colonnes
- `features_engineering` peut basculer en séquentiel si les process workers ne sont pas autorisés dans l’environnement courant.

## Artefacts d’exécution attendus

- `core/data/broker_xtb/xtb_tradable_universe.parquet`
- `core/data/broker_xtb/xtb_specs_snapshot.json`
- `core/data/reference/sp500_membership_history.csv`
- `core/data/reference/sp500_fundamentals_history.csv`
- `core/data/reference/xtb/xtb_instrument_specs.json`
- `core/data/data_fetching/dataset_2004_2025.parquet`
- `core/data/data_cleaning/dataset_cleaned_2004_2025.parquet`
- `core/data/evaluate/model_leaderboard.json`
- `core/data/evaluate/manual_orders.csv`
- `core/data/evaluate/manual_watchlist.csv`
- `core/data/evaluate/execution_checklist.json`
