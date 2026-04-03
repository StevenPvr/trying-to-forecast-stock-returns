# Features a rajouter

## Contexte

Apres audit de `core/src/meta_model/data/data_fetching/` et `core/src/meta_model/features_engineering/`, le pipeline contient deja :

- les prix et log-returns `stock_*`
- les fondamentaux `company_*`
- les signaux calendrier `calendar_*`
- les series macro `macro_*`
- les series cross-asset `cross_asset_*`
- les signaux sentiment `sentiment_*`
- les indicateurs TA standards `ta_*`
- les features quant maison `quant_*`
- les features `deep_*`
- les transformations cross-sectionnelles globales et univers `quant_cs_*`, `quant_universe_*`

Les trous les plus prometteurs ne sont donc pas des variantes de RSI ou de momentum standard. Les vraies poches d'information manquantes sont :

1. l'information de cout broker/XTB en entree modele
2. le contexte sectoriel relatif
3. la structure d'ouverture par rapport a la veille
4. les transitions de regime de volatilite
5. les evenements entreprise de type earnings

## Priorite 1 - Couts broker/XTB

Ces colonnes ne sont pas injectees dans les features aujourd'hui. Elles servent surtout a filtrer l'univers et a calculer les labels nets.

| Feature | Formule intuitive | Pourquoi c'est utile | Ou l'ajouter |
|---|---|---|---|
| `xtb_expected_intraday_cost_rate` | cout total estime pour un trade open->close | aligne le modele avec l'objectif net, evite les signaux trop nerveux sur actifs chers | `data_fetching` ou `feature_engineering/post_processing.py` |
| `xtb_expected_overnight_cost_rate` | cout total estime pour un trade close->next open | utile si le modele capte aussi une composante overnight | `data_fetching` ou `feature_engineering/post_processing.py` |
| `xtb_spread_bps` | spread broker courant du ticker | niveau brut du handicap d'execution | `data_fetching` |
| `xtb_slippage_bps` | slippage broker estime | penalise les actifs difficilement tradables | `data_fetching` |
| `xtb_long_swap_bps_daily` | swap journalier long | contexte de portage | `data_fetching` |
| `xtb_short_swap_bps_daily` | swap journalier short | contexte de portage short | `data_fetching` |
| `xtb_swap_asymmetry` | `xtb_short_swap_bps_daily - xtb_long_swap_bps_daily` | mesure l'asymetrie de portage long/short | `feature_engineering/post_processing.py` |
| `xtb_spread_to_realized_vol_21d` | `xtb_spread_bps / quant_realized_vol_21d` | cout normalise par la volatilite exploitable | `feature_engineering/post_processing.py` |
| `xtb_spread_to_gap_abs` | `xtb_spread_bps / abs(quant_gap_return)` | dit si le gap est "tradable" apres friction | `feature_engineering/post_processing.py` |

## Priorite 2 - Contexte sectoriel relatif

Le pipeline a `company_sector`, mais il n'expose pas encore de vraies features sector-relative d'entree sur les signaux court terme.

| Feature | Formule intuitive | Pourquoi c'est utile | Ou l'ajouter |
|---|---|---|---|
| `sector_relative_gap_return` | `quant_gap_return - mean_sector_gap_return_same_day` | distingue un gap idiosyncratique d'un gap de secteur | `feature_engineering/post_processing.py` |
| `sector_relative_intraday_return` | `quant_intraday_return - mean_sector_intraday_return_same_day` | mesure la force relative intra-secteur | `feature_engineering/post_processing.py` |
| `sector_relative_rsi` | `ta_momentum_rsi - mean_sector_rsi_same_day` | plus utile qu'un RSI absolu quand tout le secteur est deja surachete | `feature_engineering/post_processing.py` |
| `sector_rsi_rank` | rang percentile du RSI dans le secteur a la date | version robuste du signal sectoriel | `feature_engineering/post_processing.py` |
| `sector_gap_rank` | rang percentile du gap dans le secteur a la date | capte les outliers au sein du secteur | `feature_engineering/post_processing.py` |

## Priorite 3 - Structure d'ouverture

Le pipeline a deja `quant_gap_return`, `deep_event_gap_fill_flag`, `prev_high`, `prev_low` dans le contexte deep, mais pas les features directes les plus exploitables sur la structure de l'ouverture.

| Feature | Formule intuitive | Pourquoi c'est utile | Ou l'ajouter |
|---|---|---|---|
| `open_above_prev_high_flag` | `stock_open_price > prev_high` | regime d'ouverture en breakout haussier | `features_engineering/deep/event_features.py` ou `quant_features.py` |
| `open_below_prev_low_flag` | `stock_open_price < prev_low` | regime d'ouverture en breakdown baissier | `features_engineering/deep/event_features.py` ou `quant_features.py` |
| `open_in_prev_range_flag` | `prev_low <= stock_open_price <= prev_high` | distingue les gaps "hors range" des ouvertures normales | `features_engineering/deep/event_features.py` ou `quant_features.py` |
| `open_distance_to_prev_close_over_atr_21d` | `(stock_open_price - prev_close) / ATR_21d` | version normalisee de l'ecart d'ouverture | `quant_features.py` |
| `open_distance_to_prev_high_over_atr_21d` | `(stock_open_price - prev_high) / ATR_21d` | indique la profondeur du breakout d'ouverture | `quant_features.py` |
| `open_distance_to_prev_low_over_atr_21d` | `(stock_open_price - prev_low) / ATR_21d` | indique la profondeur du breakdown d'ouverture | `quant_features.py` |

## Priorite 4 - Transitions de regime de volatilite

Le pipeline a des niveaux de volatilite, mais pas assez de ratios de transition. Or le changement de regime est souvent plus informatif que le niveau brut.

| Feature | Formule intuitive | Pourquoi c'est utile | Ou l'ajouter |
|---|---|---|---|
| `quant_realized_vol_ratio_5d_21d` | `realized_vol_5d / realized_vol_21d` | acceleration recente de vol | `quant_features.py` |
| `quant_realized_vol_ratio_21d_63d` | `realized_vol_21d / realized_vol_63d` | regime de vol moyen vs lent | `quant_features.py` |
| `quant_gap_vol_ratio_21d_63d` | `std(gap_return,21d) / std(gap_return,63d)` | detecte un changement recent du regime de gap | `quant_features.py` |
| `quant_intraday_vol_ratio_21d_63d` | `std(intraday_return,21d) / std(intraday_return,63d)` | detecte une acceleration des mouvements intraday | `quant_features.py` |
| `quant_true_range_zscore_21d` | z-score de `true_range_pct` sur 21 jours | capte les seances d'expansion anormale | `quant_features.py` |

## Priorite 5 - Evenements earnings

Ce sont probablement les meilleures features externes si on accepte une nouvelle source de donnees point-in-time.

| Feature | Formule intuitive | Pourquoi c'est utile | Ou l'ajouter |
|---|---|---|---|
| `days_to_next_earnings` | nombre de jours de bourse avant la prochaine publication | tres utile pour l'intraday et l'overnight | nouveau module `data_fetching/earnings_pipeline.py` |
| `days_since_last_earnings` | nombre de jours depuis la derniere publication | capte la digestion post-annonce | nouveau module `data_fetching/earnings_pipeline.py` |
| `is_earnings_week` | flag si earnings a moins de 5 jours | regime evenementiel | nouveau module `data_fetching/earnings_pipeline.py` |
| `earnings_proximity_bucket` | bucket 0-1j / 2-5j / 6-10j / loin | version robuste de la proximite earnings | nouveau module `data_fetching/earnings_pipeline.py` |

## Recommandation d'implementation

Si on veut maximiser les chances de survivre a la `feature_selection`, l'ordre recommande est :

1. ajouter les features broker/XTB
2. ajouter les features sector-relative
3. ajouter la structure d'ouverture
4. ajouter les ratios de regime de volatilite
5. seulement ensuite ouvrir un chantier earnings

## Premier lot recommande

Si on veut un premier batch compact et tres defensable, commencer par ces 10 colonnes :

- `xtb_expected_intraday_cost_rate`
- `xtb_expected_overnight_cost_rate`
- `xtb_spread_to_realized_vol_21d`
- `xtb_spread_to_gap_abs`
- `xtb_swap_asymmetry`
- `sector_relative_gap_return`
- `sector_relative_intraday_return`
- `sector_relative_rsi`
- `open_above_prev_high_flag`
- `open_distance_to_prev_close_over_atr_21d`

Ce sont les 10 ajouts avec le meilleur ratio :

- information nouvelle
- lien direct avec le label net
- faible redondance avec le pipeline actuel
