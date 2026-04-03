# Bonnes Pratiques -- prevision-sp500

> Document de reference pour le projet de prevision du S&P 500 / pipeline XTB CFD.
> Chaque regle est concrete et actionnable. Les exemples DO/DON'T s'appuient sur le code existant.

---

## Table des matieres

1. [Data Science generales](#1-data-science-generales)
2. [Machine Learning](#2-machine-learning)
3. [Finance quantitative](#3-finance-quantitative)
4. [Series temporelles](#4-series-temporelles)
5. [Regles specifiques au projet](#5-regles-specifiques-au-projet)

---

## 1. Data Science generales

### 1.1 Reproductibilite

**Seeds** -- Fixer toutes les sources d'aleatoire. Le projet utilise deja `random_seed: int = 7` dans `PipelineConfig` -- propager ce pattern a tout le pipeline.

```python
# DO -- seed explicite, passe via la config
np.random.seed(config.random_seed)
sample = data.sample(frac=0.05, random_state=config.random_seed)

# DON'T -- seed implicite ou absent
sample = data.sample(frac=0.05)  # non reproductible
```

**Versioning des donnees** -- Nommer les fichiers avec la plage temporelle (`sp500_prices_2004_2025.parquet`, deja en place). Ajouter un hash SHA-256 ou `manifest.json` pour detecter les corruptions. Ne jamais committer les donnees volumineuses dans git (utiliser DVC ou stockage externe).

**Environnements** -- Verrouiller les dependances avec `uv.lock` (fait). Travailler dans `.venv/bin/python`. Epingler Python dans `.python-version` (fait : `3.11`).

### 1.2 Gestion des donnees manquantes

```python
# DO -- strategie explicite, documentee, tracable
df["adj_close"] = df.groupby("ticker")["adj_close"].transform(
    lambda s: s.ffill(limit=5)  # forward-fill max 5 jours
)
LOGGER.info("NaN restants apres ffill(5) : %d", df["adj_close"].isna().sum())

# DON'T -- fillna silencieux sans limite
df["adj_close"].fillna(method="ffill", inplace=True)  # propage indefiniment
```

- [ ] Compter et logger les NaN avant et apres traitement.
- [ ] Distinguer NaN "donnees absentes" des NaN "marche ferme" (weekend, jour ferie).
- [ ] Documenter la strategie dans un docstring. Ne jamais interpoler entre tickers.

### 1.3 Documentation du pipeline

Chaque module (`data_fetching`, `data_cleaning`, `data_preprocessing`) doit contenir un docstring de module dans `__init__.py`, et chaque fonction publique doit documenter ses parametres, retour et effets de bord.

```python
# DO
def build_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Construit le DataFrame complet des prix S&P 500.

    Etapes : 1) Charge les constituants  2) Telecharge les prix  3) Applique le zero-fill.

    Args:
        config: Configuration du pipeline (dates, chemins, retries).
    Returns:
        DataFrame [date, ticker, adj_close], trie par (date, ticker).
    """
```

### 1.4 Separation des responsabilites

| Etape | Module | Entree | Sortie |
|---|---|---|---|
| Broker snapshot | `core/src/meta_model/broker_xtb/` | PDF XTB officiel | `core/data/reference/xtb/xtb_instrument_specs.json` |
| Reference data | `core/src/meta_model/data/data_reference/` | WRDS / bootstrap public | `core/data/reference/*.csv` |
| Fetching | `core/src/meta_model/data/data_fetching/` | Références PIT + providers marché/date-level | `core/data/data_fetching/dataset_2004_2025.parquet` |
| Cleaning | `core/src/meta_model/data/data_cleaning/` | `dataset_2004_2025.parquet` | `dataset_cleaned_2004_2025.parquet` |
| Features | `core/src/meta_model/features_engineering/` | `dataset_cleaned_2004_2025.parquet` | `dataset_features_2004_2025.parquet` |
| Preprocessing | `core/src/meta_model/data/data_preprocessing/` | `dataset_features_2004_2025.parquet` | `dataset_preprocessed_2009_2025.parquet` |
| Feature selection | `core/src/meta_model/feature_selection/` | `dataset_preprocessed_2009_2025.parquet` | `dataset_preprocessed_feature_selected.parquet` |
| Optimize | `core/src/meta_model/optimize_parameters/` | dataset filtré | `trial_ledger.parquet`, `overfitting_report.json` |
| Evaluate | `core/src/meta_model/evaluate/` | prédictions + labels + contraintes portefeuille | métriques, backtests, exports manuels |

Chaque etape lit un fichier et en produit un autre. Ne pas tout mettre dans une seule fonction.

### 1.5 Formats de stockage

| Format | Usage | Pourquoi |
|---|---|---|
| **Parquet** | Stockage principal | Typage fort, compression snappy, lecture par colonnes, 10-50x plus rapide que CSV |
| **CSV** | Debug / echantillons | Lisible humainement, utile pour un `sample_5pct` de verification |

- [ ] Ne jamais stocker de dates en string dans un parquet (utiliser `datetime64[ns]`).
- [ ] Compresser avec snappy (defaut pyarrow) ou zstd pour l'archivage long terme.

### 1.6 Logging et monitoring

Le projet utilise deja `logging` (`LOGGER = logging.getLogger(__name__)`) -- bien.

```python
# DO -- niveaux de log coherents
LOGGER.info("Fetching %d tickers", len(symbols))             # progression
LOGGER.warning("yfinance batch failed (%s/%s)", attempts, n)  # recuperable
LOGGER.error("No price data retrieved")                       # echec

# DON'T
print(f"Fetching {len(symbols)} tickers")  # invisible en prod, non filtrable
```

- [ ] `logging` toujours, `print` jamais en production.
- [ ] Logger le nombre de lignes/tickers a chaque etape.
- [ ] Logger les temps d'execution (`time.perf_counter()`).

### 1.7 Typage strict

**Tout le code doit etre type.** Chaque variable, parametre et retour de fonction porte une annotation de type.

```python
# DO -- typage complet
def _parse_date(value: str | dt.date | dt.datetime) -> dt.date:
    ...

def _fetch_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
    config: PipelineConfig,
) -> dict[str, pd.Series]:
    aliases: dict[str, str] = _load_aliases(config.ticker_aliases_csv)
    results: dict[str, pd.Series] = {}
    ...

LOGGER: logging.Logger = logging.getLogger(__name__)

# DON'T -- pas de type
def _parse_date(value):
    ...

results = {}
```

- [ ] `from __future__ import annotations` en premiere ligne de chaque fichier (union `X | Y` compatible Python 3.10+).
- [ ] Installer les stubs pour les librairies tierces (`pandas-stubs`, `types-requests`, `types-pytz`).
- [ ] Pour les librairies sans stubs (yfinance, pandas_datareader), creer des stubs dans `typings/` (convention Pylance).
- [ ] Utiliser `X | None` au lieu de `Optional[X]`.
- [ ] Typer les variables locales importantes (DataFrames, dictionnaires, listes).
- [ ] **Jamais** de `# type: ignore` -- utiliser `cast()` ou `Any` quand le type checker ne peut pas inferer.
- [ ] **Zero erreur Pylance** dans tous les fichiers.

```python
from typing import Any, cast

# DO -- cast explicite quand le type checker ne peut pas inferer
last_valid: pd.Timestamp | None = cast(pd.Timestamp | None, series.last_valid_index())

# DO -- Any pour les retours de librairies non typees si cast impossible
raw_result: Any = external_lib.some_call()
data: pd.DataFrame = raw_result

# DON'T
data = pdr.get_data_stooq(...)  # type: ignore[union-attr]
```

### 1.8 Limites de taille

- [ ] **50 lignes max par fonction.** Si une fonction depasse 50 lignes, la decouper en sous-fonctions.
- [ ] **500 lignes max par fichier.** Si un fichier depasse 500 lignes, le scinder en modules.

---

## 2. Machine Learning

### 2.1 Split temporel strict

**C'est la regle la plus importante du projet.**

```python
# DO -- split par date, sans aucune fuite du futur
train = df[df["date"] < "2020-01-01"]
val   = df[(df["date"] >= "2020-01-01") & (df["date"] < "2022-01-01")]
test  = df[df["date"] >= "2022-01-01"]

# DON'T -- JAMAIS de split aleatoire sur des series temporelles
train, test = train_test_split(df, test_size=0.2)  # INTERDIT : fuite temporelle
```

- [ ] Le set de test ne doit jamais etre touche avant l'evaluation finale.
- [ ] Les dates de coupure doivent etre dans la config, pas en dur.

### 2.2 Walk-forward validation

```python
# Expanding window -- le train grandit, le test avance
splits = [
    ("2004-01-01", "2015-12-31", "2016-01-01", "2016-12-31"),
    ("2004-01-01", "2016-12-31", "2017-01-01", "2017-12-31"),
    ("2004-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
]
```

- Utiliser `TimeSeriesSplit` comme point de depart, mais verifier les frontieres par **date** (pas par index).
- Laisser un **gap** d'au moins 1-5 jours entre train et validation pour eviter la fuite via lags/rolling.

### 2.3 Prevention du data leakage

| Source de fuite | Exemple | Prevention |
|---|---|---|
| Feature calculee sur tout le dataset | `StandardScaler().fit(df_complet)` | Fit uniquement sur train |
| Information future dans les features | rolling mean sans `min_periods` | Toujours `min_periods=window` |
| Target qui fuit dans les features | `return_t+1` accessible a `t` | Verifier l'alignement temporel |
| Donnees corporate publiees apres coup | Revision de BPA | Donnees point-in-time |

```python
# DO -- pipeline sklearn avec fit sur train uniquement
pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
pipe.fit(X_train, y_train)       # scaler apprend mu/sigma du train
preds = pipe.predict(X_test)     # applique les params du train au test

# DON'T -- normalisation sur tout le dataset
df_scaled = StandardScaler().fit_transform(df_all)  # FUITE
```

### 2.4 Metriques adaptees a la finance

Les metriques classiques (MSE, RMSE) ne suffisent pas. Ajouter :

| Metrique | Mesure | Cible indicative |
|---|---|---|
| **Sharpe Ratio** | Rendement ajuste du risque (annualise) | > 1.0 interessant, > 2.0 excellent |
| **Max Drawdown** | Pire perte depuis un pic | Le plus faible possible |
| **Hit Ratio** | % de predictions de direction correctes | > 55% pour etre exploitable |
| **Profit Factor** | Gains bruts / Pertes brutes | > 1.5 |
| **Calmar Ratio** | Rendement annuel / Max Drawdown | > 1.0 |

```python
def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf / periods
    return np.sqrt(periods) * excess.mean() / excess.std()

def max_drawdown(cumulative_returns: pd.Series) -> float:
    peak = cumulative_returns.cummax()
    return ((cumulative_returns - peak) / peak).min()
```

- [ ] Metriques calculees sur le set de test out-of-sample uniquement.
- [ ] Comparer a un baseline naif (buy & hold, prevision = rendement moyen).

### 2.5 Calibration et overfitting

```python
# DO -- commencer simple
model = Ridge(alpha=1.0)  # baseline lineaire regulariee

# DON'T -- complexite prematuree
model = XGBRegressor(n_estimators=5000, max_depth=12)  # overfitting garanti
```

- [ ] Regularisation systematique (L1/L2, dropout, early stopping).
- [ ] Limiter les features a ~10-20x le nombre d'observations independantes.
- [ ] Sharpe in-sample > 3.0 et out-of-sample < 0.5 = overfitting.
- [ ] Documenter le nombre de combinaisons testees (Bonferroni / deflated Sharpe).

### 2.6 Feature importance et interpretabilite

- [ ] Calculer l'importance (permutation importance, SHAP) apres chaque entrainement.
- [ ] Eliminer les features negligeables -- elles ajoutent du bruit.
- [ ] Verifier que les features importantes ont un sens economique.

---

## 3. Finance quantitative

### 3.1 Survivorship bias

Le pipeline gere deja ce biais via `_load_constituents_from_wikipedia` (historique ajouts/retraits). Points d'attention :

- [ ] Les changements Wikipedia ne remontent que jusqu'a une certaine date (deja logue). Documenter cette limitation.
- [ ] Le zero-fill (`_apply_delisting_zero`) preserve les tickers delistes. Exclure les `0.0` du calcul des rendements.

```python
# DO -- exclure les prix a 0 du calcul des rendements
returns = df[df["adj_close"] > 0].groupby("ticker")["adj_close"].pct_change()

# DON'T -- calculer des rendements incluant la transition vers 0.0
returns = df.groupby("ticker")["adj_close"].pct_change()  # -100% puis NaN/inf
```

### 3.2 Look-ahead bias

Utiliser une information non disponible au moment de la decision. Distinct du data leakage (probleme de pipeline ML) : ici c'est un probleme de temporalite des donnees.

- Composition de l'indice connue ex-post (gere par le pipeline).
- Donnees fondamentales revisees (earnings restated).
- Prix ajustes retroactivement pour splits/dividendes : yfinance fournit un `Adj Close` recalcule retroactivement. Pour un backtest strict, stocker aussi le `Close` brut.

### 3.3 Rendements vs prix

Travailler sur les rendements, **jamais** les prix bruts (non-stationnaires) :

```python
# Arithmetique -- interpretation directe en %, portefeuille multi-actifs
df["return_simple"] = df.groupby("ticker")["adj_close"].pct_change()

# Logarithmique -- additivite temporelle, modelisation statistique
df["return_log"] = np.log(df["adj_close"] / df.groupby("ticker")["adj_close"].shift(1))
```

- [ ] Log-returns pour la modelisation, arithmetique pour le reporting.

### 3.4 Couts de transaction et slippage

Un backtest sans couts est un backtest faux.

```python
COST_BPS = 10  # 10 points de base aller-retour (conservateur)

def net_return(gross_return: float, turnover: float) -> float:
    return gross_return - turnover * COST_BPS / 10_000
```

- [ ] Documenter : couts (bps), slippage, delai d'execution (cloture ? ouverture J+1 ?).

### 3.5 Regime changes et non-stationnarite

```python
# DO -- walk-forward qui re-entraine regulierement
for train_end, test_start, test_end in walk_forward_splits:
    model.fit(X_train)  # re-entrainement a chaque fenetre
    preds = model.predict(X_test)

# DON'T -- entrainer une fois et predire sur 10 ans
model.fit(X_2004_2014)
preds = model.predict(X_2015_2025)  # les relations ont change
```

- [ ] Tester la performance par regime (bull/bear/sideways).
- [ ] Monitorer la stabilite des coefficients dans le temps.

### 3.6 Backtesting realiste -- checklist

- [ ] Couts de transaction inclus.
- [ ] Slippage inclus (ou modele simplifie).
- [ ] Aucun look-ahead bias ni survivorship bias.
- [ ] Pas de fill au prix de cloture si la decision est prise a la cloture.
- [ ] Position sizing realiste (pas de levier infini).
- [ ] Comparer au benchmark (buy & hold S&P 500).
- [ ] Reporter sur plusieurs periodes (pas cherry-picker la meilleure).

### 3.7 Point-in-time data

Les donnees doivent refleter ce qui etait **reellement disponible** au moment de la decision.

```python
# DO -- horodater avec la date de publication reelle
gdp_data = pd.DataFrame({
    "period": ["2023-Q4"], "value": [5.2],
    "release_date": ["2024-03-28"],  # PIB Q4 publie fin mars
})
# DON'T -- utiliser la date de la periode comme date de disponibilite (= look-ahead)
```

---

## 4. Series temporelles

### 4.1 Stationnarite

Les prix ne sont pas stationnaires ; les rendements le sont (souvent). Tester avant de modeliser.

```python
from statsmodels.tsa.stattools import adfuller, kpss

# ADF : H0 = non-stationnaire. p < 0.05 => stationnaire.
adf_pval = adfuller(series.dropna())[1]

# KPSS : H0 = stationnaire. p < 0.05 => non-stationnaire.
kpss_pval = kpss(series.dropna(), regression="c")[1]
```

| ADF (p < 0.05) | KPSS (p > 0.05) | Conclusion |
|---|---|---|
| Oui | Oui | Stationnaire |
| Non | Non | Non-stationnaire -- differencier |
| Oui | Non | Trend-stationary |
| Non | Oui | Racine unitaire avec break structurel |

### 4.2 Autocorrelation

- L'autocorrelation dans les rendements bruts est faible (marche efficient), mais presente dans la **volatilite** (GARCH).
- L'autocorrelation dans les residus du modele = mauvaise specification.
- Les erreurs standard classiques sont biaisees avec autocorrelation (utiliser Newey-West).

### 4.3 Feature engineering temporel

```python
# Lags
for lag in [1, 2, 5, 10, 21]:
    df[f"return_lag_{lag}"] = df.groupby("ticker")["return_log"].shift(lag)

# Rolling statistics (1 semaine, 1 mois, 1 trimestre)
for w in [5, 21, 63]:
    df[f"vol_{w}d"] = df.groupby("ticker")["return_log"].transform(
        lambda s: s.rolling(w, min_periods=w).std()
    )
    df[f"mom_{w}d"] = df.groupby("ticker")["return_log"].transform(
        lambda s: s.rolling(w, min_periods=w).sum()
    )

# Calendrier
df["day_of_week"] = df["date"].dt.dayofweek     # 0=lundi
df["month"] = df["date"].dt.month                # effet janvier
df["quarter_end"] = df["date"].dt.is_quarter_end # window dressing
```

- [ ] Toujours `min_periods=window` dans les rolling.
- [ ] Toujours `shift(1)` le target ou les features derivees du target.
- [ ] Features de calendrier = jours de **trading**, pas jours calendaires.

### 4.4 Jours feries et weekends

Le projet utilise `pd.bdate_range` -- correct mais insuffisant (ignore les jours feries US).

```python
# DO -- calendrier boursier precis
import exchange_calendars as xcals
nyse = xcals.get_calendar("XNYS")
sessions = nyse.sessions_in_range(start, end)

# Alternative sans dependance supplementaire :
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# DON'T
index = pd.bdate_range(start, end)  # inclut les jours feries federaux
```

### 4.5 Frequence et alignement temporel

```python
# Merger des donnees de frequences differentes (prix journalier + macro mensuel)
macro["date"] = macro["date"] + pd.offsets.MonthEnd(0)
df = df.merge(macro, on="date", how="left")
df["macro_feat"] = df.groupby("ticker")["macro_feat"].ffill()

# DON'T -- merger sans aligner => perd 95% des lignes
df = prices.merge(macro, on="date")
```

### 4.6 Saisonnalite

- La saisonnalite dans les rendements est faible et instable (effet janvier, effet lundi). Ne pas la surestimer.
- Utile surtout pour la **volatilite** (publication de resultats, expirations d'options).
- Tester la significativite statistique avant d'integrer une composante saisonniere.

---

## 5. Regles specifiques au projet

### 5.0 Etat courant du repo

- Le seul pipeline supporté est le pipeline `meta_model` XTB-first.
- Les couches legacy `secondary_model`, `feature_corr_pca` et `feature_selection_lag` ne font plus partie du canonique.
- Les artefacts runtime vivent dans `core/data/` et ce dossier est gitignoré.
- Le bootstrap canonique avant un vrai run est:
  1. `core/src/meta_model/broker_xtb/main.py`
  2. `core/src/meta_model/data/data_reference/main.py`
  3. `core/src/meta_model/launch/main.py`
  4. puis la chaîne `data_fetching -> data_cleaning -> features_engineering -> data_preprocessing -> feature_selection -> optimize_parameters -> evaluate`
- `data_fetching` est broker-aware:
  - univers PIT S&P 500 intersecté avec les `stock_cfd` XTB,
  - fondamentaux PIT depuis WRDS direct si `ID_WRDS` / `PASSWORD_WRDS` existent,
  - fallback prix `yfinance -> stooq -> Tiingo`.
- Les warnings `possibly delisted; no timezone found` venant de `yfinance` sont attendus sur certains anciens symboles; le pipeline continue avec les fallbacks puis élimine les tickers trop incomplets.
- `launch/main.py` est le juge officiel de readiness minimale, pas une promesse que tous les étages aval ont été rerun dans la session en cours.

### 5.1 Convention de nommage des fichiers

```
core/data/{etape}/{description}_{debut}_{fin}.{format}

Exemples :
  core/data/data_fetching/sp500_prices_2004_2025.parquet
  core/data/data_cleaning/sp500_prices_clean_2004_2025.parquet
  core/data/data_preprocessing/sp500_features_2004_2025.parquet
```

- [ ] Minuscules, underscores. Plage de dates dans le nom. Parquet principal, CSV pour debug.

### 5.2 Structure des DataFrames

Le DataFrame principal (sortie de `data_fetching`) suit un format long (tidy) :

| Colonne | Type | Description |
|---|---|---|
| `date` | `datetime64[ns]` | Date du jour de trading |
| `ticker` | `str` | Symbole du titre (ex: `AAPL`, `MSFT`) |
| `adj_close` | `float64` | Prix ajuste de cloture |

Colonnes ajoutees par les etapes suivantes : `adj_close_clean`, `is_missing`, `is_delisted` (cleaning), `return_log`, `volatility_21d`, `momentum_63d` (preprocessing).

```python
EXPECTED_DTYPES = {"date": "datetime64[ns]", "ticker": "object", "adj_close": "float64"}

def validate_schema(df: pd.DataFrame) -> None:
    for col, dtype in EXPECTED_DTYPES.items():
        assert col in df.columns, f"Colonne manquante : {col}"
        assert str(df[col].dtype) == dtype, f"{col}: {df[col].dtype} != {dtype}"
```

- [ ] Ne jamais utiliser l'index pour stocker des donnees (toujours `reset_index`).

### 5.3 Gestion du delisting

Le pipeline zero-fill (`_apply_delisting_zero`) met le prix a 0.0 apres la derniere cotation. Dans les etapes suivantes :

- [ ] Filtrer `adj_close > 0` avant de calculer les rendements.
- [ ] Le rendement du dernier jour = -100% (correct pour le survivorship bias).
- [ ] Documenter dans chaque notebook/script si les delistes sont inclus ou exclus.

### 5.4 Activation des pipelines via `main.py`

**Chaque module/etape du pipeline doit avoir un fichier `main.py`** qui sert de point d'entree unique. Ce fichier doit etre directement executable en cliquant dessus dans l'IDE (pas besoin de terminal).

```python
# core/src/meta_model/data/data_fetching/main.py

import sys
from pathlib import Path

# Remonter jusqu'a la racine du projet pour que tous les imports fonctionnent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_fetching.sp500_pipeline import PipelineConfig, run_pipeline

def main() -> None:
    config = PipelineConfig()
    run_pipeline(config)

if __name__ == "__main__":
    main()
```

- [ ] **Tout** pipeline s'active via son `main.py`, jamais en important directement un module.
- [ ] Le `sys.path` est configure en haut du fichier pour garantir l'execution directe.
- [ ] Le `if __name__ == "__main__"` est obligatoire.

### 5.4.b Modularite stricte du pipeline

Le fichier `main.py` doit rester un **orchestrateur**: chargement, enchainement des etapes, sauvegarde.
Toute logique metier (cleaning, outliers, feature engineering, validation schema) doit vivre dans des modules dedies.

```python
# DO -- main orchestration only
from core.src.data.data_cleaning.outlier_pipeline import apply_outlier_flags

def main() -> None:
    df = load_raw_dataset(INPUT_PATH)
    cleaned = apply_outlier_flags(df)
    save_cleaned(cleaned, OUTPUT_PATH, SAMPLE_PATH)

# DON'T -- logique metier lourde dans main.py
def main() -> None:
    ...  # 300+ lignes de logique outliers, rules, rolling stats, etc.
```

- [ ] `main.py` < 150 lignes, sans regles metier complexes.
- [ ] Un module metier par responsabilite (ex: `outlier_pipeline.py`, `nan_policy.py`).
- [ ] Les tests suivent le miroir des modules (`test_outlier_pipeline.py`, etc.).

### 5.5 Fichiers `paths.py` et `constants.py`

Ces fichiers se placent **a la racine du module data actif**. Pour le pipeline canonique actuel: `core/src/meta_model/data/paths.py` et `core/src/meta_model/data/constants.py`.

**`paths.py`** -- centralise tous les chemins du pipeline.

```python
# core/src/meta_model/data/paths.py

from __future__ import annotations
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
CORE_DIR: Path = PROJECT_ROOT / "core"
DATA_DIR: Path = CORE_DIR / "data"

DATA_FETCHING_DIR: Path = DATA_DIR / "data_fetching"
OUTPUT_PARQUET: Path = DATA_FETCHING_DIR / "sp500_prices_2004_2025.parquet"
OUTPUT_SAMPLE_CSV: Path = DATA_FETCHING_DIR / "sp500_prices_2004_2025_sample_5pct.csv"

DATA_CLEANING_DIR: Path = DATA_DIR / "data_cleaning"
DATA_PREPROCESSING_DIR: Path = DATA_DIR / "data_preprocessing"
```

**`constants.py`** -- centralise les constantes metier reutilisables.

```python
# core/src/meta_model/data/constants.py

from __future__ import annotations

WIKIPEDIA_SP500_URL: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_START_DATE: str = "2004-01-01"
DEFAULT_END_DATE: str = "2025-12-31"
SAMPLE_FRAC: float = 0.05
RANDOM_SEED: int = 7
CHUNK_SIZE: int = 50
MAX_RETRIES: int = 3
RETRY_SLEEP: float = 2.0
```

- [ ] **Jamais** de chemins en dur dans les fonctions -- tout passe par `paths.py`.
- [ ] **Jamais** de constantes magiques dans le code -- tout passe par `constants.py`.
- [ ] Les autres fichiers du module importent depuis `paths` et `constants`.

```python
# DO
from core.src.data.paths import OUTPUT_PARQUET
from core.src.data.constants import MAX_RETRIES

# DON'T
output_path = Path("core/data/data_fetching/sp500_prices_2004_2025.parquet")  # chemin en dur
max_retries = 3  # constante magique
```

### 5.6 Tests unitaires -- architecture miroir

Les tests dans `tests/` **reproduisent exactement** l'arborescence de `core/src/` :

```
core/src/data/data_fetching/sp500_pipeline.py
tests/data/data_fetching/test_sp500_pipeline.py

core/src/data/data_cleaning/cleaning.py
tests/data/data_cleaning/test_cleaning.py

core/src/data/data_preprocessing/preprocessing.py
tests/data/data_preprocessing/test_preprocessing.py
```

- [ ] Un fichier source = un fichier de test correspondant.
- [ ] Le fichier de test est nomme `test_{nom_du_fichier_source}.py`.
- [ ] Ecrire les tests **immediatement** apres l'ecriture d'une fonction ou d'un fichier, pas apres coup.

### 5.7 Tests directement executables

Chaque fichier de test doit etre executable de **deux facons** : individuellement (clic dans l'IDE) et via pytest.

```python
# tests/data/data_fetching/test_sp500_pipeline.py

import sys
from pathlib import Path

# Remonter jusqu'a la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd

from core.src.data.data_fetching.sp500_pipeline import (
    _apply_delisting_zero,
    _extract_tickers,
    _parse_date,
)


def test_parse_date_string():
    assert _parse_date("2020-01-15").isoformat() == "2020-01-15"


def test_extract_tickers_comma_separated():
    assert _extract_tickers("AAPL, MSFT, GOOG") == ["AAPL", "MSFT", "GOOG"]


def test_delisting_zero_fill():
    index = pd.bdate_range("2020-01-01", "2020-01-10")
    series = pd.Series([100, 101, 102, np.nan, np.nan], index=index[:5])
    result = _apply_delisting_zero(series, index)
    assert result.iloc[-1] == 0.0


def test_empty_price_map_raises():
    # Exemple avec config qui pointe vers un CSV vide
    with pytest.raises(RuntimeError, match="No price data"):
        ...


# --- Execution directe (clic dans l'IDE) ---
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] Framework : `pytest`. Lancer tout avec `uv run pytest tests/`.
- [ ] Lancer un seul fichier : clic direct ou `uv run python tests/data/data_fetching/test_sp500_pipeline.py`.
- [ ] Le bloc `if __name__ == "__main__": pytest.main([__file__, "-v"])` est **obligatoire** dans chaque fichier de test.
- [ ] Le `sys.path` en haut du fichier est **obligatoire** pour l'execution directe.
- [ ] Fixtures legeres dans `conftest.py` (10-20 tickers, 30 jours).
- [ ] **Jamais** d'appels API reels dans les tests -- mocker yfinance/stooq.

---

## Checklist avant chaque commit

```
[ ] Les tests passent (uv run pytest tests/)
[ ] Pas de print() en dehors des notebooks
[ ] Les seeds sont fixes et passent par la config
[ ] Les nouvelles features utilisent min_periods et shift correctement
[ ] Le schema du DataFrame de sortie est documente et valide
[ ] Le logging trace le nombre de lignes a chaque etape
[ ] Les fichiers de donnees ne sont pas commites dans git
[ ] Si le changement touche le bootstrap, `broker_xtb/main.py`, `data_reference/main.py` ou `launch/main.py` ont ete verifies
[ ] Chaque nouveau fichier source a son fichier de test miroir
[ ] Les main.py et fichiers de test sont directement executables (sys.path + __main__)
[ ] Les chemins passent par paths.py, les constantes par constants.py
```
