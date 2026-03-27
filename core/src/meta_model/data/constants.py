from __future__ import annotations

# --- S&P 500 Data Fetching ---

WIKIPEDIA_SP500_URL: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

DEFAULT_START_DATE: str = "2004-01-01"
DEFAULT_END_DATE: str = "2025-12-31"
SAMPLE_FRAC: float = 0.05
RANDOM_SEED: int = 7
CHUNK_SIZE: int = 50
MAX_RETRIES: int = 3
RETRY_SLEEP: float = 2.0

# --- Fundamentals ---

FUNDAMENTAL_FIELDS: tuple[str, ...] = (
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "beta",
    "profitMargins",
    "returnOnEquity",
    "enterpriseValue",
    "revenueGrowth",
    "currentRatio",
    "bookValue",
    "trailingEps",
    "forwardEps",
)
FUNDAMENTALS_SLEEP: float = 0.2

# --- Macro (FRED) ---

FRED_DAILY_SERIES: tuple[str, ...] = (
    # US Treasury Yields (courbe des taux complete)
    "DGS1",            # Treasury 1Y
    "DGS2",            # Treasury 2Y
    "DGS5",            # Treasury 5Y
    "DGS10",           # Treasury 10Y
    "DGS30",           # Treasury 30Y
    "DTB3",            # T-Bill 3M
    # Spreads de taux
    "T10Y2Y",          # Spread 10Y-2Y
    "T10Y3M",          # Spread 10Y-3M
    # Taux directeur
    "DFF",             # Fed Funds Effective Rate
    # Rendements reels & anticipations d'inflation
    "DFII10",          # TIPS 10Y (rendement reel)
    "T10YIE",          # Breakeven inflation 10Y
    "T5YIE",           # Breakeven inflation 5Y
    "T5YIFR",          # Forward inflation 5Y-5Y
    # Spreads de credit
    "BAMLH0A0HYM2",    # ICE BofA High Yield OAS
    "BAMLC0A0CM",      # ICE BofA Investment Grade OAS
    # Volatilite
    "VIXCLS",          # VIX
    # Matieres premieres
    "DCOILWTICO",      # WTI Crude Oil
    "DCOILBRENTEU",    # Brent Crude Oil
    "DHHNGSP",         # Gaz naturel Henry Hub
    # Dollar & change
    "DTWEXBGS",        # Trade Weighted Dollar Index (broad)
    "DEXUSEU",         # USD/EUR
    "DEXJPUS",         # JPY/USD
    "DEXCHUS",         # CNY/USD
    "DEXUSUK",         # USD/GBP
    # Incertitude
    "USEPUINDXD",      # Economic Policy Uncertainty Index (daily)
)
FRED_WEEKLY_SERIES: tuple[str, ...] = (
    # Politique monetaire
    "WALCL",           # Bilan Fed (Total Assets)
    # Taux hypothecaire
    "MORTGAGE30US",    # Taux hypothecaire 30 ans
    # Stress & conditions financieres
    "STLFSI4",         # St. Louis Fed Financial Stress Index
    "NFCI",            # Chicago Fed National Financial Conditions Index
)
FRED_MONTHLY_SERIES: tuple[str, ...] = (
    # US Inflation
    "CPIAUCSL",        # CPI All Items
    "CPILFESL",        # Core CPI (hors alimentation/energie)
    "PCEPI",           # PCE Price Index
    "PCEPILFE",        # Core PCE (mesure preferee de la Fed)

    # US Emploi
    "UNRATE",          # Taux de chomage
    "PAYEMS",          # Nonfarm Payrolls
    "AWHMAN",          # Heures hebdo moyennes (manufacturing)
    "CES0500000003",   # Salaire horaire moyen
    "JTSJOL",          # JOLTS Job Openings
    # US Activite economique
    "INDPRO",          # Production industrielle
    "RSAFS",           # Ventes au detail
    "DGORDER",         # Commandes de biens durables
    "TOTALSA",         # Ventes vehicules
    "BOPGSTB",         # Balance commerciale
    # US Immobilier
    "HOUST",           # Housing Starts
    "PERMIT",          # Permis de construire
    "CSUSHPISA",       # Case-Shiller Home Price Index
    # US Masse monetaire
    "M2SL",            # M2
    # US Confiance
    "UMCSENT",         # Michigan Consumer Sentiment
    # Eurozone
    "CP0000EZ19M086NEST",   # Eurozone HICP
    "LRHUTTTTEZM156S",      # Eurozone chomage
    "IRLTLT01EZM156N",      # Eurozone taux long terme
    "IRSTCI01EZM156N",      # Eurozone taux court terme
    # Chine
    "CHNCPIALLMINMEI",      # Chine CPI
)
FRED_QUARTERLY_SERIES: tuple[str, ...] = (
    "GDPC1",               # US PIB reel
    "A191RL1Q225SBEA",     # US taux de croissance PIB reel
    "CP",                  # US profits des entreprises (apres impots)
    "CLVMNACSCAB1GQEA19",  # Eurozone PIB reel
)
FRED_RATE_LIMIT_SLEEP: float = 0.5

# --- Cross-Asset ---

CROSS_ASSET_INDICES: tuple[str, ...] = (
    "^N225",      # Nikkei 225
    "^GDAXI",    # DAX
    "^FTSE",     # FTSE 100
    "^HSI",      # Hang Seng
    "000001.SS",  # Shanghai Composite
)
SECTOR_ETFS: tuple[str, ...] = (
    "XLB",   # Materials
    "XLE",   # Energy
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLK",   # Technology
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
)
RISK_APPETITE_SYMBOLS: tuple[str, ...] = (
    "TLT",      # iShares 20+ Year Treasury Bond ETF
    "GC=F",     # Gold Futures (remplace GOLDAMGBD228NLBM retire de FRED)
)

# --- Sentiment ---

TIINGO_API_URL: str = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"

AAII_SENTIMENT_URL: str = "https://www.aaii.com/files/surveys/sentiment.xls"
GPR_MONTHLY_URL: str = (
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
)

# --- Data Cleaning / Outliers ---

DEFAULT_RETURN_COL_CANDIDATES: tuple[str, ...] = (
    "stock_open_log_return",
    "open",
    "stock_adjusted_close_log_return",
    "adj_close",
)
TICKER_OUTLIER_ROLLING_WINDOW: int = 63
TICKER_OUTLIER_MIN_PERIODS: int = 20
TICKER_OUTLIER_MAD_THRESHOLD: float = 6.0
CROSS_SECTION_OUTLIER_MAD_THRESHOLD: float = 5.0
ELEVATED_RETURN_THRESHOLD: float = 0.04
EXTREME_RETURN_THRESHOLD: float = 0.08
