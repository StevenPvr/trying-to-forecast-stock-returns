from __future__ import annotations

"""Orchestrator for all data-fetching sub-pipelines (prices, macro, cross-asset, sentiment, calendar)."""

import logging
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, cast

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import numpy as np

from core.src.meta_model.broker_xtb.specs import (
    build_default_spec_provider,
    save_broker_snapshots,
)
from core.src.meta_model.broker_xtb.universe import (
    build_tradable_universe,
    save_tradable_universe,
)
from core.src.meta_model.data.constants import (
    RANDOM_SEED,
    SAMPLE_FRAC,
    XTB_DEFAULT_MAX_SPREAD_BPS,
)
from core.src.meta_model.runtime_parallelism import resolve_executor_worker_count

_TICKER_NAN_THRESHOLD: float = 1.0  # drop tickers with > 1 % NaN
_BETA_WINDOW_DAYS: int = 252
_BETA_MIN_PERIODS: int = 63
_COMPANY_SPARSE_COLUMNS: tuple[str, ...] = (
    "company_name",
    "sector",
    "industry",
    "market_cap",
    "trailing_p_e",
    "price_to_book",
    "beta",
    "profit_margins",
    "return_on_equity",
    "enterprise_value",
    "revenue_growth",
    "current_ratio",
    "book_value",
    "trailing_eps",
)

from core.src.meta_model.data.data_fetching.calendar_pipeline import CalendarConfig, build_calendar_dataset

from core.src.meta_model.data.data_fetching.cross_asset_pipeline import CrossAssetConfig, build_cross_asset_dataset
from core.src.meta_model.data.data_fetching.macro_pipeline import MacroConfig, build_macro_dataset
from core.src.meta_model.data.data_fetching.sentiment_pipeline import SentimentConfig, build_sentiment_dataset
from core.src.meta_model.data.data_fetching.sp500_pipeline import (
    PipelineConfig,
    load_constituents_table_from_wikipedia,
    build_dataset,
)
from core.src.meta_model.data.paths import (
    DATA_FETCHING_DIR,
    FUNDAMENTALS_HISTORY_CSV,
    MEMBERSHIP_HISTORY_CSV,
    MERGED_OUTPUT_PARQUET,
    MERGED_OUTPUT_SAMPLE_CSV,
    UNIVERSE_COMPANIES_XLSX,
    XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
)

LOGGER: logging.Logger = logging.getLogger(__name__)
UNIVERSE_NAME_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "company_name",
    "company_long_name",
    "company_short_name",
    "security",
    "name",
)

COLUMN_NAME_MAPPING: dict[str, str] = {
    "open": "stock_open_price",
    "high": "stock_high_price",
    "low": "stock_low_price",
    "close": "stock_close_price",
    "open_log_return": "stock_open_log_return",
    "high_log_return": "stock_high_log_return",
    "low_log_return": "stock_low_log_return",
    "close_log_return": "stock_close_log_return",
    "adj_close_log_return": "stock_adjusted_close_log_return",
    "volume": "stock_trading_volume",
    "sector": "company_sector",
    "industry": "company_industry",
    "market_cap": "company_market_cap_usd",
    "trailing_p_e": "company_trailing_pe_ratio",
    "price_to_book": "company_price_to_book_ratio",
    "beta": "company_beta",
    "profit_margins": "company_profit_margin_ratio",
    "return_on_equity": "company_return_on_equity_ratio",
    "enterprise_value": "company_enterprise_value_usd",
    "revenue_growth": "company_revenue_growth_ratio",
    "current_ratio": "company_current_ratio",
    "book_value": "company_book_value_per_share_usd",
    "trailing_eps": "company_trailing_eps_usd",
    "is_fomc_day": "calendar_is_fomc_announcement_day",
    "days_to_next_fomc": "calendar_days_until_next_fomc",
    "days_since_last_fomc": "calendar_days_since_previous_fomc",
    "is_fomc_week": "calendar_is_fomc_week",
    "is_quad_witching": "calendar_is_quadruple_witching_day",
    "days_to_next_quad_witching": "calendar_days_until_next_quadruple_witching",
    "aaii_bullish": "sentiment_aaii_bullish_share",
    "aaii_neutral": "sentiment_aaii_neutral_share",
    "aaii_bearish": "sentiment_aaii_bearish_share",
    "gpr_index": "sentiment_geopolitical_risk_index",
    "xa_dax_log_return": "cross_asset_dax_log_return",
    "xa_ftse_log_return": "cross_asset_ftse100_log_return",
    "xa_gold_log_return": "cross_asset_gold_log_return",
    "xa_hangseng_log_return": "cross_asset_hang_seng_log_return",
    "xa_nikkei_log_return": "cross_asset_nikkei225_log_return",
    "xa_shanghai_log_return": "cross_asset_shanghai_composite_log_return",
    "xa_tlt_log_return": "cross_asset_tlt_log_return",
    "xa_xlb_log_return": "cross_asset_materials_etf_log_return",
    "xa_xle_log_return": "cross_asset_energy_etf_log_return",
    "xa_xlf_log_return": "cross_asset_financials_etf_log_return",
    "xa_xli_log_return": "cross_asset_industrials_etf_log_return",
    "xa_xlk_log_return": "cross_asset_technology_etf_log_return",
    "xa_xlp_log_return": "cross_asset_consumer_staples_etf_log_return",
    "xa_xlu_log_return": "cross_asset_utilities_etf_log_return",
    "xa_xlv_log_return": "cross_asset_health_care_etf_log_return",
    "xa_xly_log_return": "cross_asset_consumer_discretionary_etf_log_return",
    "dgs5": "macro_us_treasury_5y_yield_pct",
    "dgs2": "macro_us_treasury_2y_yield_pct",
    "dgs10": "macro_us_treasury_10y_yield_pct",
    "dgs1": "macro_us_treasury_1y_yield_pct",
    "dgs30": "macro_us_treasury_30y_yield_pct",
    "dtb3": "macro_us_tbill_3m_yield_pct",
    "t10y2y": "macro_us_yield_spread_10y_minus_2y_pct",
    "t10y3m": "macro_us_yield_spread_10y_minus_3m_pct",
    "dff": "macro_us_fed_funds_effective_rate_pct",
    "dfii10": "macro_us_tips_10y_real_yield_pct",
    "t10yie": "macro_us_breakeven_inflation_10y_pct",
    "t5yie": "macro_us_breakeven_inflation_5y_pct",
    "t5yifr": "macro_us_forward_inflation_5y5y_pct",
    "bamlh0a0hym2": "macro_us_high_yield_oas_pct",
    "bamlc0a0cm": "macro_us_investment_grade_oas_pct",
    "vixcls": "macro_vix_close_level",
    "dcoilwtico": "macro_wti_crude_oil_price_usd",
    "dcoilbrenteu": "macro_brent_crude_oil_price_usd",
    "dtwexbgs": "macro_us_dollar_trade_weighted_index",
    "dhhngsp": "macro_henry_hub_natural_gas_price_usd",
    "dexjpus": "macro_fx_jpy_per_usd",
    "dexchus": "macro_fx_cny_per_usd",
    "usepuindxd": "macro_us_economic_policy_uncertainty_index",
    "dexuseu": "macro_fx_usd_per_eur",
    "walcl": "macro_fed_total_assets_usd_millions",
    "dexusuk": "macro_fx_usd_per_gbp",
    "stlfsi4": "macro_stl_financial_stress_index",
    "nfci": "macro_chicago_financial_conditions_index",
    "cpiaucsl": "macro_us_cpi_all_items_index",
    "cpilfesl": "macro_us_core_cpi_index",
    "pcepi": "macro_us_pce_price_index",
    "pcepilfe": "macro_us_core_pce_price_index",
    "unrate": "macro_us_unemployment_rate_pct",
    "mortgage30us": "macro_us_mortgage_rate_30y_pct",
    "payems": "macro_us_nonfarm_payrolls_thousands",
    "awhman": "macro_us_avg_weekly_hours_manufacturing",
    "ces0500000003": "macro_us_avg_hourly_earnings_usd",
    "jtsjol": "macro_us_jolts_job_openings_thousands",
    "indpro": "macro_us_industrial_production_index",
    "rsafs": "macro_us_retail_sales_millions_usd",
    "dgorder": "macro_us_durable_goods_orders_millions_usd",
    "totalsa": "macro_us_total_vehicle_sales_millions",
    "bopgstb": "macro_us_trade_balance_goods_services_millions_usd",
    "houst": "macro_us_housing_starts_thousands",
    "permit": "macro_us_building_permits_thousands",
    "csushpisa": "macro_us_case_shiller_home_price_index",
    "m2sl": "macro_us_m2_money_supply_billions_usd",
    "umcsent": "macro_us_consumer_sentiment_index",
    "cp0000ez19m086nest": "macro_euro_area_hicp_index",
    "lrhuttttezm156s": "macro_euro_area_unemployment_rate_pct",
    "irltlt01ezm156n": "macro_euro_area_long_term_rate_pct",
    "irstci01ezm156n": "macro_euro_area_short_term_rate_pct",
    "chncpiallminmei": "macro_china_cpi_index",
    "gdpc1": "macro_us_real_gdp_billions_chained_2017_usd",
    "a191rl1q225sbea": "macro_us_real_gdp_growth_rate_qoq_annualized_pct",
    "cp": "macro_us_corporate_profits_after_tax_billions_usd",
    "clvmnacscab1gqea19": "macro_euro_area_real_gdp_index",
}


def _column_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    return cast(pd.Series, frame.loc[:, column_name])


def _column_frame(frame: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    return cast(pd.DataFrame, frame.loc[:, column_names])


def _timestamp_or_raise(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Encountered NaT while resolving trading dates.")
    return cast(pd.Timestamp, timestamp)

_CROSS_ASSET_NAME_MAP: dict[str, str] = {
    "^N225": "nikkei", "^GDAXI": "dax", "^FTSE": "ftse",
    "^HSI": "hangseng", "000001.SS": "shanghai",
    "GC=F": "gold",
}


def _resolve_pipeline_config() -> PipelineConfig:
    membership_history_csv: Path | None = None
    fundamentals_history_csv: Path | None = None
    xtb_instrument_specs_json: Path | None = None

    if not MEMBERSHIP_HISTORY_CSV.exists():
        raise FileNotFoundError(
            f"Missing required point-in-time membership history at {MEMBERSHIP_HISTORY_CSV}.",
        )
    membership_history_csv = MEMBERSHIP_HISTORY_CSV

    if not FUNDAMENTALS_HISTORY_CSV.exists():
        raise FileNotFoundError(
            f"Missing required point-in-time fundamentals history at {FUNDAMENTALS_HISTORY_CSV}.",
        )
    fundamentals_history_csv = FUNDAMENTALS_HISTORY_CSV

    if not XTB_INSTRUMENT_SPECS_REFERENCE_JSON.exists():
        raise FileNotFoundError(
            "Missing required XTB instrument specification snapshot at "
            f"{XTB_INSTRUMENT_SPECS_REFERENCE_JSON}.",
        )
    xtb_instrument_specs_json = XTB_INSTRUMENT_SPECS_REFERENCE_JSON

    return PipelineConfig(
        membership_history_csv=membership_history_csv,
        fundamentals_history_csv=fundamentals_history_csv,
        xtb_only=True,
        xtb_instrument_specs_json=xtb_instrument_specs_json,
        require_xtb_snapshot=True,
        xtb_max_spread_bps=XTB_DEFAULT_MAX_SPREAD_BPS,
        allow_current_constituents_snapshot=False,
    )


def _clean_cross_asset_symbol(symbol: str) -> str:
    if symbol in _CROSS_ASSET_NAME_MAP:
        return _CROSS_ASSET_NAME_MAP[symbol]
    return symbol.lower().replace("-", "_").replace("=", "").replace("^", "")


def _pivot_cross_asset_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot cross-asset (date, ticker, adj_close) to wide: one col per symbol."""
    normalized: pd.DataFrame = df.copy()
    normalized["ticker"] = normalized["ticker"].apply(_clean_cross_asset_symbol)
    wide: pd.DataFrame = normalized.pivot_table(
        index="date", columns="ticker", values="adj_close",
    )
    wide.columns = [f"xa_{str(column_name)}" for column_name in wide.columns]
    return wide.reset_index()


def _nan_percentage_for_group(group: pd.DataFrame) -> float:
    return 100.0 * group.isna().sum().sum() / group.size


def _rolling_mean(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return cast(pd.Series, series.rolling(window, min_periods=min_periods).mean())


def _rolling_var(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return cast(pd.Series, series.rolling(window, min_periods=min_periods).var())





def _merge_date_features(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge all date-level DataFrames on 'date' with outer join."""
    if not frames:
        return pd.DataFrame(columns=["date"])
    result: pd.DataFrame = frames[0]
    for right in frames[1:]:
        result = result.merge(right, on="date", how="outer")
    return result


def _run_pipeline_task(
    name: str, func: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Run a single pipeline task."""
    LOGGER.info("Building %s...", name)
    return func()


def _build_cross_asset_wide() -> pd.DataFrame:
    return _pivot_cross_asset_wide(build_cross_asset_dataset(CrossAssetConfig()))


def _build_date_features(strict: bool = True) -> list[pd.DataFrame]:
    """Build all date-level feature DataFrames in parallel."""
    tasks: dict[str, Callable[[], pd.DataFrame]] = {
        "macro (FRED)": lambda: build_macro_dataset(MacroConfig()),
        "calendar features": lambda: build_calendar_dataset(CalendarConfig()),
        "sentiment (AAII + GPR)": lambda: build_sentiment_dataset(SentimentConfig()),
        "cross-asset (yfinance)": _build_cross_asset_wide,
    }
    frames: list[pd.DataFrame] = []
    failures: dict[str, Exception] = {}

    with ThreadPoolExecutor(max_workers=resolve_executor_worker_count(task_count=len(tasks))) as executor:
        futures: dict[Future[pd.DataFrame], str] = {
            executor.submit(_run_pipeline_task, name, func): name
            for name, func in tasks.items()
        }
        for future in as_completed(futures):
            name: str = futures[future]
            try:
                result: pd.DataFrame = future.result()
                frames.append(result)
            except Exception as exc:  # noqa: BLE001 - surface grouped failures
                failures[name] = exc
                LOGGER.error("%s pipeline failed: %s", name, exc)

    if failures and strict:
        failure_summary: str = ", ".join(
            f"{name}: {exc}" for name, exc in failures.items()
        )
        raise RuntimeError(f"Date feature pipelines failed: {failure_summary}")

    if failures:
        LOGGER.warning(
            "Proceeding with partial date features after failures: %s",
            ", ".join(failures.keys()),
        )

    return frames


def _drop_high_nan_tickers(
    df: pd.DataFrame, threshold: float = _TICKER_NAN_THRESHOLD,
) -> pd.DataFrame:
    """Drop tickers whose critical market columns exceed the NaN threshold."""
    price_columns: list[str] = [
        column
        for column in ("open", "high", "low", "close", "adj_close", "volume")
        if column in df.columns
    ]
    feature_cols: list[str] = (
        price_columns
        if price_columns
        else [str(column_name) for column_name in df.columns if column_name not in ("date", "ticker")]
    )
    nan_pct = cast(
        pd.Series,
        df.groupby("ticker")[feature_cols].apply(
            _nan_percentage_for_group,
        ),
    )
    bad_tickers: list[str] = [
        str(ticker)
        for ticker in cast(list[object], nan_pct[nan_pct > threshold].index.tolist())
    ]
    if bad_tickers:
        LOGGER.info(
            "Dropping %d tickers with >%.1f%% NaN: %s",
            len(bad_tickers), threshold, ", ".join(sorted(bad_tickers)),
        )
    ticker_series = _column_series(df, "ticker")
    clean = cast(pd.DataFrame, df.loc[~ticker_series.isin(bad_tickers)].reset_index(drop=True))
    LOGGER.info(
        "After ticker filter: %d rows, %d tickers remaining",
        len(clean), clean["ticker"].nunique(),
    )
    return clean


def _drop_leading_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the earliest dates that still contain any NaN (macro warm-up)."""
    unique_dates: set[pd.Timestamp] = {
        _timestamp_or_raise(value)
        for value in pd.to_datetime(_column_series(df, "date")).tolist()
        if not pd.isna(value)
    }
    dates_sorted: list[pd.Timestamp] = sorted(unique_dates)
    excluded_columns: set[str] = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        *_COMPANY_SPARSE_COLUMNS,
    }
    feature_cols: list[str] = [str(column_name) for column_name in df.columns if column_name not in excluded_columns]
    if not feature_cols:
        return df.reset_index(drop=True)
    drop_dates: list[object] = []
    for d in dates_sorted:
        day_slice = _column_frame(df.loc[_column_series(df, "date") == d], feature_cols)
        if day_slice.isna().any().any():
            drop_dates.append(d)
        else:
            break  # stop at first complete date
    if drop_dates:
        LOGGER.info(
            "Dropping %d leading dates with NaN (up to %s)",
            len(drop_dates), drop_dates[-1],
        )
    date_series = _column_series(df, "date")
    clean = cast(pd.DataFrame, df.loc[~date_series.isin(drop_dates)].reset_index(drop=True))
    min_date, max_date = _describe_date_bounds(clean)
    LOGGER.info(
        "After leading-NaN trim: %d rows, date range %s → %s",
        len(clean), min_date, max_date,
    )
    return clean


def _describe_date_bounds(df: pd.DataFrame) -> tuple[str, str]:
    """Return a safe, log-friendly date range summary for a frame."""
    if "date" not in df.columns or df.empty:
        return "n/a", "n/a"
    date_series = cast(pd.Series, pd.to_datetime(_column_series(df, "date"), errors="coerce"))
    min_value = date_series.min()
    max_value = date_series.max()
    if pd.isna(min_value) or pd.isna(max_value):
        return "n/a", "n/a"
    min_date = pd.Timestamp(min_value).date().isoformat()
    max_date = pd.Timestamp(max_value).date().isoformat()
    return min_date, max_date


def _transform_price_columns_to_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-ticker daily log-return columns while preserving raw prices."""
    if "ticker" not in df.columns:
        LOGGER.info("No 'ticker' column — skipping log-return transformation.")
        return df

    price_columns: list[str] = [
        str(column_name)
        for column_name in df.columns
        if str(column_name) in {"open", "high", "low", "close", "adj_close"} or str(column_name).startswith("xa_")
    ]
    if not price_columns:
        LOGGER.info("No price columns found — skipping log-return transformation.")
        return df

    transformed: pd.DataFrame = pd.DataFrame(df.copy())
    price_frame = _column_frame(transformed, price_columns)
    safe_prices = cast(
        pd.DataFrame,
        price_frame.where(
            price_frame > 0,
        ),
    )
    log_prices: pd.DataFrame = safe_prices.apply(np.log)
    log_return_columns: pd.DataFrame = log_prices.groupby(
        _column_series(transformed, "ticker"), sort=False,
    ).diff()
    log_return_columns = log_return_columns.rename(
        columns={col: f"{col}_log_return" for col in price_columns},
    )
    for col in log_return_columns.columns:
        transformed[col] = log_return_columns[col]

    LOGGER.info(
        "Added log-return columns for %d price columns: %s",
        len(price_columns), ", ".join(price_columns),
    )
    return transformed


def _fill_missing_beta_from_market_returns(
    df: pd.DataFrame,
    window: int = _BETA_WINDOW_DAYS,
    min_periods: int = _BETA_MIN_PERIODS,
) -> pd.DataFrame:
    """Fill missing company beta using rolling stock-vs-market return beta."""
    required_columns = {"date", "ticker", "adj_close_log_return", "beta"}
    if not required_columns.issubset(df.columns):
        return df

    enriched: pd.DataFrame = df.copy()
    asset_return = cast(
        pd.Series,
        pd.to_numeric(_column_series(enriched, "adj_close_log_return"), errors="coerce"),
    )
    market_return = cast(
        pd.Series,
        asset_return.groupby(_column_series(enriched, "date")).transform("mean"),
    )
    return_product = asset_return * market_return

    ticker_series = _column_series(enriched, "ticker")
    rolling_asset_mean = cast(
        pd.Series,
        asset_return.groupby(ticker_series).transform(
            lambda series: _rolling_mean(cast(pd.Series, series), window, min_periods),
        ),
    )
    rolling_market_mean = cast(
        pd.Series,
        market_return.groupby(ticker_series).transform(
            lambda series: _rolling_mean(cast(pd.Series, series), window, min_periods),
        ),
    )
    rolling_product_mean = cast(
        pd.Series,
        return_product.groupby(ticker_series).transform(
            lambda series: _rolling_mean(cast(pd.Series, series), window, min_periods),
        ),
    )
    rolling_market_var = cast(
        pd.Series,
        market_return.groupby(ticker_series).transform(
            lambda series: _rolling_var(cast(pd.Series, series), window, min_periods),
        ),
    )

    rolling_covariance = rolling_product_mean - (rolling_asset_mean * rolling_market_mean)
    beta_proxy = rolling_covariance / rolling_market_var.where(rolling_market_var.ne(0.0))
    missing_beta = cast(
        pd.Series,
        pd.to_numeric(_column_series(enriched, "beta"), errors="coerce").isna(),
    )
    enriched.loc[missing_beta, "beta"] = beta_proxy.loc[missing_beta]

    LOGGER.info(
        "Filled rolling beta proxy for %d rows.",
        int(cast(pd.Series, pd.to_numeric(_column_series(enriched, "beta"), errors="coerce").notna()).sum()),
    )
    return enriched


def _drop_unneeded_raw_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only raw OHLC columns; drop other raw price levels."""
    columns_to_drop: list[str] = [
        str(column_name)
        for column_name in df.columns
        if str(column_name) == "adj_close" or (str(column_name).startswith("xa_") and not str(column_name).endswith("_log_return"))
    ]
    if not columns_to_drop:
        return df

    trimmed: pd.DataFrame = df.drop(columns=columns_to_drop)
    LOGGER.info(
        "Dropped raw non-OHLC price columns: %s",
        ", ".join(columns_to_drop),
    )
    return trimmed


def _rename_columns_explicitly(df: pd.DataFrame) -> pd.DataFrame:
    """Rename dataset columns using an explicit, human-readable mapping."""
    present_mapping: dict[str, str] = {
        old_name: new_name
        for old_name, new_name in COLUMN_NAME_MAPPING.items()
        if old_name in df.columns
    }
    renamed: pd.DataFrame = df.rename(columns=present_mapping)
    LOGGER.info(
        "Renamed %d columns with explicit mapping.",
        len(present_mapping),
    )
    return renamed


def _save_merged(data: pd.DataFrame) -> dict[str, Path]:
    """Save the single merged parquet + 5 % sample CSV."""
    output_dir: Path = DATA_FETCHING_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path: Path = output_dir / MERGED_OUTPUT_PARQUET.name
    csv_path: Path = output_dir / MERGED_OUTPUT_SAMPLE_CSV.name
    data.to_parquet(parquet_path, index=False)
    sample: pd.DataFrame = data.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
    sample = sample.sort_values(["date", "ticker"]).reset_index(drop=True)
    sample.to_csv(csv_path, index=False)
    LOGGER.info("Saved merged parquet: %s (%d rows x %d cols)", parquet_path, len(data), len(data.columns))
    LOGGER.info("Saved merged sample CSV: %s", csv_path)
    return {"parquet": parquet_path, "sample_csv": csv_path}


def _build_universe_company_listing(
    data: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    listing: pd.DataFrame = pd.DataFrame({
        "ticker": sorted(
            cast(
                list[str],
                pd.Index(_column_series(data, "ticker")).dropna().astype(str).unique().tolist(),
            ),
        ),
    })
    listing["company_name"] = ""

    for column_name in UNIVERSE_NAME_CANDIDATE_COLUMNS:
        if column_name not in data.columns:
            continue
        names = (
            _column_frame(data, ["ticker", column_name])
            .dropna(subset=["ticker"])
            .assign(ticker=lambda frame: frame["ticker"].astype(str))
        )
        names = names.loc[names[column_name].astype(str).str.strip() != ""].copy()
        if names.empty:
            continue
        names = (
            names.rename(columns={column_name: "company_name"})
            .drop_duplicates(subset=["ticker"])
            .loc[:, ["ticker", "company_name"]]
        )
        listing = listing.merge(names, on="ticker", how="left", suffixes=("", "_candidate"))
        candidate_mask = listing["company_name"].astype(str).str.strip() == ""
        listing.loc[candidate_mask, "company_name"] = (
            listing.loc[candidate_mask, "company_name_candidate"].fillna("").astype(str)
        )
        listing = listing.drop(columns=["company_name_candidate"])

    listing = _fill_universe_company_names_from_membership_history(listing, config)
    listing = _fill_universe_company_names_from_snapshot(listing, config)
    listing["company_name"] = listing["company_name"].fillna("").astype(str)
    return listing.sort_values("ticker").reset_index(drop=True)


def _fill_universe_company_names_from_membership_history(
    listing: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    membership_path = getattr(config, "membership_history_csv", None)
    if not isinstance(membership_path, (str, Path)):
        return listing
    membership_path = Path(membership_path)
    if not membership_path.exists():
        return listing
    membership = pd.read_csv(membership_path)
    rename_map = {"symbol": "ticker"}
    membership = membership.rename(columns=rename_map)
    if "ticker" not in membership.columns:
        return listing
    name_column = next(
        (column for column in UNIVERSE_NAME_CANDIDATE_COLUMNS if column in membership.columns),
        None,
    )
    if name_column is None:
        return listing
    names = (
        _column_frame(membership, ["ticker", name_column])
        .dropna(subset=["ticker"])
        .assign(ticker=lambda frame: frame["ticker"].astype(str))
        .rename(columns={name_column: "company_name"})
    )
    names = names.loc[names["company_name"].astype(str).str.strip() != ""].copy()
    if names.empty:
        return listing
    names = names.drop_duplicates(subset=["ticker"]).loc[:, ["ticker", "company_name"]]
    merged = listing.merge(names, on="ticker", how="left", suffixes=("", "_membership"))
    empty_mask = merged["company_name"].astype(str).str.strip() == ""
    merged.loc[empty_mask, "company_name"] = (
        merged.loc[empty_mask, "company_name_membership"].fillna("").astype(str)
    )
    return merged.drop(columns=["company_name_membership"])


def _fill_universe_company_names_from_snapshot(
    listing: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    allow_snapshot = getattr(config, "allow_current_constituents_snapshot", False)
    if not isinstance(allow_snapshot, bool) or not allow_snapshot:
        return listing
    if not bool((listing["company_name"].astype(str).str.strip() == "").any()):
        return listing
    snapshot = load_constituents_table_from_wikipedia()
    snapshot = _column_frame(
        snapshot.drop_duplicates(subset=["ticker"]),
        ["ticker", "company_name"],
    )
    merged = listing.merge(snapshot, on="ticker", how="left", suffixes=("", "_snapshot"))
    empty_mask = merged["company_name"].astype(str).str.strip() == ""
    merged.loc[empty_mask, "company_name"] = (
        merged.loc[empty_mask, "company_name_snapshot"].fillna("").astype(str)
    )
    return merged.drop(columns=["company_name_snapshot"])


def _save_universe_company_listing(
    listing: pd.DataFrame,
    output_path: Path = UNIVERSE_COMPANIES_XLSX,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    listing.to_excel(output_path, index=False)
    LOGGER.info("Saved universe company listing Excel: %s (%d tickers)", output_path, len(listing))
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    DATA_FETCHING_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_config: PipelineConfig = _resolve_pipeline_config()

    LOGGER.info("Launching all pipelines in parallel...")
    with ThreadPoolExecutor(max_workers=resolve_executor_worker_count(task_count=2)) as executor:
        sp500_future: Future[pd.DataFrame] = executor.submit(
            build_dataset, pipeline_config,
        )
        date_future: Future[list[pd.DataFrame]] = executor.submit(
            _build_date_features,
        )
        sp500_df: pd.DataFrame = sp500_future.result()
        date_frames: list[pd.DataFrame] = date_future.result()

    date_features: pd.DataFrame = _merge_date_features(date_frames)
    LOGGER.info("Date-level features: %d cols", len(date_features.columns) - 1)

    merged: pd.DataFrame = sp500_df.merge(date_features, on="date", how="left")
    merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
    LOGGER.info("Merged dataset: %d rows x %d columns", len(merged), len(merged.columns))

    merged = _drop_high_nan_tickers(merged)
    merged = _drop_leading_nan_rows(merged)
    merged = _transform_price_columns_to_log_returns(merged)
    merged = _fill_missing_beta_from_market_returns(merged)
    merged = _drop_unneeded_raw_price_columns(merged)

    # Filter to 2007 onwards
    initial_rows: int = len(merged)
    merged = merged.loc[merged["date"] >= "2007-01-01"].copy()
    LOGGER.info(
        "Filtered to dates >= 2007-01-01: dropped %d rows (%d → %d)",
        initial_rows - len(merged), initial_rows, len(merged),
    )

    merged = _rename_columns_explicitly(merged)
    universe_listing = _build_universe_company_listing(merged, pipeline_config)
    _save_universe_company_listing(universe_listing)
    xtb_specs_path = getattr(pipeline_config, "xtb_instrument_specs_json", None)
    if bool(getattr(pipeline_config, "xtb_only", False)) and isinstance(xtb_specs_path, Path):
        spec_provider = build_default_spec_provider(
            path=xtb_specs_path,
            allow_defaults_if_missing=not pipeline_config.require_xtb_snapshot,
            require_explicit_symbols=pipeline_config.require_xtb_snapshot,
        )
        save_broker_snapshots(spec_provider)
        latest_trade_date_raw = pd.to_datetime(_column_series(merged, "date"), errors="coerce").max()
        if pd.isna(latest_trade_date_raw):
            raise RuntimeError("Unable to determine latest trade date from merged dataset.")
        latest_trade_date = _timestamp_or_raise(latest_trade_date_raw)
        tradable_universe = build_tradable_universe(
            merged,
            spec_provider,
            trade_date=latest_trade_date,
            max_spread_bps=(
                pipeline_config.xtb_max_spread_bps
                if pipeline_config.xtb_max_spread_bps is not None
                else XTB_DEFAULT_MAX_SPREAD_BPS
            ),
        )
        save_tradable_universe(tradable_universe)

    _save_merged(merged)
    LOGGER.info("Pipeline completed.")


if __name__ == "__main__":
    main()
