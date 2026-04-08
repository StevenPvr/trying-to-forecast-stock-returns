from __future__ import annotations

"""Reference-data pipeline: membership history, fundamentals, earnings from WRDS or bootstrap."""

import dataclasses
import io
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import requests

from core.src.meta_model.broker_xtb.specs import build_default_spec_provider
from core.src.meta_model.data.constants import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    SP500_CURRENT_CONSTITUENTS_URL,
    SP500_CURRENT_FINANCIALS_URL,
    SP500_HISTORICAL_COMPONENTS_URL,
)
from core.src.meta_model.data.data_reference.wrds_provider import (
    fetch_fundq_history,
    resolve_wrds_credentials,
    wrds_package_available,
)
from core.src.meta_model.data.paths import (
    FUNDAMENTALS_HISTORY_CSV,
    MEMBERSHIP_HISTORY_CSV,
    REFERENCE_EARNINGS_HISTORY_CSV,
    WRDS_FUNDQ_EXTRACT_CSV,
    XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
)

LOGGER: logging.Logger = logging.getLogger(__name__)

_REQUEST_HEADERS: dict[str, str] = {"User-Agent": "prevision-sp500/0.1"}
_STRUCTURAL_COLUMNS: tuple[str, ...] = (
    "sector",
    "industry",
)
_NUMERIC_PROXY_COLUMNS: tuple[str, ...] = (
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
FUNDAMENTALS_PROVENANCE_COLUMN: str = "fundamentals_provenance"
WRDS_EVENT_PROVENANCE: str = "wrds_event"
BOOTSTRAP_STRUCTURAL_PROVENANCE: str = "bootstrap_structural"
BOOTSTRAP_PROXY_PROVENANCE: str = "bootstrap_proxy"
UNKNOWN_ANNOUNCEMENT_SESSION: str = "unknown"
_FUNDAMENTALS_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "company_name",
    *_STRUCTURAL_COLUMNS,
    *_NUMERIC_PROXY_COLUMNS,
    FUNDAMENTALS_PROVENANCE_COLUMN,
)
_EARNINGS_COLUMNS: tuple[str, ...] = (
    "ticker",
    "announcement_date",
    "announcement_session",
    "fiscal_year",
    "fiscal_quarter",
)


def _column_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    return cast(pd.Series, frame.loc[:, column_name])


def _column_frame(frame: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    return cast(pd.DataFrame, frame.loc[:, column_names])


def _timestamp_or_raise(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Encountered NaT while building reference outputs.")
    return cast(pd.Timestamp, timestamp)


def _normalize_symbol_value(value: object, known_symbols: set[str] | None = None) -> str:
    return _normalize_symbol(str(value), known_symbols)


@dataclasses.dataclass(frozen=True)
class ReferenceBuildConfig:
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    membership_output_csv: Path = MEMBERSHIP_HISTORY_CSV
    fundamentals_output_csv: Path = FUNDAMENTALS_HISTORY_CSV
    earnings_output_csv: Path = REFERENCE_EARNINGS_HISTORY_CSV
    xtb_instrument_specs_json: Path = XTB_INSTRUMENT_SPECS_REFERENCE_JSON
    historical_components_url: str = SP500_HISTORICAL_COMPONENTS_URL
    current_constituents_url: str = SP500_CURRENT_CONSTITUENTS_URL
    current_financials_url: str = SP500_CURRENT_FINANCIALS_URL
    fundamentals_source: str = "auto"
    wrds_fundq_extract_csv: Path = WRDS_FUNDQ_EXTRACT_CSV


def _download_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, headers=_REQUEST_HEADERS, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def _normalize_symbol(symbol: str, known_symbols: set[str] | None = None) -> str:
    cleaned = str(symbol).strip()
    if known_symbols and cleaned in known_symbols:
        return cleaned
    dotted = cleaned.replace("-", ".")
    if known_symbols and dotted in known_symbols:
        return dotted
    return dotted


def _normalize_symbol_column(
    frame: pd.DataFrame,
    source_column: str,
    known_symbols: set[str] | None = None,
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["ticker"] = (
        _column_series(normalized, source_column)
        .astype(str)
        .map(lambda value: _normalize_symbol_value(value, known_symbols))
    )
    return normalized


def _load_current_constituents(url: str) -> pd.DataFrame:
    raw = _download_csv(url)
    normalized = _normalize_symbol_column(raw, "Symbol")
    result = normalized.rename(
        columns={
            "Security": "company_name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        }
    )
    return _column_frame(
        result.drop_duplicates(subset=["ticker"]),
        ["ticker", "company_name", "sector", "industry"],
    )


def _load_current_financials(
    url: str,
    known_symbols: set[str] | None = None,
) -> pd.DataFrame:
    raw = _download_csv(url)
    normalized = _normalize_symbol_column(raw, "Symbol", known_symbols)
    result = normalized.rename(
        columns={
            "Name": "company_name",
            "Price": "price",
            "Market Cap": "market_cap",
            "Price/Earnings": "trailing_p_e",
            "Price/Book": "price_to_book",
            "Earnings/Share": "trailing_eps",
        }
    )
    columns = [
        "ticker",
        "company_name",
        "price",
        "market_cap",
        "trailing_p_e",
        "price_to_book",
        "trailing_eps",
    ]
    return _column_frame(result.drop_duplicates(subset=["ticker"]), columns)


def _load_wrds_fundq_extract(path: Path) -> pd.DataFrame:
    fundq = pd.read_csv(path)
    rename_map = {"tic": "ticker", "symbol": "ticker"}
    normalized = fundq.rename(columns=rename_map).copy()
    if "ticker" not in normalized.columns:
        raise ValueError("WRDS extract must contain 'tic', 'ticker', or 'symbol'")
    if "datadate" not in normalized.columns:
        raise ValueError("WRDS extract must contain 'datadate'")
    normalized["ticker"] = normalized["ticker"].astype(str).map(_normalize_symbol)
    normalized["datadate"] = pd.to_datetime(_column_series(normalized, "datadate"), errors="coerce")
    if "rdq" in normalized.columns:
        normalized["rdq"] = pd.to_datetime(_column_series(normalized, "rdq"), errors="coerce")
    else:
        normalized["rdq"] = pd.NaT
    return normalized.sort_values(["ticker", "datadate"]).reset_index(drop=True)


def _interval_end_dates(change_dates: pd.Series, final_end: pd.Timestamp) -> pd.Series:
    next_dates = pd.to_datetime(change_dates).shift(-1)
    end_dates = next_dates.sub(pd.Timedelta(days=1))
    end_dates.iloc[-1] = final_end
    return pd.Series(pd.to_datetime(end_dates), index=change_dates.index)


def _explode_historical_components(
    historical_components: pd.DataFrame,
    current_end: pd.Timestamp,
) -> pd.DataFrame:
    ordered = historical_components.copy()
    ordered["date"] = pd.to_datetime(_column_series(ordered, "date"))
    ordered = ordered.sort_values("date").reset_index(drop=True)
    ordered["end_date"] = _interval_end_dates(_column_series(ordered, "date"), current_end)
    exploded = ordered.assign(
        ticker_list=lambda frame: frame["tickers"].astype(str).str.split(","),
    ).explode("ticker_list")
    exploded["ticker"] = exploded["ticker_list"].astype(str).str.strip()
    return _column_frame(exploded, ["ticker", "date", "end_date"])


def _merge_ticker_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, group in intervals.groupby("ticker", sort=True):
        ordered = group.sort_values(["start_date", "end_date"]).reset_index(drop=True)
        current_start = _timestamp_or_raise(ordered.loc[0, "start_date"])
        current_end = _timestamp_or_raise(ordered.loc[0, "end_date"])
        for _, row in ordered.iloc[1:].iterrows():
            next_start = _timestamp_or_raise(row["start_date"])
            next_end = _timestamp_or_raise(row["end_date"])
            if next_start <= current_end + pd.Timedelta(days=1):
                current_end = max(current_end, next_end)
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "start_date": current_start,
                    "end_date": current_end,
                }
            )
            current_start = next_start
            current_end = next_end
        rows.append(
            {
                "ticker": ticker,
                "start_date": current_start,
                "end_date": current_end,
            }
        )
    return pd.DataFrame(rows)


def build_membership_history(
    historical_components: pd.DataFrame,
    current_constituents: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_ts = _timestamp_or_raise(start_date)
    end_ts = _timestamp_or_raise(end_date)
    exploded = _explode_historical_components(historical_components, end_ts)
    exploded["ticker"] = exploded["ticker"].map(_normalize_symbol)
    clipped = exploded.rename(columns={"date": "start_date"})
    clipped = clipped.loc[
        (clipped["end_date"] >= start_ts) & (clipped["start_date"] <= end_ts)
    ].copy()
    clipped["start_date"] = clipped["start_date"].clip(lower=start_ts)
    clipped["end_date"] = clipped["end_date"].clip(upper=end_ts)
    merged = _merge_ticker_intervals(clipped)
    names = _column_frame(
        current_constituents.drop_duplicates(subset=["ticker"]),
        ["ticker", "company_name"],
    )
    result = merged.merge(names, on="ticker", how="left")
    result["company_name"] = result["company_name"].fillna("").astype(str)
    return result.sort_values(["ticker", "start_date", "end_date"]).reset_index(drop=True)


def _build_structural_rows(
    base: pd.DataFrame,
    start_ts: pd.Timestamp,
) -> pd.DataFrame:
    structural = _column_frame(base, ["ticker", "company_name", "sector", "industry"]).copy()
    structural["date"] = start_ts
    for column in _NUMERIC_PROXY_COLUMNS:
        structural[column] = np.nan
    structural[FUNDAMENTALS_PROVENANCE_COLUMN] = BOOTSTRAP_STRUCTURAL_PROVENANCE
    return _column_frame(structural, list(_FUNDAMENTALS_COLUMNS))


def _build_proxy_rows(
    base: pd.DataFrame,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    proxy = base.copy()
    proxy["date"] = end_ts
    proxy["beta"] = np.nan
    proxy["profit_margins"] = np.nan
    proxy["return_on_equity"] = np.nan
    proxy["enterprise_value"] = np.nan
    proxy["revenue_growth"] = np.nan
    proxy["current_ratio"] = np.nan
    proxy[FUNDAMENTALS_PROVENANCE_COLUMN] = BOOTSTRAP_PROXY_PROVENANCE
    price_to_book = cast(
        pd.Series,
        pd.to_numeric(_column_series(proxy, "price_to_book"), errors="coerce"),
    )
    price = cast(
        pd.Series,
        pd.to_numeric(_column_series(proxy, "price"), errors="coerce"),
    )
    valid_price_to_book = (price_to_book > 0) & (price > 0)
    proxy["book_value"] = np.nan
    proxy.loc[valid_price_to_book, "book_value"] = (
        cast(pd.Series, pd.to_numeric(_column_series(proxy.loc[valid_price_to_book], "price"), errors="coerce"))
        / cast(pd.Series, pd.to_numeric(_column_series(proxy.loc[valid_price_to_book], "price_to_book"), errors="coerce"))
    )
    proxy = proxy.drop(columns=["price"], errors="ignore")
    populated = proxy.loc[
        proxy[["market_cap", "trailing_p_e", "price_to_book", "trailing_eps"]]
        .notna()
        .any(axis=1)
    ].copy()
    return _column_frame(populated, list(_FUNDAMENTALS_COLUMNS))


def build_fundamentals_history(
    membership_history: pd.DataFrame,
    current_constituents: pd.DataFrame,
    current_financials: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_ts = _timestamp_or_raise(start_date)
    end_ts = _timestamp_or_raise(end_date)
    tickers = pd.DataFrame(
        {"ticker": sorted(cast(list[str], pd.Index(_column_series(membership_history, "ticker")).dropna().astype(str).unique().tolist()))}
    )
    base = tickers.merge(
        _column_frame(current_constituents, ["ticker", "company_name", "sector", "industry"]),
        on="ticker",
        how="left",
    )
    base = base.merge(
        current_financials,
        on="ticker",
        how="left",
        suffixes=("", "_financials"),
    )
    if "company_name_financials" in base.columns:
        base["company_name"] = (
            base["company_name"]
            .fillna(base["company_name_financials"])
            .fillna(base["ticker"])
            .astype(str)
        )
        base = base.drop(columns=["company_name_financials"])
    else:
        base["company_name"] = base["company_name"].fillna(base["ticker"]).astype(str)
    structural = _build_structural_rows(base, start_ts)
    proxy = _build_proxy_rows(base, end_ts)
    result = cast(pd.DataFrame, pd.concat([structural, proxy], ignore_index=True))
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator
    return result.where(denominator.notna() & denominator.ne(0.0))


def _numeric_column_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return cast(pd.Series, pd.to_numeric(_column_series(frame, column), errors="coerce"))
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def build_wrds_fundamentals_history(
    membership_history: pd.DataFrame,
    current_constituents: pd.DataFrame,
    wrds_fundq: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    normalized_fundq = wrds_fundq.rename(columns={"tic": "ticker", "symbol": "ticker"}).copy()
    normalized_fundq["ticker"] = _column_series(normalized_fundq, "ticker").astype(str).map(_normalize_symbol)
    numeric_columns = [
        "prccq",
        "cshoq",
        "epspxq",
        "ceqq",
        "actq",
        "lctq",
        "niq",
        "saleq",
        "dlttq",
        "dlcq",
        "cheq",
    ]
    for column in numeric_columns:
        if column in normalized_fundq.columns:
            normalized_fundq[column] = pd.to_numeric(
                normalized_fundq[column],
                errors="coerce",
            )
    tickers = set(cast(list[str], pd.Index(_column_series(membership_history, "ticker")).dropna().astype(str).tolist()))
    filtered = cast(pd.DataFrame, normalized_fundq.loc[_column_series(normalized_fundq, "ticker").isin(list(tickers))].copy())
    available_date = cast(pd.Series, _column_series(filtered, "rdq").fillna(_column_series(filtered, "datadate") + pd.Timedelta(days=45)))
    filtered["date"] = pd.to_datetime(available_date, errors="coerce")
    filtered = filtered.loc[
        filtered["date"].notna()
        & (filtered["date"] >= start_ts)
        & (filtered["date"] <= end_ts)
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=_FUNDAMENTALS_COLUMNS)
    filtered["market_cap"] = (
        cast(pd.Series, pd.to_numeric(_column_series(filtered, "prccq"), errors="coerce"))
        * cast(pd.Series, pd.to_numeric(_column_series(filtered, "cshoq"), errors="coerce"))
        * 1_000_000.0
    )
    filtered["trailing_eps"] = cast(pd.Series, pd.to_numeric(_column_series(filtered, "epspxq"), errors="coerce"))
    filtered["trailing_p_e"] = _safe_ratio(
        cast(pd.Series, pd.to_numeric(_column_series(filtered, "prccq"), errors="coerce")),
        _column_series(filtered, "trailing_eps"),
    )
    filtered["book_value"] = _safe_ratio(
        cast(pd.Series, pd.to_numeric(_column_series(filtered, "ceqq"), errors="coerce")),
        cast(pd.Series, pd.to_numeric(_column_series(filtered, "cshoq"), errors="coerce")),
    )
    filtered["price_to_book"] = _safe_ratio(
        cast(pd.Series, pd.to_numeric(_column_series(filtered, "prccq"), errors="coerce")),
        _column_series(filtered, "book_value"),
    )
    filtered["current_ratio"] = _safe_ratio(
        cast(pd.Series, pd.to_numeric(_column_series(filtered, "actq"), errors="coerce")),
        cast(pd.Series, pd.to_numeric(_column_series(filtered, "lctq"), errors="coerce")),
    )
    filtered["profit_margins"] = _safe_ratio(
        cast(pd.Series, pd.to_numeric(_column_series(filtered, "niq"), errors="coerce")),
        cast(pd.Series, pd.to_numeric(_column_series(filtered, "saleq"), errors="coerce")),
    )
    filtered["return_on_equity"] = _safe_ratio(
        cast(pd.Series, pd.to_numeric(_column_series(filtered, "niq"), errors="coerce")),
        cast(pd.Series, pd.to_numeric(_column_series(filtered, "ceqq"), errors="coerce")),
    )
    filtered["revenue_growth"] = (
        cast(pd.Series, filtered.groupby("ticker", sort=False)["saleq"].pct_change(4))
    )
    long_term_debt = _numeric_column_or_nan(filtered, "dlttq")
    current_debt = _numeric_column_or_nan(filtered, "dlcq")
    cash_equivalents = _numeric_column_or_nan(filtered, "cheq")
    enterprise_value_mask = _column_series(filtered, "market_cap").notna() & (
        long_term_debt.notna() | current_debt.notna() | cash_equivalents.notna()
    )
    filtered["enterprise_value"] = np.nan
    filtered.loc[enterprise_value_mask, "enterprise_value"] = (
        _column_series(filtered.loc[enterprise_value_mask], "market_cap")
        + (long_term_debt.fillna(0.0).loc[enterprise_value_mask] * 1_000_000.0)
        + (current_debt.fillna(0.0).loc[enterprise_value_mask] * 1_000_000.0)
        - (cash_equivalents.fillna(0.0).loc[enterprise_value_mask] * 1_000_000.0)
    )
    filtered["beta"] = np.nan
    filtered[FUNDAMENTALS_PROVENANCE_COLUMN] = WRDS_EVENT_PROVENANCE
    static_fields = _column_frame(
        current_constituents.drop_duplicates(subset=["ticker"]),
        ["ticker", "company_name", "sector", "industry"],
    )
    result = filtered.merge(static_fields, on="ticker", how="left")
    result["company_name"] = _column_series(result, "company_name").fillna(_column_series(result, "ticker")).astype(str)
    return _column_frame(result, list(_FUNDAMENTALS_COLUMNS)).sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def merge_wrds_with_bootstrap_fallback(
    membership_history: pd.DataFrame,
    current_constituents: pd.DataFrame,
    current_financials: pd.DataFrame,
    wrds_history: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if FUNDAMENTALS_PROVENANCE_COLUMN not in wrds_history.columns:
        wrds_history = pd.DataFrame(wrds_history.copy())
        wrds_history[FUNDAMENTALS_PROVENANCE_COLUMN] = WRDS_EVENT_PROVENANCE
    wrds_tickers = set(pd.Index(wrds_history["ticker"]).dropna().astype(str))
    unresolved_membership = membership_history.loc[
        ~membership_history["ticker"].isin(wrds_tickers)
    ].reset_index(drop=True)
    if unresolved_membership.empty:
        return wrds_history.sort_values(["ticker", "date"]).reset_index(drop=True)
    bootstrap_history = build_fundamentals_history(
        membership_history=unresolved_membership,
        current_constituents=current_constituents,
        current_financials=current_financials,
        start_date=start_date,
        end_date=end_date,
    )
    LOGGER.warning(
        "Backfilled %d unresolved WRDS fundamentals tickers with bootstrap rows.",
        unresolved_membership["ticker"].nunique(),
    )
    combined = pd.concat([wrds_history, bootstrap_history], ignore_index=True)
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_earnings_history(fundamentals_history: pd.DataFrame) -> pd.DataFrame:
    if fundamentals_history.empty:
        return pd.DataFrame(columns=list(_EARNINGS_COLUMNS))
    required_columns = {"date", "ticker"}
    missing_columns = required_columns.difference(fundamentals_history.columns)
    if missing_columns:
        missing_preview = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Fundamentals history missing required earnings columns: {missing_preview}",
        )
    if FUNDAMENTALS_PROVENANCE_COLUMN not in fundamentals_history.columns:
        LOGGER.warning(
            "Skipping earnings history build because fundamentals provenance is missing.",
        )
        return pd.DataFrame(columns=list(_EARNINGS_COLUMNS))
    prepared = _column_frame(fundamentals_history.copy(), ["date", "ticker"])
    prepared[FUNDAMENTALS_PROVENANCE_COLUMN] = (
        fundamentals_history[FUNDAMENTALS_PROVENANCE_COLUMN].fillna("").astype(str).str.strip()
    )
    prepared["date"] = pd.to_datetime(_column_series(prepared, "date"), errors="coerce")
    prepared["ticker"] = _column_series(prepared, "ticker").astype(str).str.strip()
    prepared = prepared.loc[
        _column_series(prepared, "date").notna()
        & _column_series(prepared, "ticker").ne("")
        & _column_series(prepared, FUNDAMENTALS_PROVENANCE_COLUMN).eq(WRDS_EVENT_PROVENANCE)
    ].drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    if prepared.empty:
        return pd.DataFrame(columns=list(_EARNINGS_COLUMNS))
    announcement_dates = pd.to_datetime(_column_series(prepared, "date"))
    earnings_history = pd.DataFrame(
        {
            "ticker": _column_series(prepared, "ticker").astype(str),
            "announcement_date": announcement_dates.dt.strftime("%Y-%m-%d"),
            "announcement_session": UNKNOWN_ANNOUNCEMENT_SESSION,
            "fiscal_year": announcement_dates.dt.year.astype("int64"),
            "fiscal_quarter": (((announcement_dates.dt.month - 1) // 3) + 1).astype("int64"),
        }
    )
    return _column_frame(earnings_history, list(_EARNINGS_COLUMNS))


def ensure_earnings_history_output(
    fundamentals_history_path: Path = FUNDAMENTALS_HISTORY_CSV,
    earnings_output_csv: Path = REFERENCE_EARNINGS_HISTORY_CSV,
) -> Path:
    if not fundamentals_history_path.exists():
        raise FileNotFoundError(
            f"Missing fundamentals history required to build earnings history: {fundamentals_history_path}",
        )
    fundamentals_history = pd.read_csv(fundamentals_history_path)
    earnings_history = build_earnings_history(fundamentals_history)
    earnings_output_csv.parent.mkdir(parents=True, exist_ok=True)
    earnings_history.to_csv(earnings_output_csv, index=False)
    return earnings_output_csv


def resolve_fundamentals_source(config: ReferenceBuildConfig) -> str:
    if config.fundamentals_source == "wrds_direct":
        return "wrds_direct"
    if config.fundamentals_source == "wrds_extract":
        return "wrds_extract"
    if config.fundamentals_source == "open_source_bootstrap":
        return "open_source_bootstrap"
    if wrds_package_available() and resolve_wrds_credentials() is not None:
        return "wrds_direct"
    if config.wrds_fundq_extract_csv.exists():
        return "wrds_extract"
    return "open_source_bootstrap"


def _load_xtb_stock_symbols(path: Path) -> set[str]:
    provider = build_default_spec_provider(
        path=path,
        allow_defaults_if_missing=True,
        require_explicit_symbols=True,
    )
    return {
        spec.symbol
        for spec in provider.specs
        if spec.instrument_group == "stock_cash"
    }


def save_reference_outputs(
    membership_history: pd.DataFrame,
    fundamentals_history: pd.DataFrame,
    earnings_history: pd.DataFrame,
    membership_output_csv: Path,
    fundamentals_output_csv: Path,
    earnings_output_csv: Path,
) -> dict[str, Path]:
    membership_output_csv.parent.mkdir(parents=True, exist_ok=True)
    fundamentals_output_csv.parent.mkdir(parents=True, exist_ok=True)
    earnings_output_csv.parent.mkdir(parents=True, exist_ok=True)
    membership_history.to_csv(membership_output_csv, index=False)
    fundamentals_history.to_csv(fundamentals_output_csv, index=False)
    earnings_history.to_csv(earnings_output_csv, index=False)
    return {
        "membership_history_csv": membership_output_csv,
        "fundamentals_history_csv": fundamentals_output_csv,
        "earnings_history_csv": earnings_output_csv,
    }


def build_reference_outputs(
    config: ReferenceBuildConfig,
    historical_components: pd.DataFrame | None = None,
    current_constituents: pd.DataFrame | None = None,
    current_financials: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    xtb_symbols = _load_xtb_stock_symbols(config.xtb_instrument_specs_json)
    resolved_constituents = (
        current_constituents
        if current_constituents is not None
        else _load_current_constituents(config.current_constituents_url)
    )
    resolved_historical_components = (
        historical_components
        if historical_components is not None
        else _download_csv(config.historical_components_url)
    )
    membership_history = build_membership_history(
        historical_components=resolved_historical_components,
        current_constituents=resolved_constituents,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    xtb_membership = membership_history.loc[
        membership_history["ticker"].isin(xtb_symbols)
    ].reset_index(drop=True)
    source = resolve_fundamentals_source(config)
    if source == "wrds_direct":
        credentials = resolve_wrds_credentials()
        if credentials is None:
            raise ValueError("WRDS direct mode requires ID_WRDS/PASSWORD_WRDS credentials.")
        try:
            wrds_fundq = fetch_fundq_history(
                credentials=credentials,
                tickers=sorted(pd.Index(xtb_membership["ticker"]).dropna().astype(str).unique()),
                start_date=config.start_date,
                end_date=config.end_date,
            )
            fundamentals_history = build_wrds_fundamentals_history(
                membership_history=xtb_membership,
                current_constituents=resolved_constituents,
                wrds_fundq=wrds_fundq,
                start_date=config.start_date,
                end_date=config.end_date,
            )
            if fundamentals_history["ticker"].nunique() < xtb_membership["ticker"].nunique():
                resolved_financials = (
                    current_financials
                    if current_financials is not None
                    else _load_current_financials(
                        config.current_financials_url,
                        known_symbols=xtb_symbols,
                    )
                )
                fundamentals_history = merge_wrds_with_bootstrap_fallback(
                    membership_history=xtb_membership,
                    current_constituents=resolved_constituents,
                    current_financials=resolved_financials,
                    wrds_history=fundamentals_history,
                    start_date=config.start_date,
                    end_date=config.end_date,
                )
        except Exception:
            if config.fundamentals_source != "auto":
                raise
            LOGGER.warning(
                "WRDS direct fundamentals fetch failed; falling back to lower-priority sources.",
                exc_info=True,
            )
            source = (
                "wrds_extract"
                if config.wrds_fundq_extract_csv.exists()
                else "open_source_bootstrap"
            )
            fundamentals_history = pd.DataFrame()
    if source == "wrds_extract":
        wrds_fundq = _load_wrds_fundq_extract(config.wrds_fundq_extract_csv)
        fundamentals_history = build_wrds_fundamentals_history(
            membership_history=xtb_membership,
            current_constituents=resolved_constituents,
            wrds_fundq=wrds_fundq,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        if fundamentals_history["ticker"].nunique() < xtb_membership["ticker"].nunique():
            resolved_financials = (
                current_financials
                if current_financials is not None
                else _load_current_financials(
                    config.current_financials_url,
                    known_symbols=xtb_symbols,
                )
            )
            fundamentals_history = merge_wrds_with_bootstrap_fallback(
                membership_history=xtb_membership,
                current_constituents=resolved_constituents,
                current_financials=resolved_financials,
                wrds_history=fundamentals_history,
                start_date=config.start_date,
                end_date=config.end_date,
            )
    if source == "open_source_bootstrap":
        resolved_financials = (
            current_financials
            if current_financials is not None
            else _load_current_financials(
                config.current_financials_url,
                known_symbols=xtb_symbols,
            )
        )
        fundamentals_history = build_fundamentals_history(
            membership_history=xtb_membership,
            current_constituents=resolved_constituents,
            current_financials=resolved_financials,
            start_date=config.start_date,
            end_date=config.end_date,
        )
    return membership_history, fundamentals_history
