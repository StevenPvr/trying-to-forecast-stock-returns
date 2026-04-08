from __future__ import annotations

"""WRDS data provider: credentials, connection, security aliases, and fundamentals."""

import dataclasses
import importlib
import os
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import sqlalchemy as sa

_MANUAL_SECURITY_ALIASES: dict[str, tuple[str, ...]] = {
    "ANTM": ("ELV",),
}


@dataclasses.dataclass(frozen=True)
class WrdsCredentials:
    username: str
    password: str


def resolve_wrds_credentials() -> WrdsCredentials | None:
    username = os.environ.get("ID_WRDS", "").strip() or os.environ.get(
        "WRDS_USER",
        "",
    ).strip()
    password = os.environ.get("PASSWORD_WRDS", "").strip() or os.environ.get(
        "WRDS_PASSWORD",
        "",
    ).strip()
    if not username or not password:
        return None
    return WrdsCredentials(username=username, password=password)


def wrds_package_available() -> bool:
    return importlib.util.find_spec("wrds") is not None


@contextmanager
def open_wrds_connection(credentials: WrdsCredentials) -> Iterator[object]:
    wrds_module = importlib.import_module("wrds")
    connection = wrds_module.Connection(
        autoconnect=False,
        verbose=False,
        wrds_username=credentials.username,
    )
    connection._password = credentials.password
    make_engine_conn = getattr(connection, "_Connection__make_sa_engine_conn", None)
    if callable(make_engine_conn):
        make_engine_conn(raise_err=True)
    else:
        connection.connect()
    connection.load_library_list()
    try:
        yield connection
    finally:
        connection.close()


def _lookup_variants(symbol: str) -> set[str]:
    base = str(symbol).strip().upper()
    if not base:
        return set()
    variants = {
        base,
        base.replace(".", ""),
        base.replace(".", "-"),
        base.replace("-", ""),
        base.replace("-", "."),
    }
    for alias in _MANUAL_SECURITY_ALIASES.get(base, ()):
        variants.update(_lookup_variants(alias))
    return {value for value in variants if value}


def _build_security_sql() -> str:
    return """
        select
            gvkey,
            tic,
            ibtic,
            iid,
            secstat,
            dldtei
        from comp.security
        where upper(tic) = any(:symbols)
           or upper(ibtic) = any(:symbols)
    """


def _build_fundq_sql() -> str:
    return """
        select
            gvkey,
            tic as source_ticker,
            datadate,
            rdq,
            prccq,
            cshoq,
            epspxq,
            ceqq,
            actq,
            lctq,
            niq,
            saleq,
            dlttq,
            dlcq,
            cheq
        from comp.fundq
        where gvkey = any(:gvkeys)
          and datadate between :start_date and :end_date
          and indfmt = 'INDL'
          and datafmt = 'STD'
          and popsrc = 'D'
          and consol = 'C'
    """


def _execute_query(
    connection: object,
    sql: str,
    params: dict[str, object],
) -> pd.DataFrame:
    sql_result = connection.connection.execute(sa.text(sql), params)
    return pd.DataFrame(sql_result.fetchall(), columns=list(sql_result.keys()))


def _fetch_security_aliases_with_connection(
    connection: object,
    tickers: list[str],
) -> pd.DataFrame:
    requested = sorted(
        {
            variant
            for ticker in tickers
            for variant in _lookup_variants(ticker)
        }
    )
    if not requested:
        return pd.DataFrame()
    result = _execute_query(
        connection,
        _build_security_sql(),
        {"symbols": requested},
    )
    if result.empty:
        return result
    result["tic"] = result["tic"].astype(str).str.upper()
    result["ibtic"] = result["ibtic"].fillna("").astype(str).str.upper()
    result["dldtei"] = pd.to_datetime(result["dldtei"], errors="coerce")
    return result.sort_values(["gvkey", "iid"]).reset_index(drop=True)


def fetch_security_aliases(
    credentials: WrdsCredentials,
    tickers: list[str],
) -> pd.DataFrame:
    with open_wrds_connection(credentials) as connection:
        return _fetch_security_aliases_with_connection(connection, tickers)


def resolve_requested_ticker_map(
    tickers: list[str],
    security_aliases: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if security_aliases.empty:
        return pd.DataFrame(columns=["requested_ticker", "gvkey", "source_ticker"])
    for ticker in tickers:
        requested = str(ticker).strip().upper()
        variants = _lookup_variants(requested)
        matches = security_aliases.loc[
            security_aliases["tic"].isin(variants)
            | security_aliases["ibtic"].isin(variants)
        ].copy()
        if matches.empty:
            continue
        matches["exact_tic"] = matches["tic"].eq(requested)
        matches["exact_ibtic"] = matches["ibtic"].eq(requested)
        matches["active_security"] = matches["secstat"].fillna("").eq("A")
        matches["open_ended"] = matches["dldtei"].isna()
        best_match = matches.sort_values(
            ["exact_tic", "exact_ibtic", "active_security", "open_ended", "iid"],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        rows.append(
            {
                "requested_ticker": requested,
                "gvkey": str(best_match["gvkey"]),
                "source_ticker": str(best_match["tic"]),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["requested_ticker"])


def fetch_fundq_history(
    credentials: WrdsCredentials,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    with open_wrds_connection(credentials) as connection:
        security_aliases = _fetch_security_aliases_with_connection(connection, tickers)
        ticker_map = resolve_requested_ticker_map(tickers, security_aliases)
        if ticker_map.empty:
            return pd.DataFrame()
        result = _execute_query(
            connection,
            _build_fundq_sql(),
            {
                "gvkeys": sorted(ticker_map["gvkey"].unique().tolist()),
                "start_date": start_date,
                "end_date": end_date,
            },
        )
    if result.empty:
        return result
    result["gvkey"] = result["gvkey"].astype(str)
    result = result.merge(ticker_map, on="gvkey", how="inner")
    result = result.rename(columns={"requested_ticker": "ticker"})
    result["datadate"] = pd.to_datetime(result["datadate"], errors="coerce")
    if "rdq" in result.columns:
        result["rdq"] = pd.to_datetime(result["rdq"], errors="coerce")
    return result.sort_values(["ticker", "datadate"]).reset_index(drop=True)
