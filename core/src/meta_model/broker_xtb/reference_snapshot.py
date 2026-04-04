from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import cast

from core.src.meta_model.data.paths import XTB_INSTRUMENT_SPECS_REFERENCE_JSON

XTB_EQUITY_TABLE_SOURCE_URL: str = "https://www.xtb.com/en/equity-table.pdf"
STOCK_CFD_PAGE_COUNT: int = 31
DEFAULT_EFFECTIVE_FROM: str = "2000-01-01"
_META_REFRESH_PATTERN: re.Pattern[str] = re.compile(
    r"url='([^']+)'",
    flags=re.IGNORECASE,
)
_US_SYMBOL_PATTERN: re.Pattern[str] = re.compile(
    r"([A-Z0-9]{1,8}(?:\.[A-Z0-9])?)\s*\.US\*?",
)
_CLOSE_ONLY_PATTERN: re.Pattern[str] = re.compile(
    r"([A-Z0-9]{1,8}(?:\.[A-Z0-9])?)\s*\.US\*?CLOSE ONLY\s*/",
)
_SYMBOL_NORMALIZATION_MAP: dict[str, str] = {
    "BFB": "BF.B",
    "BRKB": "BRK.B",
}


def _download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _resolve_pdf_url(source_url: str) -> str:
    payload = _download_bytes(source_url)
    if payload.startswith(b"%PDF"):
        return source_url
    html = payload.decode("utf-8", errors="ignore")
    match = _META_REFRESH_PATTERN.search(html)
    if match is None:
        raise RuntimeError(f"Could not resolve PDF redirect from {source_url}.")
    return match.group(1)


def download_xtb_equity_pdf(
    output_path: Path,
    *,
    source_url: str = XTB_EQUITY_TABLE_SOURCE_URL,
) -> Path:
    pdf_url = _resolve_pdf_url(source_url)
    payload = _download_bytes(pdf_url)
    if not payload.startswith(b"%PDF"):
        raise RuntimeError(f"Resolved XTB document at {pdf_url} is not a PDF.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return output_path


def _load_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "".join(
        (page.extract_text() or "")
        for page in reader.pages[:STOCK_CFD_PAGE_COUNT]
    )


def extract_us_stock_symbols_from_pdf_text(pdf_text: str) -> list[str]:
    close_only_symbols = {
        symbol.replace(" ", "")
        for symbol in _CLOSE_ONLY_PATTERN.findall(pdf_text)
    }
    extracted_symbols = [
        symbol.replace(" ", "")
        for symbol in _US_SYMBOL_PATTERN.findall(pdf_text)
    ]
    ordered_symbols: list[str] = []
    seen: set[str] = set()
    for raw_symbol in extracted_symbols:
        if raw_symbol in close_only_symbols:
            continue
        normalized_symbol = cast(str, _SYMBOL_NORMALIZATION_MAP.get(raw_symbol, raw_symbol))
        if normalized_symbol in seen:
            continue
        seen.add(normalized_symbol)
        ordered_symbols.append(normalized_symbol)
    return ordered_symbols


def _build_stock_entry(symbol: str, *, effective_from: str) -> dict[str, object]:
    # Commission-free cash equity (e.g. XTB tier): no spread/swap/slippage in simulation; full cash margin.
    return {
        "symbol": symbol,
        "instrument_group": "stock_cfd",
        "currency": "USD",
        "spread_bps": 0.0,
        "slippage_bps": 0.0,
        "long_swap_bps_daily": 0.0,
        "short_swap_bps_daily": 0.0,
        "margin_requirement": 1.0,
        "max_adv_participation": 0.05,
        "effective_from": effective_from,
        "effective_to": None,
    }


def _build_index_entries(*, effective_from: str) -> list[dict[str, object]]:
    return [
        {
            "symbol": "DE40",
            "instrument_group": "index_cfd",
            "currency": "EUR",
            "spread_bps": 8.0,
            "slippage_bps": 2.0,
            "long_swap_bps_daily": 1.5,
            "short_swap_bps_daily": 1.5,
            "margin_requirement": 0.05,
            "max_adv_participation": 0.20,
            "effective_from": effective_from,
            "effective_to": None,
        },
        {
            "symbol": "UK100",
            "instrument_group": "index_cfd",
            "currency": "GBP",
            "spread_bps": 8.0,
            "slippage_bps": 2.0,
            "long_swap_bps_daily": 1.5,
            "short_swap_bps_daily": 1.5,
            "margin_requirement": 0.05,
            "max_adv_participation": 0.20,
            "effective_from": effective_from,
            "effective_to": None,
        },
        {
            "symbol": "US100",
            "instrument_group": "index_cfd",
            "currency": "USD",
            "spread_bps": 8.0,
            "slippage_bps": 2.0,
            "long_swap_bps_daily": 1.5,
            "short_swap_bps_daily": 1.5,
            "margin_requirement": 0.05,
            "max_adv_participation": 0.20,
            "effective_from": effective_from,
            "effective_to": None,
        },
        {
            "symbol": "US500",
            "instrument_group": "index_cfd",
            "currency": "USD",
            "spread_bps": 8.0,
            "slippage_bps": 2.0,
            "long_swap_bps_daily": 1.5,
            "short_swap_bps_daily": 1.5,
            "margin_requirement": 0.05,
            "max_adv_participation": 0.20,
            "effective_from": effective_from,
            "effective_to": None,
        },
    ]


def build_xtb_reference_snapshot_payload(
    stock_symbols: list[str],
    *,
    effective_from: str = DEFAULT_EFFECTIVE_FROM,
) -> list[dict[str, object]]:
    stock_entries = [
        _build_stock_entry(symbol, effective_from=effective_from)
        for symbol in sorted(stock_symbols)
    ]
    index_entries = _build_index_entries(effective_from=effective_from)
    return stock_entries + index_entries


def build_xtb_reference_snapshot_from_pdf(
    pdf_path: Path,
    *,
    effective_from: str = DEFAULT_EFFECTIVE_FROM,
) -> list[dict[str, object]]:
    pdf_text = _load_pdf_text(pdf_path)
    stock_symbols = extract_us_stock_symbols_from_pdf_text(pdf_text)
    if not stock_symbols:
        raise RuntimeError("No US stock CFD symbols were extracted from the XTB PDF.")
    return build_xtb_reference_snapshot_payload(
        stock_symbols,
        effective_from=effective_from,
    )


def save_xtb_reference_snapshot(
    payload: list[dict[str, object]],
    *,
    output_path: Path = XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "DEFAULT_EFFECTIVE_FROM",
    "XTB_EQUITY_TABLE_SOURCE_URL",
    "build_xtb_reference_snapshot_from_pdf",
    "build_xtb_reference_snapshot_payload",
    "download_xtb_equity_pdf",
    "extract_us_stock_symbols_from_pdf_text",
    "save_xtb_reference_snapshot",
]
