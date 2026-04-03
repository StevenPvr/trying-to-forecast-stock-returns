from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.broker_xtb.reference_snapshot import (
    build_xtb_reference_snapshot_payload,
    extract_us_stock_symbols_from_pdf_text,
)


def test_extract_us_stock_symbols_excludes_close_only_and_normalizes() -> None:
    pdf_text = (
        "AAPL.USApple Inc USD 50 USD 0 0 0.30% 0.50% 15:30 - 22:00Mon - Fri"
        "BRKB.USBerkshire Hathaway Inc - class B USD 50 USD 0 0 0.30% 0.50% 15:30 - 22:00Mon - Fri"
        "IPG.US*CLOSE ONLY / Interpublic Group of Cos IncUSD 50 USD 0 0 0.30% 0.50% 15:30 - 22:00Mon - Fri"
        "MSFT.USMicrosoft Corp USD 50 USD 0 0 0.30% 0.50% 15:30 - 22:00Mon - Fri"
    )

    result = extract_us_stock_symbols_from_pdf_text(pdf_text)

    assert result == ["AAPL", "BRK.B", "MSFT"]


def test_build_xtb_reference_snapshot_payload_adds_index_overlays() -> None:
    payload = build_xtb_reference_snapshot_payload(["AAPL", "MSFT"])

    stock_symbols = [item["symbol"] for item in payload if item["instrument_group"] == "stock_cfd"]
    index_symbols = [item["symbol"] for item in payload if item["instrument_group"] == "index_cfd"]

    assert stock_symbols == ["AAPL", "MSFT"]
    assert index_symbols == ["DE40", "UK100", "US100", "US500"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
