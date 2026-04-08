from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.broker_xtb.reference_snapshot import (
    DEFAULT_EFFECTIVE_FROM,
    build_xtb_reference_snapshot_from_pdf,
    download_xtb_equity_pdf,
    save_xtb_reference_snapshot,
)
from core.src.meta_model.data.paths import XTB_INSTRUMENT_SPECS_REFERENCE_JSON

LOGGER: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """Download XTB equity PDF, extract symbols, and save the reference snapshot."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with tempfile.TemporaryDirectory(prefix="xtb-equity-pdf-") as temp_dir:
        pdf_path = Path(temp_dir) / "equity-table-uk.pdf"
        download_xtb_equity_pdf(pdf_path)
        payload = build_xtb_reference_snapshot_from_pdf(
            pdf_path,
            effective_from=DEFAULT_EFFECTIVE_FROM,
        )
    output_path = save_xtb_reference_snapshot(
        payload,
        output_path=XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
    )
    stock_count = sum(1 for item in payload if item["instrument_group"] == "stock_cash")
    LOGGER.info(
        "Saved XTB reference snapshot: %s | stock_cash=%d",
        output_path,
        stock_count,
    )


if __name__ == "__main__":
    main()
