from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from core.src.meta_model.data.data_reference.reference_pipeline import (
    ReferenceBuildConfig,
    build_earnings_history,
    build_reference_outputs,
    resolve_fundamentals_source,
    save_reference_outputs,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = ReferenceBuildConfig()
    LOGGER.info(
        "Generating reference data with fundamentals source: %s",
        resolve_fundamentals_source(config),
    )
    membership_history, fundamentals_history = build_reference_outputs(config)
    earnings_history = build_earnings_history(fundamentals_history)
    output_paths = save_reference_outputs(
        membership_history=membership_history,
        fundamentals_history=fundamentals_history,
        earnings_history=earnings_history,
        membership_output_csv=config.membership_output_csv,
        fundamentals_output_csv=config.fundamentals_output_csv,
        earnings_output_csv=config.earnings_output_csv,
    )
    LOGGER.info(
        "Saved membership history to %s (%d rows, %d tickers)",
        output_paths["membership_history_csv"],
        len(membership_history),
        membership_history["ticker"].nunique(),
    )
    LOGGER.info(
        "Saved fundamentals history to %s (%d rows, %d tickers)",
        output_paths["fundamentals_history_csv"],
        len(fundamentals_history),
        fundamentals_history["ticker"].nunique(),
    )
    LOGGER.info(
        "Saved earnings history to %s (%d rows, %d tickers)",
        output_paths["earnings_history_csv"],
        len(earnings_history),
        earnings_history["ticker"].nunique(),
    )


if __name__ == "__main__":
    main()
