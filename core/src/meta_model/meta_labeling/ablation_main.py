from __future__ import annotations

"""Entry point for the meta-labeling refinement ablation study.

Screens all refinement strategies on train OOF predictions, selects the
top performers, then confirms on validation predictions.  Produces two
parquet artefacts and logs a summary table.
"""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from core.src.meta_model.data.paths import (
    META_ABLATION_CONFIRM_PARQUET,
    META_ABLATION_SCREEN_PARQUET,
    META_TRAIN_OOF_PREDICTIONS_PARQUET,
    META_VAL_PREDICTIONS_PARQUET,
)
from core.src.meta_model.meta_labeling.ablation import (
    ABLATION_REGIMES,
    format_ablation_table,
    run_ablation_screen,
    select_top_regimes,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def run_ablation_study() -> dict[str, pd.DataFrame]:
    """Screen strategies on train OOF, confirm top picks on val."""
    started_at: float = time.perf_counter()

    LOGGER.info("=== ABLATION STUDY: loading train OOF predictions ===")
    train_oof: pd.DataFrame = pd.read_parquet(META_TRAIN_OOF_PREDICTIONS_PARQUET)
    LOGGER.info("Train OOF loaded: rows=%d", len(train_oof))

    LOGGER.info("=== STEP 1: Screen all regimes on train OOF ===")
    screen_results: pd.DataFrame = run_ablation_screen(train_oof, ABLATION_REGIMES)
    screen_results.to_parquet(META_ABLATION_SCREEN_PARQUET)
    LOGGER.info("Screen results saved to %s", META_ABLATION_SCREEN_PARQUET)
    LOGGER.info("\n%s", format_ablation_table(screen_results))

    LOGGER.info("=== STEP 2: Select top regimes ===")
    top_names: list[str] = select_top_regimes(screen_results)
    LOGGER.info("Selected for confirmation: %s", top_names)

    LOGGER.info("=== STEP 3: Confirm on validation ===")
    val_predictions: pd.DataFrame = pd.read_parquet(META_VAL_PREDICTIONS_PARQUET)
    LOGGER.info("Validation loaded: rows=%d", len(val_predictions))
    confirm_regimes = tuple(
        regime for regime in ABLATION_REGIMES if str(regime["name"]) in top_names
    )
    confirm_results: pd.DataFrame = run_ablation_screen(val_predictions, confirm_regimes)
    confirm_results.to_parquet(META_ABLATION_CONFIRM_PARQUET)
    LOGGER.info("Confirm results saved to %s", META_ABLATION_CONFIRM_PARQUET)
    LOGGER.info("\n%s", format_ablation_table(confirm_results))

    elapsed: float = time.perf_counter() - started_at
    LOGGER.info("Ablation study completed in %.1fs", elapsed)
    return {"screen": screen_results, "confirm": confirm_results}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    run_ablation_study()


if __name__ == "__main__":
    main()
