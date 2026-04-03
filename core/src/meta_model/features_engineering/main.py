from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.paths import (
    CLEANED_OUTPUT_PARQUET,
    DATA_FEATURES_ENGINEERING_DIR,
    FEATURES_OUTPUT_PARQUET,
    FEATURES_OUTPUT_SAMPLE_CSV,
)
from core.src.meta_model.features_engineering.config import (
    REQUIRED_TA_INPUT_COLUMNS,
    TA_FEATURE_PREFIX,
)
from core.src.meta_model.features_engineering.io import (
    load_cleaned_dataset,
    save_feature_dataset,
    save_lagged_feature_dataset,
)
from core.src.meta_model.features_engineering.lag_features import add_feature_lags
from core.src.meta_model.features_engineering.pipeline import (
    add_high_level_features,
    build_feature_dataset,
    build_ta_feature_dataset,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    DATA_FEATURES_ENGINEERING_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = load_cleaned_dataset(CLEANED_OUTPUT_PARQUET)
    featured = build_feature_dataset(cleaned)
    save_lagged_feature_dataset(
        featured,
        FEATURES_OUTPUT_PARQUET,
        FEATURES_OUTPUT_SAMPLE_CSV,
    )
    logging.getLogger(__name__).info("Feature engineering pipeline completed.")


__all__ = [
    "REQUIRED_TA_INPUT_COLUMNS",
    "TA_FEATURE_PREFIX",
    "add_feature_lags",
    "add_high_level_features",
    "build_feature_dataset",
    "build_ta_feature_dataset",
    "load_cleaned_dataset",
    "main",
    "save_feature_dataset",
    "save_lagged_feature_dataset",
]


if __name__ == "__main__":
    main()
