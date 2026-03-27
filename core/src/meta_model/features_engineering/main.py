from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.paths import (
    CLEANED_OUTPUT_PARQUET,
    DATA_FEATURES_ENGINEERING_DIR,
    FEATURES_OUTPUT_PARQUET,
    FEATURES_OUTPUT_SAMPLE_CSV,
)
from core.src.secondary_model.data.paths import SECONDARY_OOS_PREDICTIONS_PARQUET
from core.src.meta_model.features_engineering.config import REQUIRED_TA_INPUT_COLUMNS, TA_FEATURE_PREFIX
from core.src.meta_model.features_engineering.io import (
    load_cleaned_dataset,
    save_feature_dataset,
    save_lagged_feature_dataset,
)
from core.src.meta_model.features_engineering.lag_features import add_feature_lags
from core.src.meta_model.features_engineering.pipeline import (
    build_feature_dataset,
    build_ta_feature_dataset,
)


def _first_non_null_value(values: pd.Series) -> float:
    non_null_values = values.dropna()
    if non_null_values.empty:
        return float("nan")
    return float(non_null_values.iloc[0])


def _collapse_secondary_oos_predictions(
    secondary_oos: pd.DataFrame,
    prediction_columns: list[str],
) -> pd.DataFrame:
    duplicate_mask = secondary_oos.duplicated(["date", "ticker"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = secondary_oos.loc[duplicate_mask, ["date", "ticker", *prediction_columns]]
        ambiguous_columns = [
            column_name
            for column_name in prediction_columns
            if duplicate_rows.groupby(["date", "ticker"])[column_name].nunique(dropna=True).gt(1).any()
        ]
        if ambiguous_columns:
            raise ValueError(
                "Secondary OOS predictions contain conflicting duplicate values for columns: "
                f"{', '.join(ambiguous_columns)}"
            )
        logging.getLogger(__name__).info(
            "Collapsing %d duplicate secondary OOS rows onto unique (date, ticker) keys.",
            int(duplicate_mask.sum()),
        )

    return pd.DataFrame(
        secondary_oos.loc[:, ["date", "ticker", *prediction_columns]].groupby(
            ["date", "ticker"],
            as_index=False,
            sort=False,
        ).agg({column_name: _first_non_null_value for column_name in prediction_columns}),
    )


def merge_secondary_oos_predictions(
    featured: pd.DataFrame,
    *,
    secondary_oos_predictions_path: Path | None = None,
) -> pd.DataFrame:
    source_path = (
        SECONDARY_OOS_PREDICTIONS_PARQUET
        if secondary_oos_predictions_path is None
        else secondary_oos_predictions_path
    )
    if not source_path.exists():
        logging.getLogger(__name__).info(
            "Secondary OOS predictions not found at %s; skipping merge.",
            source_path,
        )
        return featured

    secondary_oos = pd.read_parquet(source_path)
    prediction_columns = [
        column_name
        for column_name in secondary_oos.columns
        if column_name not in {"date", "ticker", "dataset_split"}
    ]
    if not prediction_columns:
        logging.getLogger(__name__).info(
            "Secondary OOS predictions found at %s but no prediction columns were available; skipping merge.",
            source_path,
        )
        return featured

    collapsed_secondary_oos = _collapse_secondary_oos_predictions(
        secondary_oos,
        prediction_columns,
    )
    merged = featured.merge(
        collapsed_secondary_oos,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    logging.getLogger(__name__).info(
        "Merged %d secondary OOS prediction columns from %s.",
        len(prediction_columns),
        source_path,
    )
    return merged


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    DATA_FEATURES_ENGINEERING_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = load_cleaned_dataset(CLEANED_OUTPUT_PARQUET)
    featured = build_feature_dataset(cleaned)
    featured = merge_secondary_oos_predictions(featured)
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
    "build_feature_dataset",
    "build_ta_feature_dataset",
    "load_cleaned_dataset",
    "main",
    "merge_secondary_oos_predictions",
    "save_feature_dataset",
    "save_lagged_feature_dataset",
]


if __name__ == "__main__":
    main()
