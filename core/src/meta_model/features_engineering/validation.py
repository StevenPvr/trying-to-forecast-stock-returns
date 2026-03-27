from __future__ import annotations

import pandas as pd

from core.src.meta_model.features_engineering.config import REQUIRED_TA_INPUT_COLUMNS


def validate_ta_input_columns(df: pd.DataFrame) -> None:
    missing_columns: list[str] = [
        col for col in REQUIRED_TA_INPUT_COLUMNS if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required TA input columns: " + ", ".join(missing_columns),
        )


def validate_base_columns(df: pd.DataFrame) -> None:
    validate_ta_input_columns(df)

    required_columns: tuple[str, ...] = ("date", "ticker")
    missing_columns: list[str] = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Input dataset must contain required columns: " + ", ".join(missing_columns),
        )

    duplicated_columns: list[str] = df.columns[df.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(
            "Input dataset contains duplicate column names: "
            + ", ".join(duplicated_columns),
        )

    duplicate_rows: pd.Series = df.duplicated(["date", "ticker"], keep=False)
    if duplicate_rows.any():
        raise ValueError("Input dataset contains duplicate (date, ticker) rows")


def prepare_input_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared: pd.DataFrame = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")

    for column in REQUIRED_TA_INPUT_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    return prepared.sort_values(["ticker", "date"]).reset_index(drop=True)


def validate_output_dataset(df: pd.DataFrame) -> None:
    duplicated_columns: list[str] = df.columns[df.columns.duplicated()].tolist()
    if duplicated_columns:
        raise RuntimeError(
            "Output dataset contains duplicate column names: "
            + ", ".join(duplicated_columns),
        )

    duplicate_rows: pd.Series = df.duplicated(["date", "ticker"], keep=False)
    if duplicate_rows.any():
        raise RuntimeError("Output dataset contains duplicate (date, ticker) rows")
