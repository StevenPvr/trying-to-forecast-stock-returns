from __future__ import annotations

"""Helpers to enforce train-only fitting and prevent split leakage."""

import pandas as pd

from core.src.meta_model.model_contract import SPLIT_COLUMN, TRAIN_SPLIT_NAME


def assert_train_only_fit_frame(
    data: pd.DataFrame,
    *,
    split_column: str = SPLIT_COLUMN,
    train_split_name: str = TRAIN_SPLIT_NAME,
    context: str,
) -> None:
    """Raise if *data* contains rows outside the train split."""
    if split_column not in data.columns:
        raise ValueError(f"{context}: missing required split column '{split_column}'.")
    split_values = data[split_column].astype(str)
    invalid = sorted(set(split_values.unique()) - {train_split_name})
    if invalid:
        raise ValueError(
            f"{context}: fit must be train-only. Found forbidden split values: {','.join(invalid)}.",
        )
