from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def as_dataframe(value: object) -> pd.DataFrame:
    return cast(pd.DataFrame, value)


def as_series(value: object, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.Series(value.to_numpy(), index=value.index)
    return pd.Series(value, index=index)


def days_since_last_true(mask: pd.Series) -> pd.Series:
    values = mask.fillna(False).to_numpy(dtype=bool)
    result = np.full(len(values), np.nan, dtype=float)
    last_true_index = -1
    for index, value in enumerate(values):
        if value:
            last_true_index = index
            result[index] = 0.0
        elif last_true_index >= 0:
            result[index] = float(index - last_true_index)
    return pd.Series(result, index=mask.index)


__all__ = [
    "as_dataframe",
    "as_series",
    "days_since_last_true",
]
