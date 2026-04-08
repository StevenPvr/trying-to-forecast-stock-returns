from __future__ import annotations

"""One-standard-error trial selection from the Optuna study ledger."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OneStandardErrorSelection:
    trial_number: int
    objective_score: float
    objective_standard_error: float
    complexity_penalty: float
    row: dict[str, object]


def _series_float(row: pd.Series, column_name: str) -> float:
    return float(np.asarray(row.at[column_name], dtype=np.float64).item())


def _series_int(row: pd.Series, column_name: str) -> int:
    return int(np.asarray(row.at[column_name], dtype=np.int64).item())


def _normalize_row_dict(row: pd.Series) -> dict[str, object]:
    raw_dict = row.to_dict()
    return {str(key): value for key, value in raw_dict.items()}


def select_one_standard_error_trial(
    trials_frame: pd.DataFrame,
    *,
    objective_column: str = "objective_score",
    standard_error_column: str = "objective_standard_error",
    complexity_column: str = "complexity_penalty",
) -> OneStandardErrorSelection:
    completed = pd.DataFrame(
        trials_frame.loc[
            trials_frame[objective_column].notna()
            & trials_frame[standard_error_column].notna()
            & trials_frame[complexity_column].notna()
        ].copy(),
    )
    if completed.empty:
        raise ValueError("No completed optimization trials are available for one-SE selection.")

    objective_values = completed[objective_column].to_numpy(dtype=np.float64, copy=False)
    best_row_position = int(np.argmin(objective_values))
    best_row = completed.iloc[best_row_position]
    threshold = _series_float(best_row, objective_column) + _series_float(
        best_row,
        standard_error_column,
    )
    candidates = pd.DataFrame(completed.loc[completed[objective_column] <= threshold].copy())
    ordered_candidates = candidates.sort_values(
        [complexity_column, objective_column, "trial_number"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    selected_row_series = ordered_candidates.iloc[0]
    selected_row = _normalize_row_dict(selected_row_series)
    return OneStandardErrorSelection(
        trial_number=_series_int(selected_row_series, "trial_number"),
        objective_score=_series_float(selected_row_series, objective_column),
        objective_standard_error=_series_float(selected_row_series, standard_error_column),
        complexity_penalty=_series_float(selected_row_series, complexity_column),
        row=selected_row,
    )
