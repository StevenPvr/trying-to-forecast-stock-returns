from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OneStandardErrorSelection:
    trial_number: int
    objective_score: float
    objective_standard_error: float
    complexity_penalty: float
    row: dict[str, object]


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

    best_row = completed.loc[completed[objective_column].idxmin()]
    threshold = float(best_row[objective_column] + best_row[standard_error_column])
    candidates = pd.DataFrame(completed.loc[completed[objective_column] <= threshold].copy())
    ordered_candidates = candidates.sort_values(
        [complexity_column, objective_column, "trial_number"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    selected_row = ordered_candidates.iloc[0].to_dict()
    return OneStandardErrorSelection(
        trial_number=int(selected_row["trial_number"]),
        objective_score=float(selected_row[objective_column]),
        objective_standard_error=float(selected_row[standard_error_column]),
        complexity_penalty=float(selected_row[complexity_column]),
        row=selected_row,
    )
