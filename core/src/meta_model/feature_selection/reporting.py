from __future__ import annotations

import pandas as pd

from core.src.meta_model.model_contract import is_excluded_feature_column

FEATURE_NAME_COLUMN: str = "feature_name"
FEATURE_FAMILY_COLUMN: str = "feature_family"
SELECTED_COLUMN: str = "selected"
DROP_REASON_COLUMN: str = "drop_reason"


def build_selection_report(
    group_manifest: pd.DataFrame,
    selected_feature_names: list[str],
    *,
    wrapper_search_history: pd.DataFrame,
) -> pd.DataFrame:
    selected_feature_values = list(selected_feature_names)
    report = pd.DataFrame(group_manifest.copy())
    report[SELECTED_COLUMN] = report[FEATURE_NAME_COLUMN].isin(selected_feature_values)
    report[DROP_REASON_COLUMN] = "not_selected"
    report.loc[report[FEATURE_NAME_COLUMN].isin(selected_feature_values), DROP_REASON_COLUMN] = "selected"
    if not wrapper_search_history.empty:
        best_objective_series = wrapper_search_history["objective_score"]
        report["best_objective_score"] = float(best_objective_series.max())
    return report.sort_values(
        [SELECTED_COLUMN, "group_id", FEATURE_NAME_COLUMN],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def validate_filtered_dataset_matches_selection(
    filtered_dataset: pd.DataFrame,
    selected_feature_names: list[str],
) -> None:
    actual_feature_names = [
        column_name
        for column_name in filtered_dataset.columns
        if not is_excluded_feature_column(column_name)
        and pd.api.types.is_numeric_dtype(filtered_dataset[column_name])
    ]
    if actual_feature_names != selected_feature_names:
        raise ValueError("Feature schema mismatch between selected features and filtered dataset.")


__all__ = [
    "DROP_REASON_COLUMN",
    "FEATURE_FAMILY_COLUMN",
    "FEATURE_NAME_COLUMN",
    "SELECTED_COLUMN",
    "build_selection_report",
    "validate_filtered_dataset_matches_selection",
]
