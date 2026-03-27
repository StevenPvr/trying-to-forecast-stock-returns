from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.src.meta_model.data.paths import (
    XGBOOST_OPTUNA_BEST_PARAMS_JSON,
    XGBOOST_OPTUNA_TRIALS_PARQUET,
)


@dataclass(frozen=True)
class SelectedXGBoostConfiguration:
    selected_trial_number: int
    params: dict[str, Any]
    training_rounds: int


def load_selected_xgboost_configuration(
    best_params_path: Path = XGBOOST_OPTUNA_BEST_PARAMS_JSON,
    trials_path: Path = XGBOOST_OPTUNA_TRIALS_PARQUET,
) -> SelectedXGBoostConfiguration:
    if not best_params_path.exists():
        raise FileNotFoundError(f"Best-params JSON not found: {best_params_path}")
    if not trials_path.exists():
        raise FileNotFoundError(f"Trials parquet not found: {trials_path}")

    payload = json.loads(best_params_path.read_text(encoding="utf-8"))
    selected_payload = payload.get("selected_trial_one_standard_error")
    if not isinstance(selected_payload, dict):
        raise ValueError("Optimization output does not include selected_trial_one_standard_error.")

    selected_trial_number = int(selected_payload["trial_number"])
    selected_params = dict(selected_payload["params"])
    boost_rounds = int(payload["config"]["boost_rounds"])

    trials = pd.read_parquet(trials_path)
    selected_trials = pd.DataFrame(
        trials.loc[trials["trial_number"] == selected_trial_number].copy(),
    )
    if selected_trials.empty:
        raise ValueError(
            f"Selected trial {selected_trial_number} not found in optimization trials parquet.",
        )
    selected_row = selected_trials.iloc[0]
    iteration_columns = sorted(
        column_name
        for column_name in selected_trials.columns
        if column_name.startswith("fold_") and column_name.endswith("_best_iteration")
    )
    if not iteration_columns:
        return SelectedXGBoostConfiguration(
            selected_trial_number=selected_trial_number,
            params=selected_params,
            training_rounds=boost_rounds,
        )

    iteration_values = np.asarray(
        [float(selected_row[column_name]) for column_name in iteration_columns],
        dtype=np.float64,
    )
    fold_weights = np.arange(1.0, len(iteration_values) + 1.0, dtype=np.float64)
    training_rounds = int(round(float(np.average(iteration_values, weights=fold_weights))))
    training_rounds = max(1, min(training_rounds, boost_rounds))
    return SelectedXGBoostConfiguration(
        selected_trial_number=selected_trial_number,
        params=selected_params,
        training_rounds=training_rounds,
    )
