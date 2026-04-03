from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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


def _load_json_object(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    raw_payload = cast(dict[object, object], payload)
    items: list[tuple[object, object]] = list(raw_payload.items())
    return {str(key): value for key, value in items}


def _normalize_params(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Selected trial params must be a JSON object.")
    raw_params = cast(dict[object, object], payload)
    items: list[tuple[object, object]] = list(raw_params.items())
    return {str(key): value for key, value in items}


def _normalize_object_mapping(payload: object, *, error_message: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(error_message)
    raw_payload = cast(dict[object, object], payload)
    items: list[tuple[object, object]] = list(raw_payload.items())
    return {str(key): value for key, value in items}


def _object_to_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, np.integer):
        return int(cast(np.integer[Any], value))
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"{field_name} must be convertible to int.")


def _select_trial_payload(payload: dict[str, object]) -> tuple[int, dict[str, Any]]:
    best_trial_number = payload.get("best_trial_number")
    best_params = payload.get("params")
    if best_trial_number is not None and best_params is not None:
        return (
            _object_to_int(best_trial_number, field_name="best_trial_number"),
            _normalize_params(best_params),
        )

    selected_payload = _normalize_object_mapping(
        payload.get("selected_trial_one_standard_error"),
        error_message="Optimization output does not include selected_trial_one_standard_error.",
    )
    return (
        _object_to_int(
            selected_payload["trial_number"],
            field_name="selected_trial_one_standard_error.trial_number",
        ),
        _normalize_params(selected_payload["params"]),
    )


def load_selected_xgboost_configuration(
    best_params_path: Path = XGBOOST_OPTUNA_BEST_PARAMS_JSON,
    trials_path: Path = XGBOOST_OPTUNA_TRIALS_PARQUET,
) -> SelectedXGBoostConfiguration:
    if not best_params_path.exists():
        raise FileNotFoundError(f"Best-params JSON not found: {best_params_path}")
    if not trials_path.exists():
        raise FileNotFoundError(f"Trials parquet not found: {trials_path}")

    payload = _load_json_object(best_params_path)
    selected_trial_number, selected_params = _select_trial_payload(payload)
    config_payload = _normalize_object_mapping(
        payload.get("config"),
        error_message="Optimization output does not include a valid config payload.",
    )
    boost_rounds = _object_to_int(
        config_payload["boost_rounds"],
        field_name="config.boost_rounds",
    )

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
