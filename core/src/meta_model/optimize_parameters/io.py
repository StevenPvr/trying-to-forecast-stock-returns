from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from core.src.meta_model.data.paths import (
    OPTIMIZATION_OVERFITTING_REPORT_JSON,
    OPTIMIZATION_TRIAL_LEDGER_CSV,
    OPTIMIZATION_TRIAL_LEDGER_PARQUET,
    XGBOOST_OPTUNA_BEST_PARAMS_JSON,
    XGBOOST_OPTUNA_TRIALS_CSV,
    XGBOOST_OPTUNA_TRIALS_PARQUET,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def save_optimization_outputs(
    trials_frame: pd.DataFrame,
    best_params: dict[str, Any],
    overfitting_report: dict[str, Any] | None = None,
    *,
    trials_parquet_path: Path = XGBOOST_OPTUNA_TRIALS_PARQUET,
    trials_csv_path: Path = XGBOOST_OPTUNA_TRIALS_CSV,
    best_params_path: Path = XGBOOST_OPTUNA_BEST_PARAMS_JSON,
    trial_ledger_parquet_path: Path = OPTIMIZATION_TRIAL_LEDGER_PARQUET,
    trial_ledger_csv_path: Path = OPTIMIZATION_TRIAL_LEDGER_CSV,
    overfitting_report_path: Path = OPTIMIZATION_OVERFITTING_REPORT_JSON,
) -> None:
    trials_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    trials_csv_path.parent.mkdir(parents=True, exist_ok=True)
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    trial_ledger_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    trial_ledger_csv_path.parent.mkdir(parents=True, exist_ok=True)
    overfitting_report_path.parent.mkdir(parents=True, exist_ok=True)
    trials_frame.to_parquet(trials_parquet_path, index=False)
    trials_frame.to_csv(trials_csv_path, index=False)
    trials_frame.to_parquet(trial_ledger_parquet_path, index=False)
    trials_frame.to_csv(trial_ledger_csv_path, index=False)
    best_params_path.write_text(
        json.dumps(best_params, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if overfitting_report is not None:
        overfitting_report_path.write_text(
            json.dumps(overfitting_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    LOGGER.info(
        "Saved optimization outputs: %s, %s, %s, %s, %s, %s",
        trials_parquet_path,
        trials_csv_path,
        best_params_path,
        trial_ledger_parquet_path,
        trial_ledger_csv_path,
        overfitting_report_path,
    )
