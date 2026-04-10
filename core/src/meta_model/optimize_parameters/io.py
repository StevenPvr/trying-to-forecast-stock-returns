from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, TypedDict

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


class OverfittingReportPayload(TypedDict):
    trial_count: int
    pbo: float
    selected_trial_number: int
    selected_trial_objective_score: float


SaveOutputsFn = Callable[[pd.DataFrame, dict[str, Any], OverfittingReportPayload], None]


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


def _save_outputs_with_default_paths(
    trials_frame: pd.DataFrame,
    best_payload: dict[str, Any],
    overfitting_report: OverfittingReportPayload,
) -> None:
    serialized_overfitting_report: dict[str, Any] = dict(overfitting_report)
    save_optimization_outputs(
        trials_frame,
        best_payload,
        serialized_overfitting_report,
    )


def _save_outputs_with_custom_paths(
    trials_frame: pd.DataFrame,
    best_payload: dict[str, Any],
    overfitting_report: OverfittingReportPayload,
    *,
    trials_parquet_path: Path,
    trials_csv_path: Path,
    best_params_path: Path,
) -> None:
    serialized_overfitting_report: dict[str, Any] = dict(overfitting_report)
    save_optimization_outputs(
        trials_frame,
        best_payload,
        serialized_overfitting_report,
        trials_parquet_path=trials_parquet_path,
        trials_csv_path=trials_csv_path,
        best_params_path=best_params_path,
        trial_ledger_parquet_path=OPTIMIZATION_TRIAL_LEDGER_PARQUET,
        trial_ledger_csv_path=OPTIMIZATION_TRIAL_LEDGER_CSV,
        overfitting_report_path=OPTIMIZATION_OVERFITTING_REPORT_JSON,
    )


def build_default_save_outputs_fn(
    *,
    trials_parquet_path: Path | None,
    trials_csv_path: Path | None,
    best_params_path: Path | None,
) -> SaveOutputsFn:
    def save_outputs(
        trials_frame: pd.DataFrame,
        best_payload: dict[str, Any],
        overfitting_report: OverfittingReportPayload,
    ) -> None:
        if (
            trials_parquet_path is None
            and trials_csv_path is None
            and best_params_path is None
        ):
            _save_outputs_with_default_paths(trials_frame, best_payload, overfitting_report)
            return
        _save_outputs_with_custom_paths(
            trials_frame,
            best_payload,
            overfitting_report,
            trials_parquet_path=trials_parquet_path or XGBOOST_OPTUNA_TRIALS_PARQUET,
            trials_csv_path=trials_csv_path or XGBOOST_OPTUNA_TRIALS_CSV,
            best_params_path=best_params_path or XGBOOST_OPTUNA_BEST_PARAMS_JSON,
        )

    return save_outputs


def resolve_save_outputs_fn(
    save_outputs_fn: SaveOutputsFn | None,
    *,
    trials_parquet_path: Path | None,
    trials_csv_path: Path | None,
    best_params_path: Path | None,
) -> SaveOutputsFn:
    if save_outputs_fn is not None:
        return save_outputs_fn
    return build_default_save_outputs_fn(
        trials_parquet_path=trials_parquet_path,
        trials_csv_path=trials_csv_path,
        best_params_path=best_params_path,
    )
