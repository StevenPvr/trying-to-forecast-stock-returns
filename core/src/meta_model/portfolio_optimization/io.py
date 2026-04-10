from __future__ import annotations

"""I/O utilities for portfolio optimisation artifacts."""

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from core.src.meta_model.data.paths import (
    DATA_PORTFOLIO_OPTIMIZATION_DIR,
    PORTFOLIO_ALPHA_CALIBRATION_AUDIT_PARQUET,
    PORTFOLIO_BEST_PARAMS_JSON,
    PORTFOLIO_RISK_COVARIANCE_PARQUET,
    PORTFOLIO_TRAIN_CV_ALLOCATIONS_PARQUET,
    PORTFOLIO_TRAIN_CV_DAILY_PARQUET,
    PORTFOLIO_TRIAL_LEDGER_PARQUET,
    PORTFOLIO_VALIDATION_ALLOCATIONS_PARQUET,
    PORTFOLIO_VALIDATION_DAILY_PARQUET,
    PORTFOLIO_VALIDATION_SUMMARY_JSON,
)


def save_portfolio_optimization_outputs(
    *,
    best_params: Mapping[str, object],
    trial_ledger: pd.DataFrame,
    train_cv_daily: pd.DataFrame,
    train_cv_allocations: pd.DataFrame,
    validation_daily: pd.DataFrame,
    validation_allocations: pd.DataFrame,
    validation_summary: Mapping[str, object],
    alpha_calibration_audit: pd.DataFrame,
    covariance_frame: pd.DataFrame,
) -> None:
    DATA_PORTFOLIO_OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_BEST_PARAMS_JSON.write_text(
        json.dumps(best_params, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    trial_ledger.to_parquet(PORTFOLIO_TRIAL_LEDGER_PARQUET, index=False)
    train_cv_daily.to_parquet(PORTFOLIO_TRAIN_CV_DAILY_PARQUET, index=False)
    train_cv_allocations.to_parquet(PORTFOLIO_TRAIN_CV_ALLOCATIONS_PARQUET, index=False)
    validation_daily.to_parquet(PORTFOLIO_VALIDATION_DAILY_PARQUET, index=False)
    validation_allocations.to_parquet(PORTFOLIO_VALIDATION_ALLOCATIONS_PARQUET, index=False)
    PORTFOLIO_VALIDATION_SUMMARY_JSON.write_text(
        json.dumps(validation_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    alpha_calibration_audit.to_parquet(PORTFOLIO_ALPHA_CALIBRATION_AUDIT_PARQUET, index=False)
    covariance_frame.to_parquet(PORTFOLIO_RISK_COVARIANCE_PARQUET, index=True)


def load_portfolio_best_params(path: Path = PORTFOLIO_BEST_PARAMS_JSON) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Portfolio best params not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
