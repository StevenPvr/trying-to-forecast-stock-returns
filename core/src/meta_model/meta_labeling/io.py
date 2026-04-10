from __future__ import annotations

"""Artifact I/O for the meta-labeling stage."""

import json
import logging
from typing import Mapping

import pandas as pd

from core.src.meta_model.data.paths import (
    DATA_META_LABELING_DIR,
    META_BEST_PARAMS_JSON,
    META_MODEL_JSON,
    META_PRIMARY_OOS_TRAIN_TAIL_PARQUET,
    META_PRIMARY_OOS_VAL_PARQUET,
    META_STAGE_SUMMARY_JSON,
    META_TRAIN_OOF_PREDICTIONS_PARQUET,
    META_TRIAL_LEDGER_PARQUET,
    META_VAL_PREDICTIONS_PARQUET,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def save_meta_labeling_outputs(
    *,
    primary_oos_train_tail: pd.DataFrame,
    primary_oos_val: pd.DataFrame,
    meta_train_oof_predictions: pd.DataFrame,
    meta_val_predictions: pd.DataFrame,
    meta_trial_ledger: pd.DataFrame,
    meta_best_params: Mapping[str, object],
    meta_model_payload: Mapping[str, object],
    meta_stage_summary: Mapping[str, object],
) -> None:
    DATA_META_LABELING_DIR.mkdir(parents=True, exist_ok=True)
    primary_oos_train_tail.to_parquet(META_PRIMARY_OOS_TRAIN_TAIL_PARQUET, index=False)
    primary_oos_val.to_parquet(META_PRIMARY_OOS_VAL_PARQUET, index=False)
    meta_train_oof_predictions.to_parquet(META_TRAIN_OOF_PREDICTIONS_PARQUET, index=False)
    meta_val_predictions.to_parquet(META_VAL_PREDICTIONS_PARQUET, index=False)
    meta_trial_ledger.to_parquet(META_TRIAL_LEDGER_PARQUET, index=False)
    META_BEST_PARAMS_JSON.write_text(
        json.dumps(dict(meta_best_params), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    META_MODEL_JSON.write_text(
        json.dumps(dict(meta_model_payload), sort_keys=True),
        encoding="utf-8",
    )
    META_STAGE_SUMMARY_JSON.write_text(
        json.dumps(dict(meta_stage_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    LOGGER.info(
        "Meta-labeling outputs saved: dir=%s | parquets=[train_tail=%d, val=%d, oof=%d, val_pred=%d, ledger=%d]",
        DATA_META_LABELING_DIR,
        len(primary_oos_train_tail),
        len(primary_oos_val),
        len(meta_train_oof_predictions),
        len(meta_val_predictions),
        len(meta_trial_ledger),
    )


def load_meta_best_params() -> dict[str, object]:
    LOGGER.info("Loading meta best params from %s", META_BEST_PARAMS_JSON)
    return json.loads(META_BEST_PARAMS_JSON.read_text(encoding="utf-8"))


def load_meta_model_payload() -> dict[str, object]:
    LOGGER.info("Loading meta model payload from %s", META_MODEL_JSON)
    return json.loads(META_MODEL_JSON.read_text(encoding="utf-8"))
