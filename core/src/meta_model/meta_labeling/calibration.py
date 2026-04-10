from __future__ import annotations

"""Probability calibration utilities for the meta model."""

import logging
from dataclasses import dataclass
import json
from typing import cast

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from core.src.meta_model.model_contract import SPLIT_COLUMN, TRAIN_SPLIT_NAME
from core.src.meta_model.split_guard import assert_train_only_fit_frame

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FittedProbabilityCalibrator:
    x_thresholds: list[float]
    y_thresholds: list[float]

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        x = np.asarray(raw_scores, dtype=np.float64)
        if len(self.x_thresholds) == 0:
            return np.zeros(len(x), dtype=np.float64)
        return np.interp(
            x,
            np.asarray(self.x_thresholds, dtype=np.float64),
            np.asarray(self.y_thresholds, dtype=np.float64),
            left=float(self.y_thresholds[0]),
            right=float(self.y_thresholds[-1]),
        )

    def to_json_payload(self) -> dict[str, object]:
        return {
            "x_thresholds": self.x_thresholds,
            "y_thresholds": self.y_thresholds,
        }

    @staticmethod
    def from_json_payload(payload: dict[str, object]) -> "FittedProbabilityCalibrator":
        raw_x = cast(list[object], payload.get("x_thresholds", []))
        raw_y = cast(list[object], payload.get("y_thresholds", []))
        return FittedProbabilityCalibrator(
            x_thresholds=[float(value) for value in raw_x],
            y_thresholds=[float(value) for value in raw_y],
        )


def fit_probability_calibrator_train_only(
    train_predictions: pd.DataFrame,
    *,
    probability_column: str,
    label_column: str,
) -> FittedProbabilityCalibrator:
    assert_train_only_fit_frame(
        train_predictions,
        split_column=SPLIT_COLUMN,
        train_split_name=TRAIN_SPLIT_NAME,
        context="meta_labeling.probability_calibration.fit",
    )
    x = pd.to_numeric(train_predictions[probability_column], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(train_predictions[label_column], errors="coerce").to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    finite_count: int = int(finite_mask.sum())
    if finite_count < 10:
        LOGGER.warning(
            "Probability calibrator: insufficient finite samples (%d < 10), falling back to identity",
            finite_count,
        )
        return FittedProbabilityCalibrator(x_thresholds=[0.0, 1.0], y_thresholds=[0.0, 1.0])
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(x[finite_mask], y[finite_mask])
    thresholds = np.asarray(model.X_thresholds_, dtype=np.float64).tolist()
    LOGGER.info(
        "Probability calibrator fitted: finite_samples=%d / %d | thresholds=%d",
        finite_count,
        len(x),
        len(thresholds),
    )
    return FittedProbabilityCalibrator(
        x_thresholds=thresholds,
        y_thresholds=np.asarray(model.y_thresholds_, dtype=np.float64).tolist(),
    )


def serialize_probability_calibrator(calibrator: FittedProbabilityCalibrator) -> str:
    return json.dumps(calibrator.to_json_payload(), sort_keys=True)


def deserialize_probability_calibrator(payload: str) -> FittedProbabilityCalibrator:
    return FittedProbabilityCalibrator.from_json_payload(json.loads(payload))
