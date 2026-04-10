from __future__ import annotations

"""Train-only alpha calibration utilities."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from core.src.meta_model.model_contract import (
    PREDICTION_COLUMN,
    REALIZED_RETURN_COLUMN,
    SPLIT_COLUMN,
    TRAIN_SPLIT_NAME,
)
from core.src.meta_model.split_guard import assert_train_only_fit_frame


@dataclass(frozen=True)
class FittedAlphaCalibrator:
    x_thresholds: list[float]
    y_thresholds: list[float]

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if len(self.x_thresholds) == 0:
            return np.zeros(len(raw_scores), dtype=np.float64)
        x = np.asarray(raw_scores, dtype=np.float64)
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
    def from_json_payload(payload: dict[str, object]) -> "FittedAlphaCalibrator":
        return FittedAlphaCalibrator(
            x_thresholds=[float(v) for v in payload.get("x_thresholds", [])],  # type: ignore[arg-type]
            y_thresholds=[float(v) for v in payload.get("y_thresholds", [])],  # type: ignore[arg-type]
        )


def fit_alpha_calibrator_train_only(
    oof_train_predictions: pd.DataFrame,
) -> tuple[FittedAlphaCalibrator, pd.DataFrame]:
    assert_train_only_fit_frame(
        oof_train_predictions,
        split_column=SPLIT_COLUMN,
        train_split_name=TRAIN_SPLIT_NAME,
        context="portfolio_optimization.alpha_calibration.fit",
    )
    raw_scores = pd.to_numeric(oof_train_predictions[PREDICTION_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    realized_log_return = pd.to_numeric(
        oof_train_predictions[REALIZED_RETURN_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    realized_simple_return = np.exp(realized_log_return) - 1.0
    finite_mask = np.isfinite(raw_scores) & np.isfinite(realized_simple_return)
    if int(finite_mask.sum()) < 10:
        calibrator = FittedAlphaCalibrator(x_thresholds=[0.0], y_thresholds=[0.0])
        audit = oof_train_predictions.loc[:, [SPLIT_COLUMN, PREDICTION_COLUMN, REALIZED_RETURN_COLUMN]].copy()
        audit["expected_return_5d"] = 0.0
        return calibrator, audit
    model = IsotonicRegression(y_min=-0.95, y_max=2.0, increasing=True, out_of_bounds="clip")
    x_fit = raw_scores[finite_mask]
    y_fit = realized_simple_return[finite_mask]
    model.fit(x_fit, y_fit)
    x_thresholds = np.asarray(model.X_thresholds_, dtype=np.float64).tolist()
    y_thresholds = np.asarray(model.y_thresholds_, dtype=np.float64).tolist()
    calibrator = FittedAlphaCalibrator(x_thresholds=x_thresholds, y_thresholds=y_thresholds)
    expected = calibrator.transform(raw_scores)
    audit = oof_train_predictions.loc[:, [SPLIT_COLUMN, PREDICTION_COLUMN, REALIZED_RETURN_COLUMN]].copy()
    audit["expected_return_5d"] = expected
    return calibrator, audit


def serialize_alpha_calibrator(calibrator: FittedAlphaCalibrator) -> str:
    return json.dumps(calibrator.to_json_payload(), sort_keys=True)


def deserialize_alpha_calibrator(payload: str) -> FittedAlphaCalibrator:
    return FittedAlphaCalibrator.from_json_payload(json.loads(payload))
