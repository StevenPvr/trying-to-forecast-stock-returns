from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.portfolio_optimization.alpha_calibration import fit_alpha_calibrator_train_only
from core.src.meta_model.portfolio_optimization.risk_model import fit_train_only_covariance_model


def _base_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    rows: list[dict[str, object]] = []
    for idx, date in enumerate(dates):
        split = "train" if idx < 60 else "val"
        for ticker, coef in [("AAA", 0.01), ("BBB", -0.01)]:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "dataset_split": split,
                    "prediction": float(np.sin(idx) * coef + coef),
                    "target_week_hold_5sessions_close_log_return": float(np.cos(idx) * coef),
                    "stock_close_log_return_lag_1d": float(np.sin(idx / 2.0) * coef),
                },
            )
    return pd.DataFrame(rows)


def test_alpha_calibration_rejects_non_train_rows() -> None:
    frame = _base_frame()
    with pytest.raises(ValueError, match="train-only"):
        fit_alpha_calibrator_train_only(frame)


def test_risk_model_rejects_non_train_rows() -> None:
    frame = _base_frame()
    with pytest.raises(ValueError, match="train-only"):
        fit_train_only_covariance_model(frame)


def test_risk_model_uses_latest_sufficiently_covered_dates() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    rows: list[dict[str, object]] = []
    tickers = ["AAA", "BBB", "CCC"]
    for idx, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            value = float(np.sin((idx + 1) * (ticker_index + 1) / 10.0))
            if idx >= 70 and ticker != "AAA":
                value = np.nan
            rows.append({
                "date": date,
                "ticker": ticker,
                "dataset_split": "train",
                "stock_close_log_return_lag_1d": value,
            })
    frame = pd.DataFrame(rows)
    fitted = fit_train_only_covariance_model(
        frame,
        lookback_days=20,
        min_history_days=10,
    )
    assert fitted.covariance.shape[0] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
