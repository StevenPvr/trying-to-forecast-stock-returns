from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.model_contract import MODEL_TARGET_COLUMN
from core.src.meta_model.model_registry.main import (
    ModelSpec,
    build_default_model_specs,
    fit_model,
    predict_model,
)


def test_build_default_model_specs_includes_core_models() -> None:
    specs = build_default_model_specs(
        xgboost_params={"eta": 0.05, "max_depth": 3},
        xgboost_training_rounds=120,
    )

    assert [spec.model_name for spec in specs][:4] == [
        "ridge",
        "elastic_net",
        "factor_composite",
        "xgboost",
    ]


def test_build_default_model_specs_includes_lightgbm_when_installed() -> None:
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm is not installed in this environment")

    specs = build_default_model_specs(
        xgboost_params={"eta": 0.05, "max_depth": 3},
        xgboost_training_rounds=120,
    )

    assert "lightgbm" in [spec.model_name for spec in specs]


def test_fit_model_ridge_learns_simple_linear_signal() -> None:
    train_frame = pd.DataFrame({
        "date": pd.date_range("2022-01-03", periods=8, freq="B"),
        "ticker": [f"T{i}" for i in range(8)],
        "feature_a": np.arange(8, dtype=float),
        "feature_b": np.arange(8, dtype=float) * 0.5,
        MODEL_TARGET_COLUMN: np.arange(8, dtype=float) * 2.0,
    })
    test_frame = pd.DataFrame({
        "date": pd.date_range("2022-01-17", periods=2, freq="B"),
        "ticker": ["AAA", "BBB"],
        "feature_a": [8.0, 9.0],
        "feature_b": [4.0, 4.5],
    })

    artifact = fit_model(
        ModelSpec(model_name="ridge", params={"alpha": 1e-6}),
        train_frame,
        ["feature_a", "feature_b"],
    )
    predictions = predict_model(artifact, test_frame, ["feature_a", "feature_b"])

    assert artifact.model_name == "ridge"
    assert predictions.shape == (2,)
    assert float(predictions[1]) > float(predictions[0]) > 10.0


def test_factor_composite_prefers_momentum_reversal_small_size_and_liquidity() -> None:
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2022-01-03")] * 3,
        "ticker": ["AAA", "BBB", "CCC"],
        "quant_momentum_21d": [3.0, 2.0, 1.0],
        "quant_reversal_5d": [3.0, 2.0, 1.0],
        "quant_volatility_21d": [1.0, 2.0, 3.0],
        "company_market_cap_usd": [10.0, 20.0, 30.0],
        "stock_trading_volume": [30_000.0, 20_000.0, 10_000.0],
    })

    artifact = fit_model(
        ModelSpec(model_name="factor_composite"),
        frame,
        [
            "quant_momentum_21d",
            "quant_reversal_5d",
            "quant_volatility_21d",
            "company_market_cap_usd",
            "stock_trading_volume",
        ],
    )
    predictions = predict_model(
        artifact,
        frame,
        [
            "quant_momentum_21d",
            "quant_reversal_5d",
            "quant_volatility_21d",
            "company_market_cap_usd",
            "stock_trading_volume",
        ],
    )

    ranked_tickers = frame.assign(prediction=predictions).sort_values(
        "prediction",
        ascending=False,
    )["ticker"].tolist()
    assert ranked_tickers == ["AAA", "BBB", "CCC"]


def test_elastic_net_learns_monotonic_signal() -> None:
    train_frame = pd.DataFrame({
        "date": pd.date_range("2022-01-03", periods=10, freq="B"),
        "ticker": [f"T{i}" for i in range(10)],
        "feature_a": np.arange(10, dtype=float),
        "feature_b": np.arange(10, dtype=float) * 0.3,
        MODEL_TARGET_COLUMN: np.arange(10, dtype=float),
    })
    test_frame = pd.DataFrame({
        "date": pd.date_range("2022-01-31", periods=2, freq="B"),
        "ticker": ["AAA", "BBB"],
        "feature_a": [10.0, 11.0],
        "feature_b": [3.0, 3.3],
    })

    artifact = fit_model(
        ModelSpec(model_name="elastic_net"),
        train_frame,
        ["feature_a", "feature_b"],
    )
    predictions = predict_model(artifact, test_frame, ["feature_a", "feature_b"])

    assert artifact.model_name == "elastic_net"
    assert float(predictions[1]) > float(predictions[0])


def test_ridge_and_factor_composite_handle_missing_values() -> None:
    train_frame = pd.DataFrame({
        "date": pd.date_range("2022-01-03", periods=4, freq="B"),
        "ticker": ["AAA", "BBB", "CCC", "DDD"],
        "feature_a": [1.0, np.nan, 3.0, 4.0],
        "feature_b": [0.5, 1.0, np.nan, 2.0],
        "quant_momentum_21d": [3.0, np.nan, 1.0, 0.0],
        "stock_trading_volume": [30_000.0, 20_000.0, np.nan, 10_000.0],
        MODEL_TARGET_COLUMN: [1.0, 2.0, 3.0, 4.0],
    })
    test_frame = pd.DataFrame({
        "date": pd.date_range("2022-01-10", periods=2, freq="B"),
        "ticker": ["EEE", "FFF"],
        "feature_a": [5.0, np.nan],
        "feature_b": [2.5, 3.0],
        "quant_momentum_21d": [2.0, np.nan],
        "stock_trading_volume": [40_000.0, 15_000.0],
    })

    ridge_artifact = fit_model(
        ModelSpec(model_name="ridge", params={"alpha": 1e-6}),
        train_frame,
        ["feature_a", "feature_b"],
    )
    ridge_predictions = predict_model(ridge_artifact, test_frame, ["feature_a", "feature_b"])

    factor_artifact = fit_model(
        ModelSpec(model_name="factor_composite"),
        train_frame,
        ["quant_momentum_21d", "stock_trading_volume"],
    )
    factor_predictions = predict_model(
        factor_artifact,
        test_frame,
        ["quant_momentum_21d", "stock_trading_volume"],
    )

    assert np.isfinite(ridge_predictions).all()
    assert np.isfinite(factor_predictions).all()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
