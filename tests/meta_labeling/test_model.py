from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.meta_labeling import model as meta_model


def _train_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    rows: list[dict[str, object]] = []
    for date_index, date_value in enumerate(dates, start=1):
        for ticker, label_value in [("AAA", 1.0), ("BBB", 0.0)]:
            rows.append({
                "date": date_value,
                "ticker": ticker,
                "feature_a": float(date_index),
                "feature_b": float(date_index * 2),
                "meta_label": float(label_value),
            })
    return pd.DataFrame(rows)


def test_split_early_stopping_frame_uses_trailing_dates() -> None:
    train_frame = _train_frame()

    fit_frame, eval_frame = meta_model._split_early_stopping_frame(
        train_frame,
        validation_fraction=0.25,
    )

    assert eval_frame is not None
    assert pd.Timestamp(fit_frame["date"].max()) < pd.Timestamp(eval_frame["date"].min())
    assert eval_frame["date"].drop_duplicates().tolist() == [
        pd.Timestamp("2024-01-09"),
        pd.Timestamp("2024-01-10"),
    ]


def test_fit_meta_model_uses_early_stopping_and_refits_full_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_frame = _train_frame()
    train_calls: list[dict[str, object]] = []

    class _FakeBooster:
        def __init__(self, *, best_iteration: int | None = None) -> None:
            self.best_iteration = best_iteration

    class _FakeXGBoostModule:
        def train(
            self,
            *,
            params: dict[str, object],
            dtrain: object,
            num_boost_round: int,
            verbose_eval: bool,
            evals: list[tuple[object, str]] | None = None,
            early_stopping_rounds: int | None = None,
        ) -> _FakeBooster:
            del params, verbose_eval
            train_calls.append({
                "dtrain": dtrain,
                "num_boost_round": num_boost_round,
                "evals": evals,
                "early_stopping_rounds": early_stopping_rounds,
            })
            if early_stopping_rounds is not None:
                return _FakeBooster(best_iteration=12)
            return _FakeBooster()

    def fake_prepare_xgboost_feature_frame(
        frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        return pd.DataFrame(frame.loc[:, feature_columns].copy())

    def fake_build_xgboost_dmatrix(
        xgb: object,
        feature_frame: pd.DataFrame,
        label: object,
    ) -> object:
        del xgb
        return {"rows": len(feature_frame), "label": label}

    monkeypatch.setattr(meta_model, "load_xgboost_module", lambda: _FakeXGBoostModule())
    monkeypatch.setattr(meta_model, "prepare_xgboost_feature_frame", fake_prepare_xgboost_feature_frame)
    monkeypatch.setattr(meta_model, "build_xgboost_dmatrix", fake_build_xgboost_dmatrix)

    artifact = meta_model.fit_meta_model(
        train_frame,
        ["feature_a", "feature_b"],
        label_column="meta_label",
        params={"objective": "binary:logistic"},
        training_rounds=300,
        early_stopping_rounds=50,
        early_stopping_validation_fraction=0.25,
        minimum_training_rounds=5,
    )

    assert len(train_calls) == 2
    assert train_calls[0]["early_stopping_rounds"] == 50
    assert train_calls[0]["evals"] is not None
    assert train_calls[1]["early_stopping_rounds"] is None
    assert train_calls[1]["evals"] is None
    assert train_calls[1]["num_boost_round"] == 13
    assert artifact.training_rounds == 13
    assert artifact.training_metadata["selected_training_rounds"] == 13


def test_fit_meta_model_enforces_minimum_training_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_frame = _train_frame()

    class _FakeBooster:
        def __init__(self, *, best_iteration: int | None = None) -> None:
            self.best_iteration = best_iteration

    class _FakeXGBoostModule:
        def train(
            self,
            *,
            params: dict[str, object],
            dtrain: object,
            num_boost_round: int,
            verbose_eval: bool,
            evals: list[tuple[object, str]] | None = None,
            early_stopping_rounds: int | None = None,
        ) -> _FakeBooster:
            del params, dtrain, num_boost_round, verbose_eval, evals
            if early_stopping_rounds is not None:
                return _FakeBooster(best_iteration=0)
            return _FakeBooster()

    monkeypatch.setattr(meta_model, "load_xgboost_module", lambda: _FakeXGBoostModule())
    monkeypatch.setattr(
        meta_model,
        "prepare_xgboost_feature_frame",
        lambda frame, feature_columns: pd.DataFrame(frame.loc[:, feature_columns].copy()),
    )
    monkeypatch.setattr(
        meta_model,
        "build_xgboost_dmatrix",
        lambda xgb, feature_frame, label: {"rows": len(feature_frame), "label": label},
    )

    artifact = meta_model.fit_meta_model(
        train_frame,
        ["feature_a", "feature_b"],
        label_column="meta_label",
        params={"objective": "binary:logistic"},
        training_rounds=300,
        early_stopping_rounds=50,
        early_stopping_validation_fraction=0.25,
        minimum_training_rounds=25,
    )

    assert artifact.training_rounds == 25
    assert artifact.training_metadata["selected_training_rounds"] == 25
    assert artifact.training_metadata["minimum_training_rounds"] == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
