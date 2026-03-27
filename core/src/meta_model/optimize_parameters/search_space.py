from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Protocol


class TrialProtocol(Protocol):
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float: ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
    ) -> int: ...


def load_xgboost_module() -> Any:
    spec = importlib.util.find_spec("xgboost")
    if spec is None:
        raise ImportError(
            "xgboost is not installed. Add it to the environment before running optimize_parameters.",
        )
    return importlib.import_module("xgboost")


def load_optuna_module() -> Any:
    spec = importlib.util.find_spec("optuna")
    if spec is None:
        raise ImportError(
            "optuna is not installed. Add it to the environment before running optimize_parameters.",
        )
    return importlib.import_module("optuna")


def suggest_xgboost_params(
    trial: TrialProtocol,
    *,
    threads_per_fold: int,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": random_seed,
        "verbosity": 0,
        "nthread": threads_per_fold,
        "eta": trial.suggest_float("eta", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 64.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 0.90),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 0.80),
        "gamma": trial.suggest_float("gamma", 1e-3, 20.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-2, 100.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 25.0, log=True),
        "max_bin": trial.suggest_int("max_bin", 64, 256),
    }
