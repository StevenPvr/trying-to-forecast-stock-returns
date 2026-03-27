from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.optimize_parameters.config import OptimizationConfig, OPTUNA_STUDY_NAME
from core.src.meta_model.optimize_parameters.main import optimize_xgboost_parameters
from core.src.secondary_model.data.paths import (
    build_secondary_best_params_json,
    build_secondary_optimize_parameters_target_dir,
    build_secondary_optuna_trials_csv,
    build_secondary_optuna_trials_parquet,
    build_secondary_preprocessing_target_dir,
)
from core.src.secondary_model.data.targets import SECONDARY_TARGET_SPECS

LOGGER: logging.Logger = logging.getLogger(__name__)


def build_secondary_preprocessed_dataset_path(target_name: str) -> Path:
    return build_secondary_preprocessing_target_dir(target_name) / "dataset_preprocessed.parquet"


def build_secondary_study_name(target_name: str) -> str:
    return f"{OPTUNA_STUDY_NAME}_{target_name}"


def run_secondary_optimize_parameters(
    optimization_config: OptimizationConfig | None = None,
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for target_spec in SECONDARY_TARGET_SPECS:
        target_dir = build_secondary_optimize_parameters_target_dir(target_spec.name)
        target_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = build_secondary_preprocessed_dataset_path(target_spec.name)
        LOGGER.info(
            "Starting secondary hyperparameter optimization for %s from %s",
            target_spec.name,
            dataset_path,
        )
        trials_frame, best_payload = optimize_xgboost_parameters(
            dataset_path,
            optimization_config,
            study_name=build_secondary_study_name(target_spec.name),
            trials_parquet_path=build_secondary_optuna_trials_parquet(target_spec.name),
            trials_csv_path=build_secondary_optuna_trials_csv(target_spec.name),
            best_params_path=build_secondary_best_params_json(target_spec.name),
        )
        outputs[target_spec.name] = {
            "trials_frame": trials_frame,
            "best_payload": best_payload,
        }
        LOGGER.info(
            "Completed secondary hyperparameter optimization for %s",
            target_spec.name,
        )
    return outputs


run_secondary_xgboost_parameter_optimization = run_secondary_optimize_parameters


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_secondary_optimize_parameters()
    LOGGER.info("Secondary optimize_parameters pipeline completed.")


__all__ = [
    "build_secondary_preprocessed_dataset_path",
    "build_secondary_study_name",
    "main",
    "run_secondary_optimize_parameters",
    "run_secondary_xgboost_parameter_optimization",
]


if __name__ == "__main__":
    main()
