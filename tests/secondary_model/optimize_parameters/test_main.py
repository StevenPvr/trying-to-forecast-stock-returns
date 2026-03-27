from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.secondary_model.data.targets import SECONDARY_TARGET_SPECS
from core.src.secondary_model.optimize_parameters import main as secondary_optimize_main


class TestRunSecondaryOptimizeParameters:
    def test_routes_each_target_to_its_preprocessed_dataset_and_target_outputs(
        self,
        monkeypatch,
    ) -> None:
        captured_calls: list[dict[str, Any]] = []

        def _fake_optimize_xgboost_parameters(
            dataset_path: Path,
            optimization_config: Any = None,
            *,
            study_name: str,
            trials_parquet_path: Path,
            trials_csv_path: Path,
            best_params_path: Path,
        ) -> tuple[pd.DataFrame, dict[str, Any]]:
            del optimization_config
            captured_calls.append({
                "dataset_path": dataset_path,
                "study_name": study_name,
                "trials_parquet_path": trials_parquet_path,
                "trials_csv_path": trials_csv_path,
                "best_params_path": best_params_path,
            })
            return pd.DataFrame(), {}

        monkeypatch.setattr(
            secondary_optimize_main,
            "optimize_xgboost_parameters",
            _fake_optimize_xgboost_parameters,
        )

        outputs = secondary_optimize_main.run_secondary_optimize_parameters()

        expected_target_names = [target_spec.name for target_spec in SECONDARY_TARGET_SPECS]
        assert list(outputs.keys()) == expected_target_names
        assert [call["dataset_path"].parent.name for call in captured_calls] == expected_target_names
        assert [call["study_name"] for call in captured_calls] == [
            f"xgboost_walk_forward_rmse_{target_name}" for target_name in expected_target_names
        ]
        assert [call["trials_parquet_path"].parent.name for call in captured_calls] == expected_target_names
        assert [call["trials_csv_path"].parent.name for call in captured_calls] == expected_target_names
        assert [call["best_params_path"].parent.name for call in captured_calls] == expected_target_names


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
