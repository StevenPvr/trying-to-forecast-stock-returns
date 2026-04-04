from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.feature_selection.selection_pipeline import build_final_candidate_feature_names


def test_build_final_candidate_feature_names_rescues_broker_features() -> None:
    sfi_scores = pd.DataFrame(
        {
            "feature_name": [
                "quant_alpha",
                "xtb_spread_bps",
                "xtb_expected_intraday_cost_rate_lag_21d",
                "xtb_spread_to_gap_abs_lag_21d",
            ],
            "feature_family": [
                "quant",
                "broker",
                "broker",
                "broker",
            ],
            "objective_score": [0.5, 0.0, 0.01, 0.02],
            "daily_rank_ic_mean": [0.5, 0.0, 0.01, 0.02],
            "coverage_fraction": [1.0, 1.0, 1.0, 1.0],
            "passes_coverage": [True, True, True, True],
        },
    )

    candidate_names = build_final_candidate_feature_names(
        sfi_scores,
        target_survivors=["quant_alpha", "xtb_spread_to_gap_abs_lag_21d"],
    )

    assert candidate_names == [
        "quant_alpha",
        "xtb_spread_to_gap_abs_lag_21d",
        "xtb_expected_intraday_cost_rate_lag_21d",
        "xtb_spread_bps",
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
