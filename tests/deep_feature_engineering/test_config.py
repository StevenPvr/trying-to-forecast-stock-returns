from __future__ import annotations

import sys
from pathlib import Path

from unittest.mock import patch

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.paths import FEATURE_MDA_FILTERED_FEATURES_PARQUET
from core.src.deep_feature_engineering.config import DEFAULT_FEATURE_PARQUET
from core.src.deep_feature_engineering.main import run_deep_feature_engineering


def test_deep_feature_engineering_uses_base_mda_output_by_default() -> None:
    assert DEFAULT_FEATURE_PARQUET == FEATURE_MDA_FILTERED_FEATURES_PARQUET


def test_deep_feature_engineering_delegates_to_unified_feature_pipeline() -> None:
    cleaned = pd.DataFrame({"date": [], "ticker": []})
    featured = pd.DataFrame({"date": [], "ticker": [], "deep_feature": []})

    with (
        patch("core.src.deep_feature_engineering.main.load_cleaned_dataset", return_value=cleaned),
        patch("core.src.deep_feature_engineering.main.build_feature_dataset", return_value=featured),
        patch("core.src.deep_feature_engineering.main.save_lagged_feature_dataset") as save_mock,
    ):
        result = run_deep_feature_engineering()

    assert result is featured
    save_mock.assert_called_once()
