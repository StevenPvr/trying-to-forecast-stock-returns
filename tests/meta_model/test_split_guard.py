from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.split_guard import assert_train_only_fit_frame


def test_assert_train_only_fit_frame_accepts_train_only() -> None:
    frame = pd.DataFrame({"dataset_split": ["train", "train"]})
    assert_train_only_fit_frame(frame, context="unit-test")


def test_assert_train_only_fit_frame_rejects_non_train() -> None:
    frame = pd.DataFrame({"dataset_split": ["train", "val"]})
    with pytest.raises(ValueError, match="train-only"):
        assert_train_only_fit_frame(frame, context="unit-test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
