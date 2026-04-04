"""Tests for yfinance download serialization lock."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from core.src.meta_model.data.data_fetching.yfinance_download_lock import YFINANCE_DOWNLOAD_LOCK


def test_yfinance_lock_is_reusable_mutex() -> None:
    prototype = threading.Lock()
    assert type(YFINANCE_DOWNLOAD_LOCK) is type(prototype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
