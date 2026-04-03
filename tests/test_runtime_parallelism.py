from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.runtime_parallelism import (
    resolve_available_cpu_count,
    resolve_executor_worker_count,
)


def test_resolve_available_cpu_count_uses_all_detected_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.src.meta_model.runtime_parallelism.os.cpu_count", lambda: 12)

    assert resolve_available_cpu_count() == 12


def test_resolve_executor_worker_count_caps_at_task_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.src.meta_model.runtime_parallelism.os.cpu_count", lambda: 12)

    assert resolve_executor_worker_count(task_count=3) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
