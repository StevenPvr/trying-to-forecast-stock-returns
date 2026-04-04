from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from _pytest.monkeypatch import MonkeyPatch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.optimize_parameters.acceleration import (  # noqa: E402
    normalize_requested_accelerator,
    resolve_acceleration_plan,
)
from core.src.meta_model.optimize_parameters.config import OptimizationConfig  # noqa: E402


class _FakeXGBoostModule:
    pass


class TestNormalizeRequestedAccelerator:
    def test_accepts_supported_values(self) -> None:
        assert normalize_requested_accelerator("auto") == "auto"
        assert normalize_requested_accelerator("CPU") == "cpu"
        assert normalize_requested_accelerator(" cuda ") == "cuda"

    def test_raises_for_unsupported_value(self) -> None:
        with pytest.raises(ValueError, match="Unsupported compute accelerator"):
            normalize_requested_accelerator("metal")


class TestResolveAccelerationPlan:
    def test_respects_forced_cpu_mode(self) -> None:
        plan = resolve_acceleration_plan(
            OptimizationConfig(compute_accelerator="cpu"),
            xgb_module=_FakeXGBoostModule(),
        )

        assert plan.accelerator == "cpu"
        assert plan.use_gpu_matrix_cache is False

    def test_auto_falls_back_to_cpu_when_cuda_build_is_missing(self, monkeypatch: MonkeyPatch) -> None:
        def _cuda_build_flag(_: Any) -> bool:
            return False

        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.acceleration._extract_cuda_build_flag",
            _cuda_build_flag,
        )

        plan = resolve_acceleration_plan(
            OptimizationConfig(compute_accelerator="auto"),
            xgb_module=_FakeXGBoostModule(),
        )

        assert plan.accelerator == "cpu"

    def test_auto_uses_cuda_when_probe_succeeds(self, monkeypatch: MonkeyPatch) -> None:
        def _cuda_build_flag(_: Any) -> bool:
            return True

        def _cuda_probe(_: Any, __: int) -> tuple[bool, str | None]:
            return True, None

        def _gpu_name(_: int) -> str:
            return "NVIDIA L40S"

        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.acceleration._extract_cuda_build_flag",
            _cuda_build_flag,
        )
        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.acceleration._probe_cuda_training",
            _cuda_probe,
        )
        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.acceleration._detect_nvidia_gpu_name",
            _gpu_name,
        )

        plan = resolve_acceleration_plan(
            OptimizationConfig(compute_accelerator="auto", gpu_device_id=0),
            xgb_module=_FakeXGBoostModule(),
        )

        assert plan.accelerator == "cuda"
        assert plan.gpu_name == "NVIDIA L40S"
        assert plan.use_gpu_matrix_cache is True

    def test_forced_cuda_raises_when_probe_fails(self, monkeypatch: MonkeyPatch) -> None:
        def _cuda_build_flag(_: Any) -> bool:
            return True

        def _cuda_probe(_: Any, __: int) -> tuple[bool, str | None]:
            return False, "CUDA init failed"

        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.acceleration._extract_cuda_build_flag",
            _cuda_build_flag,
        )
        monkeypatch.setattr(
            "core.src.meta_model.optimize_parameters.acceleration._probe_cuda_training",
            _cuda_probe,
        )

        with pytest.raises(RuntimeError, match="runtime CUDA probe failed"):
            resolve_acceleration_plan(
                OptimizationConfig(compute_accelerator="cuda"),
                xgb_module=_FakeXGBoostModule(),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
