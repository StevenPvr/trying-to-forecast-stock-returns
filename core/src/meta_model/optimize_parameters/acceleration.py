from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from core.src.meta_model.optimize_parameters.config import OptimizationConfig


@dataclass(frozen=True)
class AccelerationPlan:
    requested_accelerator: str
    accelerator: str
    gpu_device_id: int
    gpu_name: str | None
    reason: str
    use_gpu_matrix_cache: bool


def normalize_requested_accelerator(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "cuda"}:
        return normalized
    raise ValueError(
        "Unsupported compute accelerator. Use one of: auto, cpu, cuda.",
    )


def _extract_cuda_build_flag(xgb_module: Any) -> bool | None:
    build_info_fn = getattr(xgb_module, "build_info", None)
    if not callable(build_info_fn):
        return None
    try:
        build_info = build_info_fn()
    except Exception:
        return None
    if not isinstance(build_info, dict):
        return None
    raw_flag = cast(object | None, build_info.get("USE_CUDA"))
    if raw_flag is None:
        raw_flag = cast(object | None, build_info.get("USE_NCCL"))
    if isinstance(raw_flag, str):
        lowered = raw_flag.strip().lower()
        return lowered in {"true", "1", "on", "yes"}
    if isinstance(raw_flag, (bool, int)):
        return bool(raw_flag)
    return None


def _probe_cuda_training(xgb_module: Any, gpu_device_id: int) -> tuple[bool, str | None]:
    try:
        feature_matrix = np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]],
            dtype=np.float32,
        )
        target_array = np.asarray([0.0, 1.0, 0.5, 0.2], dtype=np.float32)
        dmatrix = xgb_module.DMatrix(
            feature_matrix,
            label=target_array,
            feature_names=["f0", "f1"],
        )
        xgb_module.train(
            params={
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": "cuda",
                "gpu_id": gpu_device_id,
                "max_depth": 2,
                "eta": 0.1,
                "verbosity": 0,
            },
            dtrain=dmatrix,
            num_boost_round=1,
            evals=[(dmatrix, "train")],
            verbose_eval=False,
        )
        return True, None
    except Exception as error:
        return False, str(error)


def _detect_nvidia_gpu_name(device_id: int) -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    if not output:
        return None
    for raw_line in output.splitlines():
        parts = [part.strip() for part in raw_line.split(",", maxsplit=1)]
        if len(parts) != 2:
            continue
        index_text, gpu_name = parts
        try:
            if int(index_text) == device_id:
                return gpu_name
        except ValueError:
            continue
    return None


def resolve_acceleration_plan(
    config: OptimizationConfig,
    *,
    xgb_module: Any,
) -> AccelerationPlan:
    requested = normalize_requested_accelerator(config.compute_accelerator)
    cuda_flag = _extract_cuda_build_flag(xgb_module)

    if requested == "cpu":
        return AccelerationPlan(
            requested_accelerator=requested,
            accelerator="cpu",
            gpu_device_id=config.gpu_device_id,
            gpu_name=None,
            reason="CPU forced by optimization config.",
            use_gpu_matrix_cache=False,
        )

    if cuda_flag is False and requested == "cuda":
        raise RuntimeError(
            "CUDA acceleration requested, but current xgboost build does not include CUDA support.",
        )

    cuda_available, cuda_error = _probe_cuda_training(xgb_module, config.gpu_device_id)
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA acceleration requested, but runtime CUDA probe failed: "
            f"{cuda_error or 'unknown error'}",
        )

    if requested == "auto":
        if cuda_flag is False:
            return AccelerationPlan(
                requested_accelerator=requested,
                accelerator="cpu",
                gpu_device_id=config.gpu_device_id,
                gpu_name=None,
                reason="xgboost build has no CUDA support.",
                use_gpu_matrix_cache=False,
            )
        if not cuda_available:
            return AccelerationPlan(
                requested_accelerator=requested,
                accelerator="cpu",
                gpu_device_id=config.gpu_device_id,
                gpu_name=None,
                reason=(
                    "CUDA runtime probe failed; using CPU fallback"
                    f" ({cuda_error or 'unknown error'})."
                ),
                use_gpu_matrix_cache=False,
            )

    gpu_name = _detect_nvidia_gpu_name(config.gpu_device_id)
    return AccelerationPlan(
        requested_accelerator=requested,
        accelerator="cuda",
        gpu_device_id=config.gpu_device_id,
        gpu_name=gpu_name,
        reason="CUDA runtime probe succeeded.",
        use_gpu_matrix_cache=config.enable_gpu_matrix_cache,
    )
