from __future__ import annotations

from dataclasses import dataclass

from core.src.meta_model.runtime_parallelism import resolve_available_cpu_count


@dataclass(frozen=True)
class ParallelismPlan:
    total_cores: int
    fold_workers: int
    threads_per_fold: int


def _resolve_memory_safe_fold_workers(
    usable_cores: int,
    fold_count: int,
    feature_matrix_bytes: int | None,
) -> int:
    if feature_matrix_bytes is None or feature_matrix_bytes <= 0:
        return max(1, min(fold_count, usable_cores))

    feature_matrix_mb = feature_matrix_bytes / (1024.0 * 1024.0)
    if feature_matrix_mb >= 2048.0:
        return 1
    if feature_matrix_mb >= 1024.0:
        return max(1, min(fold_count, usable_cores, 2))
    if feature_matrix_mb >= 512.0:
        return max(1, min(fold_count, usable_cores, 3))
    return max(1, min(fold_count, usable_cores))


def resolve_parallelism(
    total_cores: int | None,
    fold_count: int,
    feature_matrix_bytes: int | None = None,
) -> ParallelismPlan:
    detected_cores = total_cores if total_cores is not None else resolve_available_cpu_count()
    usable_cores = max(1, int(detected_cores))
    fold_workers = _resolve_memory_safe_fold_workers(
        usable_cores=usable_cores,
        fold_count=fold_count,
        feature_matrix_bytes=feature_matrix_bytes,
    )
    threads_per_fold = max(1, usable_cores // fold_workers)
    return ParallelismPlan(
        total_cores=usable_cores,
        fold_workers=fold_workers,
        threads_per_fold=threads_per_fold,
    )
