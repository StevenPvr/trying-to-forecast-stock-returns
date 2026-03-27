from __future__ import annotations

import os
from dataclasses import dataclass


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
    detected_cores = total_cores if total_cores is not None else max(1, (os.cpu_count() or 1) - 1)
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
