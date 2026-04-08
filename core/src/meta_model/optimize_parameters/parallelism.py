from __future__ import annotations

"""Worker-count resolution for Optuna and per-fold XGBoost parallelism."""

from dataclasses import dataclass

from core.src.meta_model.runtime_parallelism import resolve_available_cpu_count


@dataclass(frozen=True)
class ParallelismPlan:
    total_cores: int
    fold_workers: int
    threads_per_fold: int


def _resolve_fold_workers(usable_cores: int, fold_count: int) -> int:
    """Run up to one worker per fold, capped by available CPU cores."""
    return max(1, min(fold_count, usable_cores))


def resolve_parallelism(
    total_cores: int | None,
    fold_count: int,
    feature_matrix_bytes: int | None = None,
    accelerator: str = "cpu",
) -> ParallelismPlan:
    """Build CPU/GPU thread layout for walk-forward fold evaluation.

    ``feature_matrix_bytes`` is kept for call-site compatibility but no longer
    reduces parallelism (user policy: use all detected cores).

    CPU: ``fold_workers = min(fold_count, cores)``, then ``threads_per_fold`` is
    set to ``ceil(cores / fold_workers)`` so total XGBoost threads target full
    core use (mild oversubscription when ``cores`` does not divide evenly).

    CUDA: single fold worker, all cores as ``threads_per_fold`` for the GPU path.
    """
    del feature_matrix_bytes
    detected_cores = total_cores if total_cores is not None else resolve_available_cpu_count()
    usable_cores = max(1, int(detected_cores))
    if accelerator == "cuda":
        return ParallelismPlan(
            total_cores=usable_cores,
            fold_workers=1,
            threads_per_fold=usable_cores,
        )
    fold_workers = _resolve_fold_workers(usable_cores, fold_count)
    threads_per_fold = max(1, (usable_cores + fold_workers - 1) // fold_workers)
    return ParallelismPlan(
        total_cores=usable_cores,
        fold_workers=fold_workers,
        threads_per_fold=threads_per_fold,
    )
