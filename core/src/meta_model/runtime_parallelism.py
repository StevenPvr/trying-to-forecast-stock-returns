from __future__ import annotations

import os


def resolve_available_cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def resolve_executor_worker_count(
    *,
    task_count: int | None = None,
    requested_workers: int | None = None,
) -> int:
    if requested_workers is not None:
        if requested_workers <= 0:
            raise ValueError("requested_workers must be strictly positive.")
        worker_count = requested_workers
    else:
        worker_count = resolve_available_cpu_count()
    if task_count is None:
        return worker_count
    return max(1, min(worker_count, task_count))


def resolve_requested_worker_count(requested_workers: int | None = None) -> int:
    if requested_workers is not None:
        if requested_workers <= 0:
            raise ValueError("requested_workers must be strictly positive.")
        return requested_workers
    return resolve_available_cpu_count()


__all__ = [
    "resolve_available_cpu_count",
    "resolve_executor_worker_count",
    "resolve_requested_worker_count",
]
