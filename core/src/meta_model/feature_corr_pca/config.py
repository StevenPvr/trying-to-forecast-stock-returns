from __future__ import annotations

import os
from datetime import date

TRAIN_END_DATE: date = date(2018, 11, 30)
DEFAULT_CORRELATION_THRESHOLD: float = 0.95
DEFAULT_KERNEL: str = "rbf"
DEFAULT_COMPONENTS_PER_GROUP: int = 1
CORRELATION_BATCH_ROWS: int = 2048
TRANSFORM_BATCH_ROWS: int = 2048
MIN_SHARED_ROWS_FOR_CORRELATION: int = 32
MAX_KERNEL_PCA_TRAIN_ROWS: int = 2048
MAX_IN_FLIGHT_FUTURES_MULTIPLIER: int = 2
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "stock_close_price",
    "target_main",
    "dataset_split",
    "row_position",
)


def resolve_max_workers(max_workers: int | None = None) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)
