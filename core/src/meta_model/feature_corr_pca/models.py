from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KernelPCAGroupModel:
    component_feature_name: str
    member_features: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    train_scaled_fit: np.ndarray
    gamma: float
    principal_vector: np.ndarray
    train_row_mean: np.ndarray
    train_grand_mean: float
    train_complete_rows: int
    fit_rows_used: int
