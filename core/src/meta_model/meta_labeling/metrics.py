from __future__ import annotations

"""Binary classification metrics for the meta-labeling stage."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from core.src.meta_model.meta_labeling.features import META_PROBABILITY_COLUMN
from core.src.meta_model.meta_labeling.labels import META_LABEL_COLUMN


def has_binary_label_support(frame: pd.DataFrame) -> bool:
    labels = pd.to_numeric(frame[META_LABEL_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    finite = labels[np.isfinite(labels)]
    if finite.size == 0:
        return False
    return np.unique(finite.astype(np.int64)).size >= 2


def binary_average_precision(
    frame: pd.DataFrame,
) -> float:
    labels = pd.to_numeric(frame[META_LABEL_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    probabilities = pd.to_numeric(frame[META_PROBABILITY_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(labels) & np.isfinite(probabilities)
    finite_labels = labels[finite_mask]
    finite_probabilities = probabilities[finite_mask]
    if finite_labels.size == 0 or np.unique(finite_labels.astype(np.int64)).size < 2:
        return 0.0
    return float(average_precision_score(finite_labels, finite_probabilities))


def balanced_binary_sample_weights(
    labels: np.ndarray,
) -> np.ndarray:
    if labels.size == 0:
        return np.asarray([], dtype=np.float64)
    sample_weights = np.ones(labels.shape[0], dtype=np.float64)
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return sample_weights
    total_count = float(labels.shape[0])
    class_count = float(unique_labels.size)
    for class_value in unique_labels:
        class_mask = labels == class_value
        class_samples = float(np.sum(class_mask))
        if class_samples <= 0.0:
            continue
        sample_weights[class_mask] = total_count / (class_count * class_samples)
    return sample_weights


def binary_mcc(
    frame: pd.DataFrame,
    *,
    threshold: float,
    use_balanced_class_weights: bool,
) -> float:
    labels = pd.to_numeric(frame[META_LABEL_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    probabilities = pd.to_numeric(frame[META_PROBABILITY_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(labels) & np.isfinite(probabilities)
    finite_labels = labels[finite_mask].astype(np.int64)
    finite_predictions = (probabilities[finite_mask] >= float(threshold)).astype(np.int64)
    if finite_labels.size == 0:
        return 0.0
    sample_weight = (
        balanced_binary_sample_weights(finite_labels)
        if use_balanced_class_weights
        else None
    )
    return float(matthews_corrcoef(finite_labels, finite_predictions, sample_weight=sample_weight))


def binary_roc_auc(
    frame: pd.DataFrame,
) -> float:
    labels = pd.to_numeric(frame[META_LABEL_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    probabilities = pd.to_numeric(frame[META_PROBABILITY_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(labels) & np.isfinite(probabilities)
    finite_labels = labels[finite_mask]
    finite_probabilities = probabilities[finite_mask]
    if finite_labels.size == 0 or np.unique(finite_labels.astype(np.int64)).size < 2:
        return 0.5
    return float(roc_auc_score(finite_labels, finite_probabilities))


def binary_logloss(
    frame: pd.DataFrame,
) -> float:
    labels = pd.to_numeric(frame[META_LABEL_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    probabilities = pd.to_numeric(frame[META_PROBABILITY_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(labels) & np.isfinite(probabilities)
    finite_labels = labels[finite_mask]
    finite_probabilities = probabilities[finite_mask]
    if finite_labels.size == 0:
        return float("inf")
    clipped = np.clip(finite_probabilities, 1e-6, 1.0 - 1e-6)
    return float(-np.mean((finite_labels * np.log(clipped)) + ((1.0 - finite_labels) * np.log(1.0 - clipped))))
