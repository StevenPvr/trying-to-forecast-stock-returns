from __future__ import annotations

"""Model registry: specification, training, and prediction for all supported model families."""

from core.src.meta_model.model_registry.main import (
    ModelArtifact,
    ModelSpec,
    build_default_model_specs,
    fit_model,
    predict_model,
)

__all__ = [
    "ModelArtifact",
    "ModelSpec",
    "build_default_model_specs",
    "fit_model",
    "predict_model",
]
