"""Evaluation pipeline: training, backtesting, and performance reporting."""

from __future__ import annotations

from typing import Any


def run_evaluate_pipeline(*args: Any, **kwargs: Any) -> tuple[object, object, dict[str, object]]:
    from core.src.meta_model.evaluate.main import run_evaluate_pipeline as _run

    return _run(*args, **kwargs)


__all__ = ["run_evaluate_pipeline"]
