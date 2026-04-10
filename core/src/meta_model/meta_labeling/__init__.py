"""Meta-labeling stage for refining the primary alpha signal."""

from __future__ import annotations

from typing import Any


def run_meta_labeling_pipeline(*args: Any, **kwargs: Any) -> dict[str, object]:
    from core.src.meta_model.meta_labeling.main import run_meta_labeling_pipeline as _run

    return _run(*args, **kwargs)


__all__ = ["run_meta_labeling_pipeline"]
