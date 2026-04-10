"""Train-only portfolio optimisation stage."""

from __future__ import annotations

from typing import Any


def run_portfolio_optimization_pipeline(*args: Any, **kwargs: Any) -> dict[str, object]:
    from core.src.meta_model.portfolio_optimization.main import run_portfolio_optimization_pipeline as _run

    return _run(*args, **kwargs)


__all__ = ["run_portfolio_optimization_pipeline"]
