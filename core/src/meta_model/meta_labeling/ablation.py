from __future__ import annotations

"""Signal-level ablation study for meta-labeling refinement strategies.

Screens all strategies on train OOF predictions, then confirms top picks
on validation predictions.  No model retraining or portfolio optimisation
-- pure signal diagnostics.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core.src.meta_model.meta_labeling.features import REFINED_PREDICTION_COLUMN
from core.src.meta_model.meta_labeling.refinement import compute_refined_signal
from core.src.meta_model.model_contract import DATE_COLUMN, WEEK_HOLD_NET_RETURN_COLUMN

LOGGER: logging.Logger = logging.getLogger(__name__)

TARGET_COLUMN_FOR_ABLATION: str = WEEK_HOLD_NET_RETURN_COLUMN

ABLATION_REGIMES: tuple[dict[str, Any], ...] = (
    {"name": "baseline_hard_050", "strategy": "binary_gate"},
    {"name": "candidate_only", "strategy": "candidate_only"},
    {"name": "soft_shifted_045", "strategy": "soft_shifted", "soft_shifted_floor": 0.45},
    {"name": "rank_blend_025", "strategy": "rank_blend", "rank_blend_lambda": 0.25},
    {"name": "rank_blend_050", "strategy": "rank_blend", "rank_blend_lambda": 0.50},
    {"name": "rank_blend_100", "strategy": "rank_blend", "rank_blend_lambda": 1.00},
    {"name": "no_meta", "strategy": "no_meta"},
)

METRIC_KEYS: tuple[str, ...] = (
    "rank_ic",
    "rank_ic_std",
    "rank_ic_ir",
    "q5_q1_spread_bps",
    "breadth",
    "effective_n",
)


def _compute_daily_rank_ic(
    refined: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
) -> pd.Series:
    """Spearman correlation between refined signal and realised return per date."""
    df = pd.DataFrame({"refined": refined, "target": target, "date": dates})
    results: list[dict[str, object]] = []
    for date_val, group in df.groupby("date"):
        if len(group) < 5:
            continue
        finite_mask = np.isfinite(group["refined"]) & np.isfinite(group["target"])
        g = group.loc[finite_mask]
        if len(g) < 5 or g["refined"].std() < 1e-12:
            results.append({"date": date_val, "rank_ic": 0.0})
            continue
        corr: float = float(spearmanr(g["refined"], g["target"]).statistic)
        results.append({"date": date_val, "rank_ic": corr})
    if not results:
        return pd.Series(dtype=np.float64)
    return pd.DataFrame(results).set_index("date")["rank_ic"]


def _compute_quintile_spread(
    refined: np.ndarray,
    target: np.ndarray,
    dates: np.ndarray,
) -> float:
    """Mean return of top quintile minus bottom quintile, annualised in bps."""
    df = pd.DataFrame({"refined": refined, "target": target, "date": dates})
    spreads: list[float] = []
    for _, group in df.groupby("date"):
        finite = group.loc[np.isfinite(group["refined"]) & np.isfinite(group["target"])]
        if len(finite) < 5:
            continue
        quintile = pd.qcut(finite["refined"], 5, labels=False, duplicates="drop")
        if quintile.nunique() < 2:
            continue
        top = finite.loc[quintile == quintile.max(), "target"].mean()
        bottom = finite.loc[quintile == quintile.min(), "target"].mean()
        spreads.append(top - bottom)
    if not spreads:
        return 0.0
    mean_daily_spread: float = float(np.mean(spreads))
    return mean_daily_spread * 252 * 10_000


def _compute_breadth(refined: np.ndarray, dates: np.ndarray) -> float:
    """Mean number of non-zero refined predictions per date."""
    df = pd.DataFrame({"refined": refined, "date": dates})
    daily_counts = df.groupby("date")["refined"].apply(
        lambda s: (s.abs() > 1e-12).sum(),
    )
    return float(daily_counts.mean()) if len(daily_counts) > 0 else 0.0


def _compute_effective_n(refined: np.ndarray, dates: np.ndarray) -> float:
    """Mean Herfindahl-inverse of absolute refined signal weights per date."""
    df = pd.DataFrame({"refined": refined, "date": dates})
    eff_ns: list[float] = []
    for _, group in df.groupby("date"):
        abs_scores = np.abs(group["refined"].to_numpy(dtype=np.float64))
        total = abs_scores.sum()
        if total < 1e-12:
            eff_ns.append(0.0)
            continue
        weights = abs_scores / total
        hhi = float(np.sum(weights**2))
        eff_ns.append(1.0 / hhi if hhi > 0 else 0.0)
    return float(np.mean(eff_ns)) if eff_ns else 0.0


def compute_regime_metrics(
    frame: pd.DataFrame,
    regime_spec: dict[str, Any],
) -> dict[str, float]:
    """Apply one strategy and compute signal-level diagnostics."""
    strategy_kwargs: dict[str, Any] = {
        k: v for k, v in regime_spec.items() if k != "name"
    }
    refined_frame = compute_refined_signal(frame, **strategy_kwargs)
    refined_arr = refined_frame[REFINED_PREDICTION_COLUMN].to_numpy(dtype=np.float64)
    target_arr = pd.to_numeric(
        frame[TARGET_COLUMN_FOR_ABLATION], errors="coerce",
    ).to_numpy(dtype=np.float64)
    dates_arr = frame[DATE_COLUMN].to_numpy()
    daily_ic = _compute_daily_rank_ic(refined_arr, target_arr, dates_arr)
    rank_ic: float = float(daily_ic.mean()) if len(daily_ic) > 0 else 0.0
    rank_ic_std: float = float(daily_ic.std()) if len(daily_ic) > 1 else 1.0
    rank_ic_ir: float = rank_ic / rank_ic_std if rank_ic_std > 1e-12 else 0.0
    q5_q1: float = _compute_quintile_spread(refined_arr, target_arr, dates_arr)
    breadth: float = _compute_breadth(refined_arr, dates_arr)
    eff_n: float = _compute_effective_n(refined_arr, dates_arr)
    return {
        "rank_ic": rank_ic,
        "rank_ic_std": rank_ic_std,
        "rank_ic_ir": rank_ic_ir,
        "q5_q1_spread_bps": q5_q1,
        "breadth": breadth,
        "effective_n": eff_n,
    }


def run_ablation_screen(
    frame: pd.DataFrame,
    regimes: tuple[dict[str, Any], ...] = ABLATION_REGIMES,
) -> pd.DataFrame:
    """Screen all strategies on the given prediction panel."""
    rows: list[dict[str, float | str]] = []
    for regime in regimes:
        name: str = str(regime["name"])
        LOGGER.info("Ablation screen: computing metrics for regime=%s", name)
        metrics = compute_regime_metrics(frame, regime)
        rows.append({"regime": name, **metrics})
    result = pd.DataFrame(rows).set_index("regime")
    return result


def select_top_regimes(
    screen_results: pd.DataFrame,
    *,
    n_top: int = 2,
    forced: tuple[str, ...] = ("candidate_only", "no_meta"),
) -> list[str]:
    """Pick top-N by rank_ic_ir plus forced baselines."""
    ranked = screen_results.sort_values("rank_ic_ir", ascending=False)
    top_names: list[str] = ranked.index[:n_top].tolist()
    for name in forced:
        if name in ranked.index and name not in top_names:
            top_names.append(name)
    return top_names


def format_ablation_table(results: pd.DataFrame) -> str:
    """Format results as a readable fixed-width table."""
    header = f"{'regime':<25s} {'rank_ic':>8s} {'ic_ir':>8s} {'q5q1_bps':>10s} {'breadth':>8s} {'eff_n':>8s}"
    lines: list[str] = [header, "-" * len(header)]
    for regime, row in results.iterrows():
        lines.append(
            f"{str(regime):<25s} {row['rank_ic']:>8.4f} {row['rank_ic_ir']:>8.2f} "
            f"{row['q5_q1_spread_bps']:>10.1f} {row['breadth']:>8.1f} {row['effective_n']:>8.1f}",
        )
    return "\n".join(lines)
