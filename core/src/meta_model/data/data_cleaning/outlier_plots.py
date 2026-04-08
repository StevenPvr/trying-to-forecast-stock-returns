from __future__ import annotations

"""Outlier diagnostic plots: distribution histograms, box plots, and time-series overlays."""

import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from core.src.meta_model.data.constants import DEFAULT_RETURN_COL_CANDIDATES

LOGGER: logging.Logger = logging.getLogger(__name__)

_OKABE_ITO_COLORS: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
}


def _get_pyplot() -> Any:
    if not os.environ.get("MPLBACKEND"):
        matplotlib: Any = import_module("matplotlib")
        matplotlib.use("Agg", force=True)
    return import_module("matplotlib.pyplot")


def _apply_publication_style(plt: Any) -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_figure(fig: Any, output_path: Path) -> None:
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    pdf_path: Path = output_path.with_suffix(".pdf")
    fig.savefig(str(pdf_path), bbox_inches="tight")


def _resolve_return_column(df: pd.DataFrame) -> str | None:
    for candidate in DEFAULT_RETURN_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _plot_return_distribution(
    df: pd.DataFrame,
    return_col: str,
    output_path: Path,
) -> None:
    plt: Any = _get_pyplot()
    _apply_publication_style(plt)
    fig, ax = plt.subplots(figsize=(10, 6))
    normal_mask = cast(pd.Series, ~cast(pd.Series, df["is_outlier_flag"]))
    outlier_mask = cast(pd.Series, df["is_outlier_flag"])

    normal_values: pd.Series = df.loc[normal_mask, return_col].dropna()
    outlier_values: pd.Series = df.loc[outlier_mask, return_col].dropna()

    if not normal_values.empty:
        ax.hist(
            normal_values,
            bins=80,
            alpha=0.6,
            density=True,
            color=_OKABE_ITO_COLORS["blue"],
            label="normal",
        )
    if not outlier_values.empty:
        ax.hist(
            outlier_values,
            bins=80,
            alpha=0.6,
            density=True,
            color=_OKABE_ITO_COLORS["vermillion"],
            label="outlier",
        )

    ax.set_title("Distribution des rendements d'ouverture log et outliers")
    ax.set_xlabel("stock_open_log_return")
    ax.set_ylabel("densite")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_outlier_rate_timeline(df: pd.DataFrame, output_path: Path) -> None:
    plt: Any = _get_pyplot()
    _apply_publication_style(plt)
    daily_rate = cast(
        pd.Series,
        df.groupby("date", sort=True)["is_outlier_flag"].mean(),
    )
    daily: pd.DataFrame = daily_rate.reset_index(name="outlier_rate")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily["date"], daily["outlier_rate"], color=_OKABE_ITO_COLORS["vermillion"], linewidth=1.2)
    ax.set_title("Taux journalier d'outliers")
    ax.set_xlabel("date")
    ax.set_ylabel("proportion d'outliers")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_severity_counts(df: pd.DataFrame, output_path: Path) -> None:
    plt: Any = _get_pyplot()
    _apply_publication_style(plt)
    severity_order: list[str] = ["normal", "elevated", "extreme", "data_error"]
    counts = cast(
        pd.Series,
        cast(pd.Series, df["outlier_severity"])
        .value_counts()
        .reindex(severity_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    heights: np.ndarray = counts.astype(float).to_numpy()
    ax.bar(
        counts.index.astype(str).tolist(),
        heights,
        color=[
            _OKABE_ITO_COLORS["blue"],
            _OKABE_ITO_COLORS["green"],
            _OKABE_ITO_COLORS["orange"],
            _OKABE_ITO_COLORS["vermillion"],
        ],
    )
    ax.set_title("Nombre d'observations par severite")
    ax.set_xlabel("severite")
    ax.set_ylabel("nombre d'observations")
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def create_outlier_plots(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Create scientific-level outlier diagnostic plots and save them to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols: set[str] = {"date", "is_outlier_flag", "outlier_severity"}
    if not required_cols.issubset(df.columns):
        LOGGER.warning(
            "Missing columns for outlier plots (%s). Skipping figure generation.",
            ", ".join(sorted(required_cols - set(df.columns))),
        )
        return {}

    return_col: str | None = _resolve_return_column(df)
    if return_col is None:
        LOGGER.warning("No return column found for outlier plots. Skipping figure generation.")
        return {}

    clean_df: pd.DataFrame = df.copy()
    clean_df["date"] = pd.to_datetime(clean_df["date"])

    paths: dict[str, Path] = {
        "return_distribution": output_dir / "outlier_return_distribution.png",
        "daily_outlier_rate": output_dir / "outlier_daily_rate.png",
        "severity_counts": output_dir / "outlier_severity_counts.png",
    }

    _plot_return_distribution(clean_df, return_col, paths["return_distribution"])
    _plot_outlier_rate_timeline(clean_df, paths["daily_outlier_rate"])
    _plot_severity_counts(clean_df, paths["severity_counts"])

    LOGGER.info("Saved outlier plots in %s", output_dir)
    return paths
