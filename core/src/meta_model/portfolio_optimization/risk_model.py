from __future__ import annotations

"""Train-only covariance model for portfolio optimisation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from core.src.meta_model.model_contract import DATE_COLUMN, SPLIT_COLUMN, TICKER_COLUMN, TRAIN_SPLIT_NAME
from core.src.meta_model.split_guard import assert_train_only_fit_frame


@dataclass(frozen=True)
class FittedRiskModel:
    covariance: pd.DataFrame
    lookback_days: int

    def subset(self, tickers: list[str]) -> np.ndarray:
        covariance = self.covariance.reindex(index=tickers, columns=tickers)
        matrix = covariance.to_numpy(dtype=np.float64, copy=True)
        if matrix.size == 0:
            return matrix
        diag = np.diag(matrix).copy()
        finite_diag = diag[np.isfinite(diag) & (diag > 0.0)]
        fallback_var = float(np.mean(finite_diag)) if finite_diag.size else 0.05
        for idx in range(len(tickers)):
            if not np.isfinite(matrix[idx, idx]) or matrix[idx, idx] <= 0.0:
                matrix[idx, idx] = fallback_var
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        symmetric = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(symmetric)
        clipped = np.clip(eigvals, 1e-8, None)
        return eigvecs @ np.diag(clipped) @ eigvecs.T


def fit_train_only_covariance_model(
    data: pd.DataFrame,
    *,
    return_column: str = "stock_close_log_return_lag_1d",
    lookback_days: int = 63,
    min_history_days: int = 40,
) -> FittedRiskModel:
    assert_train_only_fit_frame(
        data,
        split_column=SPLIT_COLUMN,
        train_split_name=TRAIN_SPLIT_NAME,
        context="portfolio_optimization.risk_model.fit",
    )
    if return_column not in data.columns:
        raise ValueError(f"Missing return column required for risk model: {return_column}")
    ordered = data.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    pivot = ordered.pivot(index=DATE_COLUMN, columns=TICKER_COLUMN, values=return_column)
    if len(pivot.index) < lookback_days:
        raise ValueError("Risk model fit failed: insufficient dates for requested lookback window.")
    lookback_pivot = pd.DataFrame()
    valid_columns: list[object] = []
    for end_index in range(len(pivot.index), lookback_days - 1, -1):
        candidate = pd.DataFrame(pivot.iloc[end_index - lookback_days:end_index].copy())
        candidate_valid_columns = [
            column
            for column in candidate.columns
            if int(candidate[column].notna().sum()) >= min_history_days
        ]
        if candidate_valid_columns:
            lookback_pivot = candidate
            valid_columns = candidate_valid_columns
            break
    if lookback_pivot.empty or not valid_columns:
        raise ValueError("Risk model fit failed: no ticker reached min history requirement.")
    matrix = lookback_pivot.loc[:, valid_columns].fillna(0.0).to_numpy(dtype=np.float64)
    lw = LedoitWolf()
    lw.fit(matrix)
    covariance = pd.DataFrame(
        lw.covariance_,
        index=[str(col) for col in valid_columns],
        columns=[str(col) for col in valid_columns],
    )
    return FittedRiskModel(covariance=covariance, lookback_days=lookback_days)
