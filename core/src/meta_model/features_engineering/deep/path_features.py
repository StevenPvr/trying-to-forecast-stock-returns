from __future__ import annotations

import numpy as np
import pandas as pd

from core.src.meta_model.features_engineering.deep.helpers import as_series
from core.src.meta_model.features_engineering.deep.price_context import PriceContext


def add_path_features(data: pd.DataFrame, context: PriceContext) -> None:
    return_sign = pd.Series(np.sign(context.returns_1d.to_numpy()), index=context.index)
    prev_return_sign = pd.Series(np.sign(context.returns_1d.shift(1).to_numpy()), index=context.index)
    gap_sign = pd.Series(np.sign(context.gap_return.to_numpy()), index=context.index)
    intraday_sign = pd.Series(np.sign(context.intraday_return.to_numpy()), index=context.index)

    sign_flip = (return_sign != prev_return_sign) & prev_return_sign.ne(0.0)
    agreement = (gap_sign == intraday_sign) & gap_sign.ne(0.0) & intraday_sign.ne(0.0)
    reversal = (gap_sign != intraday_sign) & gap_sign.ne(0.0) & intraday_sign.ne(0.0)

    data["deep_path_sign_flip_rate_21d"] = sign_flip.astype(float).rolling(21, min_periods=21).mean()
    data["deep_path_gap_intraday_agreement_rate_21d"] = agreement.astype(float).rolling(21, min_periods=21).mean()
    data["deep_path_gap_intraday_reversal_rate_21d"] = reversal.astype(float).rolling(21, min_periods=21).mean()
    data["deep_path_close_near_high_rate_21d"] = context.close_location.gt(0.6).astype(float).rolling(21, min_periods=21).mean()
    data["deep_path_close_near_low_rate_21d"] = context.close_location.lt(-0.6).astype(float).rolling(21, min_periods=21).mean()
    data["deep_path_positive_tail_share_21d"] = context.returns_1d.gt(context.return_std_21d).astype(float).rolling(21, min_periods=21).mean()
    data["deep_path_negative_tail_share_21d"] = context.returns_1d.lt(-context.return_std_21d).astype(float).rolling(21, min_periods=21).mean()
    data["deep_state_return_autocorr_sign_21d"] = as_series(
        return_sign.rolling(21, min_periods=21).corr(prev_return_sign),
        context.index,
    )


__all__ = ["add_path_features"]
