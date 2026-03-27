from __future__ import annotations

import pandas as pd
from ta import add_all_ta_features

from core.src.meta_model.features_engineering.config import TA_FEATURE_PREFIX


def add_ta_features_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    ordered_group: pd.DataFrame = group.sort_values("date").reset_index(drop=True)
    return add_all_ta_features(
        ordered_group,
        open="stock_open_price",
        high="stock_high_price",
        low="stock_low_price",
        close="stock_close_price",
        volume="stock_trading_volume",
        fillna=False,
        colprefix=TA_FEATURE_PREFIX,
    )
