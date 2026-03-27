from __future__ import annotations

import pandas as pd

from core.src.meta_model.features_engineering.deep.event_features import add_event_features
from core.src.meta_model.features_engineering.deep.path_features import add_path_features
from core.src.meta_model.features_engineering.deep.price_context import build_price_context
from core.src.meta_model.features_engineering.deep.state_features import add_state_features


def add_deep_price_features_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    enriched = group.sort_values("date").reset_index(drop=True).copy()
    context = build_price_context(enriched)
    add_event_features(enriched, context)
    add_path_features(enriched, context)
    add_state_features(enriched, context)
    return enriched


__all__ = ["add_deep_price_features_for_ticker"]
