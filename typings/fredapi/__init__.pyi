from __future__ import annotations

import pandas as pd


class Fred:
    def __init__(self, api_key: str | None = None) -> None: ...
    def get_series(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pd.Series: ...  # type: ignore[type-arg]
