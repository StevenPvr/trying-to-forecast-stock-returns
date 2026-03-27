from __future__ import annotations

import pandas as pd

def get_data_stooq(
    symbols: str | list[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame: ...
