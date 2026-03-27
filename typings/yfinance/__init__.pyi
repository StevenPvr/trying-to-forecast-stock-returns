from __future__ import annotations

import pandas as pd

def download(
    tickers: str | list[str],
    start: str | None = None,
    end: str | None = None,
    actions: bool = False,
    threads: bool = True,
    ignore_tz: bool | None = None,
    group_by: str = "column",
    auto_adjust: bool = True,
    back_adjust: bool = False,
    repair: bool = False,
    keepna: bool = False,
    progress: bool = True,
    period: str | None = None,
    interval: str = "1d",
    prepost: bool = False,
    rounding: bool = False,
    timeout: int = 10,
    session: object | None = None,
    multi_level_index: bool = True,
) -> pd.DataFrame | None: ...

class Ticker:
    def __init__(self, ticker: str) -> None: ...
    @property
    def info(self) -> dict[str, object]: ...
