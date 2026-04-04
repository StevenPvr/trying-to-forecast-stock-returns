"""Serialize yfinance.download calls — the library is not thread-safe across concurrent downloads."""

from __future__ import annotations

import threading

YFINANCE_DOWNLOAD_LOCK: threading.Lock = threading.Lock()
