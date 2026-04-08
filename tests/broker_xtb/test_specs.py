from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.broker_xtb.specs import build_default_spec_provider


def test_strict_snapshot_provider_rejects_unknown_symbols(
    tmp_path: Path,
) -> None:
    specs_path = tmp_path / "xtb_instrument_specs.json"
    specs_path.write_text(
        json.dumps([
            {
                "symbol": "AAPL",
                "instrument_group": "stock_cash",
                "currency": "USD",
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
                "long_swap_bps_daily": 0.0,
                "short_swap_bps_daily": 0.0,
                "margin_requirement": 1.0,
                "max_adv_participation": 0.05,
                "effective_from": "2000-01-01",
                "effective_to": None,
            },
        ]),
        encoding="utf-8",
    )
    provider = build_default_spec_provider(
        path=specs_path,
        allow_defaults_if_missing=False,
        require_explicit_symbols=True,
    )

    resolved = provider.resolve("AAPL", pd.Timestamp("2024-01-02"))

    assert resolved.symbol == "AAPL"
    with pytest.raises(KeyError, match="MSFT"):
        provider.resolve("MSFT", pd.Timestamp("2024-01-02"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
