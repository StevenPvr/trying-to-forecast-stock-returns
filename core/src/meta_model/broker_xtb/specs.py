from __future__ import annotations

"""XTB instrument specification management.

Loads, validates, and resolves per-symbol trading specifications (spreads,
swaps, margins, commissions) from a JSON snapshot produced by the reference
snapshot pipeline.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

LOGGER: logging.Logger = logging.getLogger(__name__)

from core.src.meta_model.data.paths import (
    XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
    XTB_MARGIN_SNAPSHOT_JSON,
    XTB_SPECS_SNAPSHOT_JSON,
    XTB_SWAP_SNAPSHOT_JSON,
)


@dataclass(frozen=True)
class XtbInstrumentSpec:
    """Immutable trading specification for a single XTB instrument."""

    symbol: str
    instrument_group: str
    currency: str
    spread_bps: float
    slippage_bps: float
    long_swap_bps_daily: float
    short_swap_bps_daily: float
    margin_requirement: float
    max_adv_participation: float
    effective_from: str
    effective_to: str | None = None
    commission_rate: float = 0.002
    monthly_commission_free_turnover_eur: float = 100_000.0
    minimum_commission_eur: float = 10.0
    minimum_order_value_eur: float = 10.0
    fx_conversion_bps: float = 50.0
    annual_custody_fee_rate: float = 0.0002
    custody_fee_threshold_eur: float = 250_000.0
    monthly_custody_fee_min_eur: float = 10.0
    supports_fractional_orders: bool = True


@dataclass(frozen=True)
class BrokerSpecProvider:
    """Registry that resolves instrument specifications by symbol and date.

    Resolves explicit per-symbol specifications without silently changing
    trading behavior.
    """

    specs: tuple[XtbInstrumentSpec, ...]
    fallback_to_defaults: bool = False
    _specs_by_symbol: dict[str, tuple[XtbInstrumentSpec, ...]] = field(init=False, repr=False)
    _default_stock_spec: XtbInstrumentSpec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        grouped_specs: dict[str, list[XtbInstrumentSpec]] = {}
        default_stock_spec: XtbInstrumentSpec | None = None
        for spec in self.specs:
            grouped_specs.setdefault(spec.symbol.upper(), []).append(spec)
            if spec.symbol == "__default_stock__":
                default_stock_spec = spec
        ordered_specs = {
            symbol: tuple(
                sorted(
                    symbol_specs,
                    key=lambda current_spec: (
                        current_spec.effective_from,
                        current_spec.effective_to or "",
                    ),
                ),
            )
            for symbol, symbol_specs in grouped_specs.items()
        }
        if default_stock_spec is None:
            default_stock_spec = _build_default_specs()[0]
        object.__setattr__(self, "_specs_by_symbol", ordered_specs)
        object.__setattr__(self, "_default_stock_spec", default_stock_spec)

    def find_explicit_specs(
        self,
        symbol: str,
        *,
        instrument_group: str | None = None,
    ) -> tuple[XtbInstrumentSpec, ...]:
        """Return all explicit specs for *symbol*, optionally filtered by group."""
        normalized_symbol = symbol.upper()
        return tuple(
            spec
            for spec in self._specs_by_symbol.get(normalized_symbol, tuple())
            if not spec.symbol.startswith("__default")
            and (instrument_group is None or spec.instrument_group == instrument_group)
        )

    def available_symbols(
        self,
        *,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        instrument_group: str | None = None,
        max_spread_bps: float | None = None,
    ) -> set[str]:
        """Return the set of symbols whose validity overlaps *[start_date, end_date]*."""
        del max_spread_bps
        return {
            spec.symbol.upper()
            for spec in self.specs
            if not spec.symbol.startswith("__default")
            and (instrument_group is None or spec.instrument_group == instrument_group)
            and pd.Timestamp(spec.effective_from) <= end_date
            and (spec.effective_to is None or pd.Timestamp(spec.effective_to) >= start_date)
        }

    def validate_snapshot(self, *, require_explicit_symbols: bool) -> None:
        """Raise ``ValueError`` if the snapshot contains no explicit symbols when required."""
        if not require_explicit_symbols:
            return
        explicit_specs = [
            spec for spec in self.specs if not spec.symbol.startswith("__default")
        ]
        if not explicit_specs:
            raise ValueError(
                "XTB instrument specification snapshot must contain at least one explicit tradable symbol.",
            )

    def resolve(self, symbol: str, trade_date: pd.Timestamp) -> XtbInstrumentSpec:
        """Return the most recent spec valid at *trade_date* for *symbol*."""
        normalized_symbol = symbol.upper()
        for spec in reversed(self._specs_by_symbol.get(normalized_symbol, tuple())):
            if pd.Timestamp(spec.effective_from) <= trade_date and (
                spec.effective_to is None
                or trade_date <= pd.Timestamp(spec.effective_to)
            ):
                return spec
        if not self.fallback_to_defaults:
            raise KeyError(
                f"No explicit XTB instrument spec found for symbol {normalized_symbol}.",
            )
        return self._default_stock_spec


def _build_default_specs() -> tuple[XtbInstrumentSpec, ...]:
    return (
        XtbInstrumentSpec(
            symbol="__default_stock__",
            instrument_group="stock_cash",
            currency="USD",
            spread_bps=0.0,
            slippage_bps=0.0,
            long_swap_bps_daily=0.0,
            short_swap_bps_daily=0.0,
            margin_requirement=1.0,
            max_adv_participation=0.05,
            effective_from="2000-01-01",
        ),
    )


def load_instrument_specs(
    path: Path = XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
    *,
    allow_defaults_if_missing: bool = False,
) -> tuple[XtbInstrumentSpec, ...]:
    """Load instrument specifications from a JSON snapshot.

    Args:
        path: Path to the JSON snapshot file.
        allow_defaults_if_missing: Return built-in defaults when *path* is absent.

    Returns:
        Tuple of instrument specifications.
    """
    if not path.exists():
        if allow_defaults_if_missing:
            return _build_default_specs()
        raise FileNotFoundError(
            f"Missing XTB instrument specification snapshot at {path}.",
        )
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if not raw_payload and not allow_defaults_if_missing:
        raise ValueError(
            f"XTB instrument specification snapshot at {path} is empty.",
        )
    explicit_specs = tuple(XtbInstrumentSpec(**item) for item in raw_payload)
    if not allow_defaults_if_missing:
        return explicit_specs
    default_specs = _build_default_specs()
    explicit_symbols = {spec.symbol for spec in explicit_specs}
    return explicit_specs + tuple(
        spec for spec in default_specs if spec.symbol not in explicit_symbols
    )


def build_default_spec_provider(
    path: Path = XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
    *,
    allow_defaults_if_missing: bool = False,
    require_explicit_symbols: bool = True,
) -> BrokerSpecProvider:
    """Build a provider from disk, optionally enforcing explicit symbols."""
    provider = BrokerSpecProvider(
        specs=load_instrument_specs(
            path,
            allow_defaults_if_missing=allow_defaults_if_missing,
        ),
        fallback_to_defaults=not require_explicit_symbols,
    )
    provider.validate_snapshot(require_explicit_symbols=require_explicit_symbols)
    return provider


def save_broker_snapshots(
    provider: BrokerSpecProvider,
    *,
    specs_path: Path = XTB_SPECS_SNAPSHOT_JSON,
    swap_path: Path = XTB_SWAP_SNAPSHOT_JSON,
    margin_path: Path = XTB_MARGIN_SNAPSHOT_JSON,
) -> None:
    """Persist full-spec, swap-only, and margin-only snapshots as JSON."""
    specs_payload = [asdict(spec) for spec in provider.specs]
    swap_payload = [
        {
            "symbol": spec.symbol,
            "long_swap_bps_daily": spec.long_swap_bps_daily,
            "short_swap_bps_daily": spec.short_swap_bps_daily,
            "effective_from": spec.effective_from,
            "effective_to": spec.effective_to,
        }
        for spec in provider.specs
    ]
    margin_payload = [
        {
            "symbol": spec.symbol,
            "margin_requirement": spec.margin_requirement,
            "max_adv_participation": spec.max_adv_participation,
            "effective_from": spec.effective_from,
            "effective_to": spec.effective_to,
        }
        for spec in provider.specs
    ]
    for output_path, payload in (
        (specs_path, specs_payload),
        (swap_path, swap_payload),
        (margin_path, margin_payload),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
