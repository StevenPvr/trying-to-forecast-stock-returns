from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from core.src.meta_model.data.paths import (
    XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
    XTB_MARGIN_SNAPSHOT_JSON,
    XTB_SPECS_SNAPSHOT_JSON,
    XTB_SWAP_SNAPSHOT_JSON,
)

DEFAULT_STOCK_SYMBOLS: frozenset[str] = frozenset()
DEFAULT_INDEX_SYMBOLS: frozenset[str] = frozenset({"US500", "US100", "DE40", "UK100"})


@dataclass(frozen=True)
class XtbInstrumentSpec:
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


@dataclass(frozen=True)
class BrokerSpecProvider:
    specs: tuple[XtbInstrumentSpec, ...]
    fallback_to_defaults: bool = True
    _specs_by_symbol: dict[str, tuple[XtbInstrumentSpec, ...]] = field(init=False, repr=False)
    _default_stock_spec: XtbInstrumentSpec = field(init=False, repr=False)
    _default_index_spec: XtbInstrumentSpec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        grouped_specs: dict[str, list[XtbInstrumentSpec]] = {}
        default_stock_spec: XtbInstrumentSpec | None = None
        default_index_spec: XtbInstrumentSpec | None = None
        for spec in self.specs:
            grouped_specs.setdefault(spec.symbol.upper(), []).append(spec)
            if spec.symbol == "__default_stock__":
                default_stock_spec = spec
            if spec.symbol == "__default_index__":
                default_index_spec = spec
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
        fallback_specs = {spec.symbol: spec for spec in _build_default_specs()}
        if default_stock_spec is None:
            default_stock_spec = fallback_specs["__default_stock__"]
        if default_index_spec is None:
            default_index_spec = fallback_specs["__default_index__"]
        object.__setattr__(self, "_specs_by_symbol", ordered_specs)
        object.__setattr__(self, "_default_stock_spec", default_stock_spec)
        object.__setattr__(self, "_default_index_spec", default_index_spec)

    def find_explicit_specs(
        self,
        symbol: str,
        *,
        instrument_group: str | None = None,
    ) -> tuple[XtbInstrumentSpec, ...]:
        normalized_symbol = symbol.upper()
        matching_specs = tuple(
            spec
            for spec in self._specs_by_symbol.get(normalized_symbol, tuple())
            if not spec.symbol.startswith("__default")
            and (instrument_group is None or spec.instrument_group == instrument_group)
        )
        return matching_specs

    def available_symbols(
        self,
        *,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        instrument_group: str | None = None,
        max_spread_bps: float | None = None,
    ) -> set[str]:
        return {
            spec.symbol.upper()
            for spec in self.specs
            if not spec.symbol.startswith("__default")
            and (instrument_group is None or spec.instrument_group == instrument_group)
            and pd.Timestamp(spec.effective_from) <= end_date
            and (
                spec.effective_to is None
                or pd.Timestamp(spec.effective_to) >= start_date
            )
            and (max_spread_bps is None or spec.spread_bps <= max_spread_bps)
        }

    def validate_snapshot(self, *, require_explicit_symbols: bool) -> None:
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
        if normalized_symbol in DEFAULT_INDEX_SYMBOLS:
            return self._default_index_spec
        return self._default_stock_spec


def _build_default_specs() -> tuple[XtbInstrumentSpec, ...]:
    return (
        XtbInstrumentSpec(
            symbol="__default_stock__",
            instrument_group="stock_cfd",
            currency="USD",
            spread_bps=0.0,
            slippage_bps=0.0,
            long_swap_bps_daily=0.0,
            short_swap_bps_daily=0.0,
            margin_requirement=1.0,
            max_adv_participation=0.05,
            effective_from="2000-01-01",
        ),
        XtbInstrumentSpec(
            symbol="__default_index__",
            instrument_group="index_cfd",
            currency="USD",
            spread_bps=8.0,
            slippage_bps=2.0,
            long_swap_bps_daily=1.500,
            short_swap_bps_daily=1.500,
            margin_requirement=0.05,
            max_adv_participation=0.20,
            effective_from="2000-01-01",
        ),
    )


def load_instrument_specs(
    path: Path = XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
    *,
    allow_defaults_if_missing: bool = True,
) -> tuple[XtbInstrumentSpec, ...]:
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
    merged_specs = explicit_specs + tuple(
        spec for spec in default_specs if spec.symbol not in explicit_symbols
    )
    return merged_specs


def build_default_spec_provider(
    path: Path = XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
    *,
    allow_defaults_if_missing: bool = True,
    require_explicit_symbols: bool = False,
) -> BrokerSpecProvider:
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
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
