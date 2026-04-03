from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from core.src.meta_model.broker_xtb.specs import build_default_spec_provider
from core.src.meta_model.data.paths import (
    FUNDAMENTALS_HISTORY_CSV,
    MEMBERSHIP_HISTORY_CSV,
    XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LaunchReadinessReport:
    is_ready: bool
    missing_paths: list[str]
    stock_cfd_count: int
    index_cfd_count: int
    lightgbm_available: bool
    fred_api_key_available: bool
    required_paths: dict[str, str]


def _build_required_paths() -> dict[str, str]:
    return {
        "membership_history_csv": str(MEMBERSHIP_HISTORY_CSV),
        "fundamentals_history_csv": str(FUNDAMENTALS_HISTORY_CSV),
        "xtb_instrument_specs_json": str(XTB_INSTRUMENT_SPECS_REFERENCE_JSON),
    }


def _list_missing_paths(required_paths: dict[str, str]) -> list[str]:
    return [
        path
        for path in required_paths.values()
        if not Path(path).exists()
    ]


def build_launch_readiness_report() -> LaunchReadinessReport:
    required_paths = _build_required_paths()
    missing_paths = _list_missing_paths(required_paths)
    stock_cfd_count = 0
    index_cfd_count = 0
    if not missing_paths:
        provider = build_default_spec_provider(
            path=XTB_INSTRUMENT_SPECS_REFERENCE_JSON,
            allow_defaults_if_missing=False,
            require_explicit_symbols=True,
        )
        stock_cfd_count = sum(
            1 for spec in provider.specs if spec.instrument_group == "stock_cfd"
        )
        index_cfd_count = sum(
            1 for spec in provider.specs if spec.instrument_group == "index_cfd"
        )
    lightgbm_available = importlib.util.find_spec("lightgbm") is not None
    fred_api_key_available = bool(os.environ.get("FRED_API_KEY", "").strip())
    is_ready = (
        not missing_paths
        and stock_cfd_count > 0
        and lightgbm_available
        and fred_api_key_available
    )
    return LaunchReadinessReport(
        is_ready=is_ready,
        missing_paths=missing_paths,
        stock_cfd_count=stock_cfd_count,
        index_cfd_count=index_cfd_count,
        lightgbm_available=lightgbm_available,
        fred_api_key_available=fred_api_key_available,
        required_paths=required_paths,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = build_launch_readiness_report()
    payload = dataclasses.asdict(report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if report.is_ready:
        LOGGER.info("Launch readiness check passed.")
        return
    LOGGER.error("Launch readiness check failed.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
