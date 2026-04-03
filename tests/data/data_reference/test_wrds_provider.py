from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_reference.wrds_provider import (
    WrdsCredentials,
    fetch_fundq_history,
    resolve_requested_ticker_map,
    resolve_wrds_credentials,
)


def test_resolve_wrds_credentials_reads_project_env_names() -> None:
    with patch.dict(
        "os.environ",
        {"ID_WRDS": "user123", "PASSWORD_WRDS": "secret"},
        clear=True,
    ):
        credentials = resolve_wrds_credentials()

    assert credentials == WrdsCredentials(username="user123", password="secret")


def test_resolve_wrds_credentials_falls_back_to_wrds_standard_names() -> None:
    with patch.dict(
        "os.environ",
        {"WRDS_USER": "user123", "WRDS_PASSWORD": "secret"},
        clear=True,
    ):
        credentials = resolve_wrds_credentials()

    assert credentials == WrdsCredentials(username="user123", password="secret")


def test_fetch_fundq_history_uses_non_interactive_connection() -> None:
    fake_connection = MagicMock()
    fake_connection._Connection__make_sa_engine_conn = MagicMock()
    security_result = MagicMock()
    security_result.fetchall.return_value = [("001234", "MRSH", "MMC", "01", "A", None)]
    security_result.keys.return_value = ["gvkey", "tic", "ibtic", "iid", "secstat", "dldtei"]
    fundq_result = MagicMock()
    fundq_result.fetchall.return_value = [("001234", "MRSH", pd.Timestamp("2020-03-31"))]
    fundq_result.keys.return_value = ["gvkey", "source_ticker", "datadate"]
    fake_connection.connection.execute.side_effect = [security_result, fundq_result]
    fake_wrds_module = MagicMock()
    fake_wrds_module.Connection.return_value = fake_connection
    credentials = WrdsCredentials(username="user123", password="secret")

    with patch.dict(sys.modules, {"wrds": fake_wrds_module}):
        result = fetch_fundq_history(
            credentials=credentials,
            tickers=["MMC"],
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

    fake_wrds_module.Connection.assert_called_once_with(
        autoconnect=False,
        verbose=False,
        wrds_username="user123",
    )
    assert fake_connection._password == "secret"
    fake_connection._Connection__make_sa_engine_conn.assert_called_once_with(
        raise_err=True,
    )
    assert fake_connection.connection.execute.call_count == 2
    fake_connection.load_library_list.assert_called_once()
    fake_connection.close.assert_called_once()
    assert list(result["ticker"]) == ["MMC"]


def test_resolve_requested_ticker_map_prefers_ibtic_and_manual_aliases() -> None:
    security_aliases = pd.DataFrame(
        {
            "gvkey": ["001234", "001235", "001236"],
            "tic": ["MRSH", "ELV", "GOOG"],
            "ibtic": ["MMC", "ATHI", "GOOG/1"],
            "iid": ["01", "01", "03"],
            "secstat": ["A", "A", "A"],
            "dldtei": [pd.NaT, pd.NaT, pd.NaT],
        }
    )

    result = resolve_requested_ticker_map(["MMC", "ANTM", "GOOG"], security_aliases)

    assert list(result["requested_ticker"]) == ["MMC", "ANTM", "GOOG"]
    assert list(result["gvkey"]) == ["001234", "001235", "001236"]
    assert list(result["source_ticker"]) == ["MRSH", "ELV", "GOOG"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
