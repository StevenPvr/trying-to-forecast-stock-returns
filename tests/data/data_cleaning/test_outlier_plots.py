from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_cleaning.outlier_plots import _get_pyplot, create_outlier_plots


class TestGetPyplot:
    @patch("core.src.meta_model.data.data_cleaning.outlier_plots.import_module")
    @patch("core.src.meta_model.data.data_cleaning.outlier_plots.os.environ", {})
    def test_forces_agg_backend_when_env_not_set(
        self,
        mock_import_module: MagicMock,
    ) -> None:
        matplotlib_module = MagicMock()
        pyplot_module = MagicMock()
        mock_import_module.side_effect = [matplotlib_module, pyplot_module]

        result = _get_pyplot()

        matplotlib_module.use.assert_called_once_with("Agg", force=True)
        assert result is pyplot_module


class TestCreateOutlierPlots:
    def test_creates_expected_plot_files(self, tmp_path: Path) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03",
            ]),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "stock_open_log_return": [0.01, 0.01, 0.02, 0.50],
            "is_outlier_flag": [False, False, False, True],
            "outlier_severity": ["normal", "normal", "normal", "extreme"],
        })

        paths: dict[str, Path] = create_outlier_plots(df, tmp_path)

        assert set(paths.keys()) == {
            "return_distribution",
            "daily_outlier_rate",
            "severity_counts",
        }
        for path in paths.values():
            assert path.exists()

    def test_returns_empty_when_missing_required_columns(self, tmp_path: Path) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "ticker": ["AAPL"],
        })
        paths: dict[str, Path] = create_outlier_plots(df, tmp_path)
        assert paths == {}


if __name__ == "__main__":
    pytest_module = __import__("pytest")
    pytest_module.main([__file__, "-v"])
