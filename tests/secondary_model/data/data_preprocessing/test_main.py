from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.secondary_model.data.data_preprocessing.main import (
    SECONDARY_TARGET_SPECS,
    assign_secondary_dataset_splits,
    create_future_drawdown_target,
    create_future_realized_vol_target,
    create_future_trend_target,
    create_future_volume_regime_target,
    main,
)

def _make_feature_df() -> pd.DataFrame:
    dates = pd.bdate_range("2008-12-15", periods=3400)
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(("AAPL", "MSFT"), start=1):
        close_prices = 100.0 * ticker_index * np.power(1.002, np.arange(len(dates), dtype=float))
        trading_volume = 1_000_000.0 + (ticker_index * 10_000.0) + np.arange(len(dates), dtype=float) * 250.0
        for i, current_date in enumerate(dates):
            rows.append(
                {
                    "date": current_date,
                    "ticker": ticker,
                    "stock_close_price": close_prices[i],
                    "stock_trading_volume": trading_volume[i],
                    "feature_alpha": np.sin(i / 13.0) + ticker_index,
                    "feature_beta": np.cos(i / 17.0),
                },
            )
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


class TestSecondaryTargets:
    def test_creates_future_trend_target(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=8, freq="B"),
                "ticker": ["AAPL"] * 8,
                "stock_close_price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
                "stock_trading_volume": [1_000_000.0] * 8,
            },
        )

        result = create_future_trend_target(df)

        assert result.loc[0, "target_main"] == pytest.approx(np.log(105.0 / 100.0))

    def test_creates_future_realized_vol_target(self) -> None:
        growth_rate = 1.01
        prices = 100.0 * np.power(growth_rate, np.arange(10, dtype=float))
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="B"),
                "ticker": ["AAPL"] * 10,
                "stock_close_price": prices,
                "stock_trading_volume": [1_000_000.0] * 10,
            },
        )

        result = create_future_realized_vol_target(df)

        assert result.loc[0, "target_main"] == pytest.approx(0.0)

    def test_creates_future_volume_regime_target(self) -> None:
        volumes = np.array(
            [100.0] * 20 + [200.0] * 5 + [250.0] * 5,
            dtype=float,
        )
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=len(volumes), freq="B"),
                "ticker": ["AAPL"] * len(volumes),
                "stock_close_price": np.linspace(100.0, 130.0, len(volumes)),
                "stock_trading_volume": volumes,
            },
        )

        result = create_future_volume_regime_target(df)

        expected = np.log(200.0 / 100.0)
        assert result.loc[19, "target_main"] == pytest.approx(expected)

    def test_creates_future_drawdown_target(self) -> None:
        prices = [100.0, 99.0, 98.0, 101.0, 102.0, 103.0, 104.0]
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=len(prices), freq="B"),
                "ticker": ["AAPL"] * len(prices),
                "stock_close_price": prices,
                "stock_trading_volume": [1_000_000.0] * len(prices),
            },
        )

        result = create_future_drawdown_target(df)

        expected = min(np.log(99.0 / 100.0), np.log(98.0 / 100.0), np.log(101.0 / 100.0), np.log(102.0 / 100.0), np.log(103.0 / 100.0))
        assert result.loc[0, "target_main"] == pytest.approx(expected)


class TestSecondarySplits:
    def test_assigns_20pct_train_then_10pct_val_then_oos_test(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2009-01-02",
                        "2010-01-04",
                        "2011-01-03",
                        "2012-01-02",
                        "2013-01-02",
                        "2014-01-02",
                        "2015-01-02",
                        "2016-01-04",
                        "2017-01-02",
                        "2018-11-30",
                        "2019-02-01",
                        "2021-11-30",
                        "2022-02-01",
                    ],
                ),
                "ticker": ["AAPL"] * 13,
                "stock_close_price": np.linspace(100.0, 112.0, 13),
                "stock_trading_volume": np.linspace(1_000_000.0, 1_100_000.0, 13),
                "target_main": np.linspace(0.0, 1.2, 13),
            },
        )

        result = assign_secondary_dataset_splits(df)

        assert result["dataset_split"].tolist() == [
            "train",
            "train",
            "val",
            "test",
            "test",
            "test",
            "test",
            "test",
            "test",
            "test",
            "test",
            "test",
            "test",
        ]


class TestMain:
    def test_builds_four_secondary_preprocessed_datasets(self, tmp_path: Path) -> None:
        input_path = tmp_path / "features.parquet"
        _make_feature_df().to_parquet(input_path, index=False)
        output_dir = tmp_path / "secondary_preprocessing"

        with (
            patch(
                "core.src.secondary_model.data.data_preprocessing.main.build_secondary_greedy_filtered_features_parquet",
                side_effect=lambda target_name: input_path,
            ),
            patch("core.src.secondary_model.data.data_preprocessing.main.SECONDARY_DATA_PREPROCESSING_DIR", output_dir),
        ):
            main()

        for target_spec in SECONDARY_TARGET_SPECS:
            target_dir = output_dir / target_spec.name
            assert (target_dir / "dataset_preprocessed.parquet").exists()
            assert (target_dir / "dataset_preprocessed_train.parquet").exists()
            assert (target_dir / "dataset_preprocessed_val.parquet").exists()
            assert (target_dir / "dataset_preprocessed_test.parquet").exists()

            full_dataset = pd.read_parquet(target_dir / "dataset_preprocessed.parquet")
            train_dataset = pd.read_parquet(target_dir / "dataset_preprocessed_train.parquet")
            val_dataset = pd.read_parquet(target_dir / "dataset_preprocessed_val.parquet")
            test_dataset = pd.read_parquet(target_dir / "dataset_preprocessed_test.parquet")

            assert "target_main" in full_dataset.columns
            assert set(full_dataset["dataset_split"].unique()) == {"train", "val", "test"}
            assert set(train_dataset["dataset_split"].unique()) == {"train"}
            assert set(val_dataset["dataset_split"].unique()) == {"val"}
            assert set(test_dataset["dataset_split"].unique()) == {"test"}
