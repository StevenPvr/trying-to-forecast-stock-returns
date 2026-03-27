from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.feature_corr_pca.main import (
    find_correlated_feature_groups,
    apply_feature_corr_kernel_pca,
    derive_secondary_prediction_start_date,
    run_feature_corr_pca,
)


def _make_feature_df() -> pd.DataFrame:
    dates = pd.bdate_range("2010-01-01", periods=80)
    rows: list[dict[str, object]] = []
    for ticker in ("AAA", "BBB"):
        base = np.linspace(0.0, 1.0, len(dates))
        correlated = base * 2.0 + 0.001
        independent = np.sin(np.linspace(0.0, 6.0, len(dates)))
        for index, current_date in enumerate(dates):
            rows.append({
                "date": current_date,
                "ticker": ticker,
                "stock_close_price": 100.0 + index,
                "feature_a": base[index],
                "feature_b": correlated[index],
                "feature_c": independent[index],
            })
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def test_finds_highly_correlated_feature_groups() -> None:
    data = _make_feature_df()

    groups = find_correlated_feature_groups(
        data,
        feature_columns=["feature_a", "feature_b", "feature_c"],
        threshold=0.95,
    )

    assert groups == [["feature_a", "feature_b"]]


def test_run_feature_corr_pca_replaces_correlated_groups_and_saves_mapping(tmp_path: Path) -> None:
    input_path = tmp_path / "features.parquet"
    output_path = tmp_path / "features_corr_pca.parquet"
    sample_path = tmp_path / "features_corr_pca_sample.csv"
    mapping_path = tmp_path / "feature_corr_pca_mapping.json"
    _make_feature_df().to_parquet(input_path, index=False)

    transformed, mapping = run_feature_corr_pca(
        feature_parquet_path=input_path,
        output_parquet_path=output_path,
        output_sample_csv_path=sample_path,
        output_mapping_json_path=mapping_path,
        correlation_threshold=0.95,
        return_transformed_data=True,
    )

    assert transformed is not None
    assert output_path.exists()
    assert sample_path.exists()
    assert mapping_path.exists()
    assert "feature_c" in transformed.columns
    assert "corr_pca_group_001" in transformed.columns
    assert "feature_a" not in transformed.columns
    assert "feature_b" not in transformed.columns
    assert mapping["applied_groups"][0]["member_features"] == ["feature_a", "feature_b"]

    saved_mapping = json.loads(mapping_path.read_text())
    assert saved_mapping["applied_groups"][0]["component_feature_name"] == "corr_pca_group_001"


def test_corr_pca_uses_only_train_split_for_group_detection() -> None:
    dates = pd.to_datetime(
        [
            "2018-11-28",
            "2018-11-29",
            "2018-11-30",
            "2019-02-01",
            "2019-02-04",
            "2022-02-01",
            "2022-02-02",
        ],
    )
    data = pd.DataFrame({
        "date": dates,
        "ticker": ["AAA"] * len(dates),
        "stock_close_price": np.linspace(100.0, 105.0, len(dates)),
        "feature_train_anchor": [0.0, 1.0, 0.0, 5.0, 6.0, 7.0, 8.0],
        "feature_future_only_match": [1.0, 1.0, 0.0, 5.0, 6.0, 7.0, 8.0],
        "feature_other": [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    })

    transformed, mapping = apply_feature_corr_kernel_pca(
        data,
        correlation_threshold=0.95,
    )

    assert transformed.columns.tolist() == data.columns.tolist()
    assert mapping["applied_groups"] == []


def test_derive_secondary_prediction_start_date_uses_first_fully_available_row(tmp_path: Path) -> None:
    input_path = tmp_path / "features_with_secondary_predictions.parquet"
    data = _make_feature_df()
    data["pred_future_trend_5d"] = np.nan
    data["pred_future_realized_vol_5d"] = np.nan
    data.loc[data.index[4]:, "pred_future_trend_5d"] = 0.1
    data.loc[data.index[6]:, "pred_future_realized_vol_5d"] = 0.2
    data.to_parquet(input_path, index=False)

    start_date = derive_secondary_prediction_start_date(input_path)

    assert start_date == pd.Timestamp(data.loc[6, "date"])


def test_run_feature_corr_pca_filters_rows_before_secondary_start_date(tmp_path: Path) -> None:
    input_path = tmp_path / "features_with_burn.parquet"
    output_path = tmp_path / "features_corr_pca.parquet"
    sample_path = tmp_path / "features_corr_pca_sample.csv"
    mapping_path = tmp_path / "feature_corr_pca_mapping.json"
    data = _make_feature_df()
    data["pred_future_trend_5d"] = np.nan
    data["pred_future_realized_vol_5d"] = np.nan
    data.loc[data.index[4]:, "pred_future_trend_5d"] = 0.1
    data.loc[data.index[6]:, "pred_future_realized_vol_5d"] = 0.2
    data.to_parquet(input_path, index=False)
    expected_start_date_raw = pd.Timestamp(data.loc[6, "date"])
    if pd.isna(expected_start_date_raw):
        raise AssertionError("Expected start date cannot be NaT in the test fixture.")
    expected_start_date = cast(pd.Timestamp, expected_start_date_raw)

    transformed, mapping = run_feature_corr_pca(
        feature_parquet_path=input_path,
        output_parquet_path=output_path,
        output_sample_csv_path=sample_path,
        output_mapping_json_path=mapping_path,
        correlation_threshold=0.95,
        start_date=expected_start_date,
        return_transformed_data=True,
    )

    assert transformed is not None
    assert transformed["date"].min() == expected_start_date
    assert mapping["start_date"] == str(expected_start_date.date())
