from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock
import time
from typing import Any, cast

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.feature_selection.grouping import (
    BucketGroupingResult,
    FeatureBucketKey,
    FeatureGroup,
    build_feature_buckets,
    build_feature_groups,
    normalize_feature_stem,
    partition_feature_bucket,
)


class TestFeatureGrouping:
    def test_normalize_feature_stem_removes_window_and_lag_suffixes(self) -> None:
        assert normalize_feature_stem("quant_momentum_21d") == "quant_momentum"
        assert normalize_feature_stem("quant_momentum_lag_5") == "quant_momentum"
        assert normalize_feature_stem("macro_inflation_rate") == "macro_inflation_rate"

    def test_build_feature_buckets_groups_by_family_and_stem(self) -> None:
        buckets = build_feature_buckets(
            [
                "quant_momentum_21d",
                "quant_momentum_63d",
                "quant_value_21d",
                "macro_cpi_1m",
            ],
        )

        assert buckets[
            FeatureBucketKey(family="quant", stem="quant_momentum")
        ] == ["quant_momentum_21d", "quant_momentum_63d"]
        assert buckets[
            FeatureBucketKey(family="quant", stem="quant_value")
        ] == ["quant_value_21d"]
        assert buckets[
            FeatureBucketKey(family="macro", stem="macro_cpi_1m")
        ] == ["macro_cpi_1m"]

    def test_partition_feature_bucket_keeps_strongly_correlated_features_together(self) -> None:
        sampled_frame = pd.DataFrame(
            {
                "quant_combo_21d": [1.0, 2.0, 3.0, 4.0, 5.0],
                "quant_combo_63d": [1.1, 2.1, 3.1, 4.1, 5.1],
                "quant_other_21d": [1.0, 0.0, 1.0, 0.0, 1.0],
            },
        )

        groups = partition_feature_bucket(
            FeatureBucketKey(family="quant", stem="quant_combo"),
            sampled_frame,
            max_group_size=2,
        )

        group_members = [set(group.feature_names) for group in groups]
        assert {"quant_combo_21d", "quant_combo_63d"} in group_members

    def test_build_feature_groups_parallel_matches_serial(self) -> None:
        feature_names = [
            "quant_alpha_21d",
            "quant_alpha_63d",
            "quant_beta_21d",
            "quant_beta_63d",
        ]
        frame_by_bucket: dict[tuple[str, ...], pd.DataFrame] = {
            ("quant_alpha_21d", "quant_alpha_63d"): pd.DataFrame(
                {
                    "quant_alpha_21d": [1.0, 2.0, 3.0],
                    "quant_alpha_63d": [1.1, 2.1, 3.1],
                },
            ),
            ("quant_beta_21d", "quant_beta_63d"): pd.DataFrame(
                {
                    "quant_beta_21d": [3.0, 2.0, 1.0],
                    "quant_beta_63d": [3.1, 2.1, 1.1],
                },
            ),
        }

        class DummyCache:
            def build_sampled_feature_frame(
                self,
                bucket_feature_names: list[str],
                *,
                sample_size: int,
            ) -> pd.DataFrame:
                del sample_size
                return pd.DataFrame(frame_by_bucket[tuple(bucket_feature_names)].copy())

        serial_groups, serial_manifest = build_feature_groups(
            cast(Any, DummyCache()),
            feature_names,
            sample_size=10,
            max_group_size=4,
            parallel_workers=1,
        )
        parallel_groups, parallel_manifest = build_feature_groups(
            cast(Any, DummyCache()),
            feature_names,
            sample_size=10,
            max_group_size=4,
            parallel_workers=2,
        )

        assert serial_groups == parallel_groups
        pd.testing.assert_frame_equal(serial_manifest, parallel_manifest)

    def test_build_feature_groups_parallel_uses_multiple_workers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        feature_names = [
            "quant_alpha_21d",
            "quant_beta_21d",
            "quant_gamma_21d",
            "quant_delta_21d",
        ]
        state = {"active": 0, "max_active": 0}
        state_lock = Lock()

        def slow_group_single_bucket(
            cache: object,
            bucket_index: int,
            bucket_key: FeatureBucketKey,
            bucket_feature_names: list[str],
            *,
            sample_size: int,
            max_group_size: int,
        ) -> BucketGroupingResult:
            del cache, sample_size, max_group_size
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            with state_lock:
                state["active"] -= 1
            bucket_groups = [
                FeatureGroup(
                    group_id=f"{bucket_key.family}:{bucket_key.stem}:0:1",
                    family=bucket_key.family,
                    stem=bucket_key.stem,
                    feature_names=tuple(bucket_feature_names),
                ),
            ]
            return BucketGroupingResult(
                bucket_index=bucket_index,
                bucket_key=bucket_key,
                bucket_feature_count=len(bucket_feature_names),
                bucket_groups=bucket_groups,
                manifest_rows=[
                    {
                        "group_id": bucket_groups[0].group_id,
                        "feature_family": bucket_key.family,
                        "feature_stem": bucket_key.stem,
                        "group_level": 0,
                        "parent_group_id": None,
                        "feature_name": bucket_feature_names[0],
                    },
                ],
            )

        monkeypatch.setattr(
            "core.src.meta_model.feature_selection.grouping._group_single_bucket",
            slow_group_single_bucket,
        )

        class DummyCache:
            def build_sampled_feature_frame(
                self,
                bucket_feature_names: list[str],
                *,
                sample_size: int,
            ) -> pd.DataFrame:
                del bucket_feature_names, sample_size
                return pd.DataFrame()

        build_feature_groups(
            cast(Any, DummyCache()),
            feature_names,
            sample_size=10,
            max_group_size=4,
            parallel_workers=4,
        )

        assert state["max_active"] > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
