from __future__ import annotations

from collections import OrderedDict
import logging
from pathlib import Path
from threading import RLock
from typing import cast

import numpy as np
import pandas as pd

from core.src.meta_model.feature_selection.io import FeatureSelectionMetadata
from core.src.meta_model.model_contract import DATE_COLUMN, MODEL_TARGET_COLUMN, REALIZED_RETURN_COLUMN, SPLIT_COLUMN, TICKER_COLUMN

"""LRU-cached feature array store for the feature selection pipeline."""

LOGGER: logging.Logger = logging.getLogger(__name__)
SCORING_CONTEXT_COLUMNS: tuple[str, ...] = (
    DATE_COLUMN,
    TICKER_COLUMN,
    SPLIT_COLUMN,
    MODEL_TARGET_COLUMN,
    REALIZED_RETURN_COLUMN,
    "company_sector",
    "company_beta",
    "stock_open_price",
    "stock_trading_volume",
)


class FeatureSelectionRuntimeCache:
    """Memory-bounded LRU cache of feature arrays backed by Parquet or in-memory DataFrame."""
    def __init__(
        self,
        dataset_source: Path | pd.DataFrame,
        metadata: FeatureSelectionMetadata,
        *,
        random_seed: int,
        max_cache_gib: float,
    ) -> None:
        self._dataset_path = dataset_source if isinstance(dataset_source, Path) else None
        self._dataset_frame = (
            pd.DataFrame(dataset_source.copy()) if isinstance(dataset_source, pd.DataFrame) else None
        )
        self._metadata = metadata
        self._random_seed = random_seed
        self._max_cache_bytes = int(max_cache_gib * (1024.0 ** 3))
        self._feature_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._cache_lock = RLock()
        self._train_context = self._load_train_context_frame()

    @property
    def train_row_count(self) -> int:
        return len(self._train_context)

    def build_feature_frame(
        self,
        feature_names: list[str],
        *,
        row_indices: np.ndarray | None = None,
    ) -> pd.DataFrame:
        target_row_indices = self._resolve_row_indices(row_indices)
        context_columns = [
            column_name
            for column_name in self._train_context.columns
            if column_name not in set(feature_names)
        ]
        frame = pd.DataFrame(
            self._train_context.loc[:, context_columns].take(target_row_indices).reset_index(drop=True),
        )
        if not feature_names:
            return frame
        self._ensure_feature_arrays(feature_names)
        with self._cache_lock:
            feature_arrays = {
                feature_name: self._feature_cache[feature_name]
                for feature_name in feature_names
                if feature_name in self._feature_cache
            }
        missing_from_cache = [feature_name for feature_name in feature_names if feature_name not in feature_arrays]
        if missing_from_cache:
            LOGGER.info(
                "Feature selection cache fallback: loading %d columns directly from parquet because they are not retained in cache",
                len(missing_from_cache),
            )
            feature_arrays.update(self._load_feature_arrays_direct(missing_from_cache))
        selected_feature_frame = pd.DataFrame(
            {
                feature_name: feature_arrays[feature_name][target_row_indices]
                for feature_name in feature_names
            },
            index=frame.index,
        )
        return pd.concat([frame, selected_feature_frame], axis=1)

    def build_sampled_feature_frame(
        self,
        feature_names: list[str],
        *,
        sample_size: int,
    ) -> pd.DataFrame:
        sample_positions = self._build_sample_positions(sample_size)
        feature_frame = self.build_feature_frame(feature_names, row_indices=sample_positions)
        sampled_frame = cast(pd.DataFrame, feature_frame.loc[:, feature_names].copy())
        return pd.DataFrame(sampled_frame)

    def get_feature_array(self, feature_name: str) -> np.ndarray:
        return self.get_feature_array_slice(feature_name)

    def get_feature_array_slice(
        self,
        feature_name: str,
        *,
        row_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        self._ensure_feature_arrays([feature_name])
        with self._cache_lock:
            feature_array = self._feature_cache[feature_name]
        if row_indices is None:
            return np.asarray(feature_array, dtype=np.float32)
        target_row_indices = self._resolve_row_indices(row_indices)
        return np.asarray(feature_array[target_row_indices], dtype=np.float32)

    def feature_coverage_fraction(self, feature_name: str) -> float:
        feature_array = self.get_feature_array(feature_name)
        if feature_array.size == 0:
            return 0.0
        finite_mask = np.isfinite(feature_array)
        return float(np.mean(finite_mask))

    def _load_train_context_frame(self) -> pd.DataFrame:
        available_columns = [column_name for column_name in SCORING_CONTEXT_COLUMNS if column_name in self._metadata.available_columns]
        if self._dataset_frame is not None:
            loaded = pd.DataFrame(self._dataset_frame.loc[:, available_columns].copy())
        else:
            if self._dataset_path is None:
                raise ValueError("Feature selection cache requires either a dataset path or dataset frame.")
            loaded = pd.read_parquet(self._dataset_path, columns=available_columns)
        ordered = pd.DataFrame(loaded.take(self._metadata.canonical_order))
        train_frame = pd.DataFrame(ordered.take(self._metadata.train_row_indices).reset_index(drop=True))
        if DATE_COLUMN in train_frame.columns:
            date_series = cast(pd.Series, train_frame[DATE_COLUMN])
            train_frame[DATE_COLUMN] = pd.to_datetime(date_series)
        return train_frame

    def _resolve_row_indices(self, row_indices: np.ndarray | None) -> np.ndarray:
        if row_indices is None:
            return np.arange(self.train_row_count, dtype=np.int64)
        return np.asarray(row_indices, dtype=np.int64)

    def _build_sample_positions(self, sample_size: int) -> np.ndarray:
        if sample_size <= 0 or self.train_row_count <= sample_size:
            return np.arange(self.train_row_count, dtype=np.int64)
        rng = np.random.default_rng(self._random_seed)
        sampled = rng.choice(np.arange(self.train_row_count, dtype=np.int64), size=sample_size, replace=False)
        return np.sort(sampled)

    def build_temporal_sample_row_indices(
        self,
        sample_size: int,
        *,
        minimum_date_count: int = 1,
    ) -> np.ndarray:
        if sample_size <= 0 or self.train_row_count <= sample_size:
            return np.arange(self.train_row_count, dtype=np.int64)
        if minimum_date_count <= 0:
            raise ValueError("minimum_date_count must be strictly positive.")
        date_values = pd.to_datetime(cast(pd.Series, self._train_context[DATE_COLUMN])).to_numpy(copy=False)
        if date_values.size == 0:
            return np.arange(0, dtype=np.int64)
        boundary_positions = np.flatnonzero(date_values[1:] != date_values[:-1]) + 1
        start_positions = np.concatenate((np.asarray([0], dtype=np.int64), boundary_positions))
        stop_positions = np.concatenate((boundary_positions, np.asarray([len(date_values)], dtype=np.int64)))
        rows_per_date = stop_positions - start_positions
        unique_date_count = len(start_positions)
        median_rows_per_date = max(1, int(np.median(rows_per_date)))
        target_date_count = min(
            unique_date_count,
            max(minimum_date_count, int(np.ceil(sample_size / median_rows_per_date))),
        )
        sampled_date_positions = np.unique(
            np.linspace(0, unique_date_count - 1, num=target_date_count, dtype=np.int64),
        )
        rows_per_selected_date = sample_size // len(sampled_date_positions)
        remainder = sample_size % len(sampled_date_positions)
        sampled_rows: list[np.ndarray] = []
        for offset, date_position in enumerate(sampled_date_positions.tolist()):
            start = int(start_positions[date_position])
            stop = int(stop_positions[date_position])
            available = stop - start
            row_budget = rows_per_selected_date + (1 if offset < remainder else 0)
            row_count = min(available, max(1, row_budget))
            if row_count >= available:
                sampled_rows.append(np.arange(start, stop, dtype=np.int64))
                continue
            sampled_rows.append(
                np.unique(
                    np.linspace(start, stop - 1, num=row_count, dtype=np.int64),
                ),
            )
        return np.sort(np.concatenate(sampled_rows)).astype(np.int64, copy=False)

    def _ensure_feature_arrays(self, feature_names: list[str]) -> None:
        with self._cache_lock:
            missing_columns = [feature_name for feature_name in feature_names if feature_name not in self._feature_cache]
            if not missing_columns:
                self._touch_feature_arrays(feature_names)
                return
        if self._dataset_frame is not None:
            loaded = pd.DataFrame(self._dataset_frame.loc[:, missing_columns].copy())
        else:
            if self._dataset_path is None:
                raise ValueError("Feature selection cache requires either a dataset path or dataset frame.")
            loaded = pd.read_parquet(self._dataset_path, columns=missing_columns)
        ordered = pd.DataFrame(loaded.take(self._metadata.canonical_order))
        train_frame = pd.DataFrame(ordered.take(self._metadata.train_row_indices).reset_index(drop=True))
        with self._cache_lock:
            for feature_name in missing_columns:
                if feature_name in self._feature_cache:
                    continue
                feature_series = cast(pd.Series, train_frame[feature_name])
                feature_array = feature_series.to_numpy(dtype=np.float32, copy=False)
                cached_array = np.array(feature_array, dtype=np.float32, copy=True)
                self._feature_cache[feature_name] = cached_array
                self._cache_bytes += cached_array.nbytes
            self._touch_feature_arrays(feature_names)
            self._enforce_cache_limit()

    def _load_feature_arrays_direct(self, feature_names: list[str]) -> dict[str, np.ndarray]:
        if self._dataset_frame is not None:
            loaded = pd.DataFrame(self._dataset_frame.loc[:, feature_names].copy())
        else:
            if self._dataset_path is None:
                raise ValueError("Feature selection cache requires either a dataset path or dataset frame.")
            loaded = pd.read_parquet(self._dataset_path, columns=feature_names)
        ordered = pd.DataFrame(loaded.take(self._metadata.canonical_order))
        train_frame = pd.DataFrame(ordered.take(self._metadata.train_row_indices).reset_index(drop=True))
        feature_arrays: dict[str, np.ndarray] = {}
        for feature_name in feature_names:
            feature_series = cast(pd.Series, train_frame[feature_name])
            feature_arrays[feature_name] = feature_series.to_numpy(dtype=np.float32, copy=True)
        return feature_arrays

    def _touch_feature_arrays(self, feature_names: list[str]) -> None:
        for feature_name in feature_names:
            self._feature_cache.move_to_end(feature_name)

    def _enforce_cache_limit(self) -> None:
        while self._cache_bytes > self._max_cache_bytes and self._feature_cache:
            _, feature_array = self._feature_cache.popitem(last=False)
            self._cache_bytes -= feature_array.nbytes


__all__ = ["FeatureSelectionRuntimeCache", "SCORING_CONTEXT_COLUMNS"]
