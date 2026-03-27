from __future__ import annotations

import logging
import importlib
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT: Path = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.constants import RANDOM_SEED, SAMPLE_FRAC
from core.src.meta_model.data.data_preprocessing.main import (
    assign_dataset_splits,
    create_target_main,
    filter_from_start_date,
    remove_rows_with_missing_values,
)
from core.src.meta_model.data.paths import (
    DATA_FEATURE_SELECTION_DIR,
    FEATURE_CORR_PCA_OUTPUT_PARQUET,
    GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_CSV,
    GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_PARQUET,
    GREEDY_FORWARD_SELECTION_SCORES_CSV,
    GREEDY_FORWARD_SELECTION_SCORES_PARQUET,
    GREEDY_FORWARD_SELECTION_SELECTED_FEATURES_CSV,
    GREEDY_FORWARD_SELECTION_SELECTED_FEATURES_PARQUET,
)
from core.src.meta_model.feature_selection_lag.greedy_forward_selection.config import (
    CANDIDATE_FEATURE_NAME_COLUMN,
    SFIModelConfig,
    TARGET_COLUMN,
    TRAIN_HOLDOUT_FRACTION,
    TRAIN_SAMPLE_FRACTION,
    TRAIN_SPLIT_NAME,
)

LOGGER: logging.Logger = logging.getLogger(__name__)
EXCLUDED_FEATURE_COLUMNS = frozenset({"date", "ticker"})
PERSISTED_METADATA_COLUMNS = ("date", "ticker", "stock_close_price", TARGET_COLUMN)
MIN_RMSE_IMPROVEMENT: float = 1e-6


@dataclass(frozen=True)
class HoldoutWindow:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_row_count: int
    validation_row_count: int


class CatBoostRegressorProtocol(Protocol):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series],
        early_stopping_rounds: int,
        use_best_model: bool,
        verbose: bool = False,
    ) -> object: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class CatBoostRegressorFactory(Protocol):
    def __call__(
        self,
        *,
        random_seed: int,
        allow_writing_files: bool,
        iterations: int,
        depth: int,
        learning_rate: float,
        loss_function: str,
    ) -> CatBoostRegressorProtocol: ...


def _as_dataframe(value: object) -> pd.DataFrame:
    return cast(pd.DataFrame, value)


def _as_series(value: object) -> pd.Series:
    return cast(pd.Series, value)


def _format_duration(seconds: float) -> str:
    rounded_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(rounded_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{mins:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def load_catboost_regressor_class() -> CatBoostRegressorFactory:
    spec = importlib.util.find_spec("catboost")
    if spec is None:
        raise ImportError("catboost is not installed. Add it to the environment before running greedy selection.")
    module = importlib.import_module("catboost")
    return cast(CatBoostRegressorFactory, module.CatBoostRegressor)


def load_selection_scaffold(path: Path) -> pd.DataFrame:
    source = _as_dataframe(pd.read_parquet(path, columns=["date", "ticker", "stock_close_price"]))
    ordered = _as_dataframe(source.sort_values(["date", "ticker"]).reset_index(drop=True))
    ordered["row_position"] = np.arange(len(ordered), dtype=np.int64)
    filtered = filter_from_start_date(ordered)
    targeted = create_target_main(filtered)
    split_ready = assign_dataset_splits(targeted)
    cleaned = remove_rows_with_missing_values(split_ready, required_columns=[TARGET_COLUMN])
    scaffold = _as_dataframe(
        cleaned.loc[:, ["row_position", "date", "ticker", TARGET_COLUMN, "dataset_split"]]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True),
    )
    LOGGER.info(
        "Loaded greedy selection scaffold: %d rows (%d train / %d val / %d test).",
        len(scaffold),
        int((scaffold["dataset_split"] == "train").sum()),
        int((scaffold["dataset_split"] == "val").sum()),
        int((scaffold["dataset_split"] == "test").sum()),
    )
    return scaffold


def build_train_holdout_window(
    train_scaffold: pd.DataFrame,
    holdout_fraction: float = TRAIN_HOLDOUT_FRACTION,
) -> HoldoutWindow:
    if train_scaffold.empty:
        raise ValueError("Train scaffold is empty; cannot build holdout window.")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in the open interval (0, 1).")

    ordered_train = _as_dataframe(train_scaffold.sort_values(["date", "ticker"]).reset_index(drop=True))
    unique_dates = pd.Index(pd.to_datetime(ordered_train["date"]).drop_duplicates().sort_values())
    if len(unique_dates) < 2:
        raise ValueError("At least two unique train dates are required to build a holdout window.")

    validation_date_count = max(1, int(np.ceil(len(unique_dates) * holdout_fraction)))
    validation_dates = unique_dates[-validation_date_count:]
    train_dates = unique_dates[:-validation_date_count]
    if train_dates.empty:
        raise ValueError("Holdout split removed every train date; increase the train history or shrink holdout_fraction.")

    train_end = cast(pd.Timestamp, train_dates.max())
    validation_start = cast(pd.Timestamp, validation_dates.min())
    validation_end = cast(pd.Timestamp, validation_dates.max())
    train_row_count = int((ordered_train["date"] <= train_end).sum())
    validation_row_count = int(
        ((ordered_train["date"] >= validation_start) & (ordered_train["date"] <= validation_end)).sum(),
    )
    LOGGER.info(
        (
            "Built single holdout split on sampled train rows: "
            "train=%d rows (%s -> %s) | validation=%d rows (%s -> %s)."
        ),
        train_row_count,
        ordered_train["date"].min(),
        train_end,
        validation_row_count,
        validation_start,
        validation_end,
    )
    return HoldoutWindow(
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        train_row_count=train_row_count,
        validation_row_count=validation_row_count,
    )


def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    target_mean = float(np.mean(y_true))
    total_sum_squares = float(np.sum(np.square(y_true - target_mean)))
    if total_sum_squares <= float(np.finfo(np.float64).eps):
        return 0.0
    residual_sum_squares = float(np.sum(np.square(y_true - y_pred)))
    return 1.0 - (residual_sum_squares / total_sum_squares)


def _compute_zero_baseline_rmse(y_true: np.ndarray) -> float:
    zero_predictions = np.zeros_like(y_true, dtype=np.float64)
    return _compute_rmse(y_true, zero_predictions)


def _feature_priority(feature_name: str) -> tuple[int, str]:
    if feature_name.startswith("ta_"):
        return (0, feature_name)
    if feature_name.startswith("quant_"):
        return (1, feature_name)
    if feature_name.startswith("macro_"):
        return (2, feature_name)
    if feature_name.startswith("calendar_"):
        return (3, feature_name)
    if feature_name.startswith("sentiment_"):
        return (4, feature_name)
    if feature_name.startswith("cross_asset_"):
        return (5, feature_name)
    if feature_name.startswith("company_"):
        return (6, feature_name)
    if feature_name.startswith("stock_"):
        return (8, feature_name)
    return (7, feature_name)


def _select_earliest_train_sample(
    train_scaffold: pd.DataFrame,
    sample_fraction: float = TRAIN_SAMPLE_FRACTION,
) -> pd.DataFrame:
    if train_scaffold.empty:
        raise ValueError("Train scaffold is empty; cannot sample train rows.")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in the interval (0, 1].")
    ordered_train_scaffold = _as_dataframe(
        train_scaffold.sort_values(["date", "ticker"]).reset_index(drop=True),
    )
    sample_size = max(1, int(len(ordered_train_scaffold) * sample_fraction))
    sampled_train_scaffold = _as_dataframe(
        ordered_train_scaffold.iloc[:sample_size].copy().reset_index(drop=True),
    )
    LOGGER.info(
        (
            "Using the earliest %.1f%% of train rows for greedy forward selection: "
            "%d/%d rows kept (%s -> %s)."
        ),
        sample_fraction * 100.0,
        len(sampled_train_scaffold),
        len(ordered_train_scaffold),
        sampled_train_scaffold["date"].min(),
        sampled_train_scaffold["date"].max(),
    )
    return sampled_train_scaffold


def build_candidate_feature_columns(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature parquet not found: {path}")
    schema: pa.Schema = cast(pa.Schema, pq.read_schema(path))
    return sorted(
        (
            column_name
            for column_name in schema.names
            if column_name not in EXCLUDED_FEATURE_COLUMNS
        ),
        key=_feature_priority,
    )


def load_ranked_candidate_feature_columns(
    feature_parquet_path: Path,
    sfi_scores_path: Path | None = None,
) -> list[str]:
    del sfi_scores_path
    return build_candidate_feature_columns(feature_parquet_path)


def load_train_feature_series(
    path: Path,
    train_scaffold: pd.DataFrame,
    feature_name: str,
) -> pd.Series:
    feature_column = _as_series(pd.read_parquet(path, columns=[feature_name])[feature_name])
    aligned_values = feature_column.iloc[train_scaffold["row_position"].to_numpy()].to_numpy()
    return pd.Series(aligned_values, index=train_scaffold.index, name=feature_name)


def score_feature_subset(
    feature_frame: pd.DataFrame,
    feature_names: list[str],
    holdout_window: object,
    model_config: SFIModelConfig | None = None,
) -> dict[str, object]:
    if feature_frame.empty:
        raise ValueError("Feature frame is empty; cannot score subset.")
    config = model_config or SFIModelConfig()
    dates = _as_series(pd.to_datetime(feature_frame["date"]))
    features = _as_dataframe(feature_frame.loc[:, feature_names])
    targets = _as_series(feature_frame[TARGET_COLUMN])

    train_end = cast(pd.Timestamp, getattr(holdout_window, "train_end"))
    validation_start = cast(pd.Timestamp, getattr(holdout_window, "validation_start"))
    validation_end = cast(pd.Timestamp, getattr(holdout_window, "validation_end"))

    train_mask = dates <= train_end
    validation_mask = (dates >= validation_start) & (dates <= validation_end)
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("No valid train/validation rows remained for subset scoring.")

    model = load_catboost_regressor_class()(
        random_seed=config.random_seed,
        allow_writing_files=config.allow_writing_files,
        iterations=config.iterations,
        depth=config.depth,
        learning_rate=config.learning_rate,
        loss_function=config.loss_function,
    )
    model.fit(
        features.loc[train_mask],
        targets.loc[train_mask],
        eval_set=(features.loc[validation_mask], targets.loc[validation_mask]),
        early_stopping_rounds=config.early_stopping_rounds,
        use_best_model=True,
        verbose=False,
    )
    predictions = np.asarray(model.predict(features.loc[validation_mask]), dtype=np.float64)
    y_true = targets.loc[validation_mask].to_numpy(dtype=np.float64)
    return {
        "mean_rmse": _compute_rmse(y_true, predictions),
        "baseline_rmse": _compute_zero_baseline_rmse(y_true),
        "mean_r2": _compute_r2(y_true, predictions),
        "train_row_count": int(train_mask.sum()),
        "validation_row_count": int(validation_mask.sum()),
    }


def create_selected_features_summary(evaluation_scores: pd.DataFrame) -> pd.DataFrame:
    selected_rows = _as_dataframe(
        evaluation_scores.loc[_as_series(evaluation_scores["selected_at_iteration"]).astype(bool)].copy(),
    )
    if selected_rows.empty:
        return selected_rows.reset_index(drop=True)
    return _as_dataframe(selected_rows.sort_values(["iteration"]).reset_index(drop=True))


def create_retained_features_summary(selected_features: pd.DataFrame) -> pd.DataFrame:
    if selected_features.empty:
        return pd.DataFrame(columns=["feature_name"])
    retained = pd.DataFrame({
        "feature_name": _as_series(selected_features[CANDIDATE_FEATURE_NAME_COLUMN]).astype(str),
    })
    return _as_dataframe(retained.sort_values("feature_name").reset_index(drop=True))


def build_filtered_feature_dataset(
    feature_parquet_path: Path,
    retained_features: pd.DataFrame,
    scaffold: pd.DataFrame,
    metadata_columns_to_keep: tuple[str, ...] = ("date", "ticker", "stock_close_price"),
) -> pd.DataFrame:
    if not feature_parquet_path.exists():
        raise FileNotFoundError(f"Feature parquet not found: {feature_parquet_path}")
    retained_feature_names = set(
        _as_series(retained_features["feature_name"]).astype(str).tolist(),
    )
    schema: pa.Schema = cast(pa.Schema, pq.read_schema(feature_parquet_path))
    columns_to_keep = [
        column_name
        for column_name in schema.names
        if column_name in retained_feature_names or column_name in set(metadata_columns_to_keep)
    ]
    feature_dataset = _as_dataframe(pd.read_parquet(feature_parquet_path, columns=columns_to_keep))
    target_frame = _as_dataframe(
        scaffold.loc[:, ["date", "ticker", TARGET_COLUMN]].copy(),
    )
    filtered_dataset = _as_dataframe(
        feature_dataset.merge(target_frame, on=["date", "ticker"], how="inner"),
    )
    LOGGER.info(
        "Built filtered greedy feature dataset: %d rows x %d cols | %d retained features.",
        len(filtered_dataset),
        len(filtered_dataset.columns),
        len(retained_feature_names),
    )
    return filtered_dataset


def _save_greedy_forward_selection_outputs(
    evaluation_scores: pd.DataFrame,
    selected_features: pd.DataFrame,
    retained_features: pd.DataFrame,
    filtered_features: pd.DataFrame,
    scores_parquet_path: Path,
    scores_csv_path: Path,
    selected_features_parquet_path: Path,
    selected_features_csv_path: Path,
    filtered_features_parquet_path: Path,
    filtered_features_csv_path: Path,
) -> None:
    scores_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    scores_csv_path.parent.mkdir(parents=True, exist_ok=True)
    selected_features_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    selected_features_csv_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_features_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_features_csv_path.parent.mkdir(parents=True, exist_ok=True)

    evaluation_scores.to_parquet(scores_parquet_path, index=False)
    evaluation_scores.to_csv(scores_csv_path, index=False)
    selected_features.to_parquet(selected_features_parquet_path, index=False)
    selected_features.to_csv(selected_features_csv_path, index=False)
    retained_features.to_parquet(
        selected_features_parquet_path.with_name("feature_greedy_forward_selection_retained.parquet"),
        index=False,
    )
    retained_features.to_csv(
        selected_features_csv_path.with_name("feature_greedy_forward_selection_retained.csv"),
        index=False,
    )
    filtered_features.to_parquet(filtered_features_parquet_path, index=False)
    filtered_sample = _as_dataframe(
        filtered_features.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True),
    )
    filtered_sample.to_csv(filtered_features_csv_path, index=False)


def run_greedy_forward_selection(
    feature_parquet_path: Path = FEATURE_CORR_PCA_OUTPUT_PARQUET,
    model_config: SFIModelConfig | None = None,
    selection_scaffold: pd.DataFrame | None = None,
    scores_parquet_path: Path | None = None,
    scores_csv_path: Path | None = None,
    selected_features_parquet_path: Path | None = None,
    selected_features_csv_path: Path | None = None,
    filtered_features_parquet_path: Path | None = None,
    filtered_features_csv_path: Path | None = None,
    metadata_columns_to_keep: tuple[str, ...] = ("date", "ticker", "stock_close_price"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline_started_at = time.perf_counter()
    LOGGER.info("Feature greedy forward selection started for %s", feature_parquet_path)
    scores_parquet_path = scores_parquet_path or GREEDY_FORWARD_SELECTION_SCORES_PARQUET
    scores_csv_path = scores_csv_path or GREEDY_FORWARD_SELECTION_SCORES_CSV
    selected_features_parquet_path = (
        selected_features_parquet_path or GREEDY_FORWARD_SELECTION_SELECTED_FEATURES_PARQUET
    )
    selected_features_csv_path = selected_features_csv_path or GREEDY_FORWARD_SELECTION_SELECTED_FEATURES_CSV
    filtered_features_parquet_path = (
        filtered_features_parquet_path or GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_PARQUET
    )
    filtered_features_csv_path = filtered_features_csv_path or GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_CSV
    scaffold = selection_scaffold if selection_scaffold is not None else load_selection_scaffold(feature_parquet_path)
    full_train_scaffold = _as_dataframe(
        scaffold.loc[scaffold["dataset_split"] == TRAIN_SPLIT_NAME].copy(),
    )
    train_scaffold = _select_earliest_train_sample(full_train_scaffold, sample_fraction=TRAIN_SAMPLE_FRACTION)
    LOGGER.info(
        "Sampled train block ready for greedy forward selection: %d rows across %d unique dates.",
        len(train_scaffold),
        _as_series(pd.to_datetime(train_scaffold["date"])).nunique(),
    )
    holdout_window = build_train_holdout_window(train_scaffold, holdout_fraction=TRAIN_HOLDOUT_FRACTION)
    candidate_features = load_ranked_candidate_feature_columns(feature_parquet_path)
    LOGGER.info(
        "Starting greedy forward selection on %d candidate features from %s.",
        len(candidate_features),
        feature_parquet_path,
    )

    base_frame = _as_dataframe(train_scaffold.loc[:, ["row_position", "date", TARGET_COLUMN]].copy())
    validation_mask = (
        (_as_series(pd.to_datetime(base_frame["date"])) >= holdout_window.validation_start)
        & (_as_series(pd.to_datetime(base_frame["date"])) <= holdout_window.validation_end)
    )
    baseline_rmse = _compute_zero_baseline_rmse(
        _as_series(base_frame.loc[validation_mask, TARGET_COLUMN]).to_numpy(dtype=np.float64),
    )
    current_best_rmse = baseline_rmse
    current_feature_frame = _as_dataframe(base_frame.copy())
    selected_feature_names: list[str] = []
    evaluation_rows: list[dict[str, object]] = []
    iteration = 0

    def append_rejection_row(
        candidate_name: str,
        *,
        feature_names: list[str],
        rejection_reason: str,
    ) -> None:
        evaluation_rows.append({
            "iteration": iteration,
            CANDIDATE_FEATURE_NAME_COLUMN: candidate_name,
            "subset_size": len(feature_names),
            "trial_feature_names": "|".join(feature_names),
            "trial_rmse": np.nan,
            "baseline_rmse": baseline_rmse,
            "trial_r2": np.nan,
            "previous_best_rmse": current_best_rmse,
            "marginal_rmse_gain": np.nan,
            "improves_over_previous": False,
            "selected_at_iteration": False,
            "train_row_count": 0,
            "validation_row_count": 0,
            "rejection_reason": rejection_reason,
        })
    
    def evaluate_candidate(candidate_name: str, *, stage_label: str) -> bool:
        nonlocal iteration
        nonlocal current_best_rmse
        nonlocal current_feature_frame

        iteration += 1
        iteration_started_at = time.perf_counter()
        LOGGER.info(
            (
                "Greedy %s iteration %d: testing candidate=%s | "
                "selected=%d | current_best_rmse=%.6f"
            ),
            stage_label,
            iteration,
            candidate_name,
            len(selected_feature_names),
            current_best_rmse,
        )
        candidate_series = load_train_feature_series(feature_parquet_path, train_scaffold, candidate_name)
        feature_names = [*selected_feature_names, candidate_name]
        non_null_candidate_values = candidate_series.dropna()
        if not non_null_candidate_values.empty and non_null_candidate_values.nunique(dropna=True) <= 1:
            append_rejection_row(
                candidate_name,
                feature_names=feature_names,
                rejection_reason="constant_feature",
            )
            LOGGER.info(
                "Rejected %s at %s iteration %d | reason=constant_feature | elapsed=%s",
                candidate_name,
                stage_label,
                iteration,
                _format_duration(time.perf_counter() - iteration_started_at),
            )
            return False
        trial_feature_frame = _as_dataframe(current_feature_frame.copy())
        trial_feature_frame[candidate_name] = candidate_series.to_numpy()
        trial_subset = _as_dataframe(
            trial_feature_frame.dropna(subset=[TARGET_COLUMN, *feature_names]).reset_index(drop=True),
        )

        if trial_subset.empty:
            append_rejection_row(
                candidate_name,
                feature_names=feature_names,
                rejection_reason="empty_feature_frame",
            )
            LOGGER.info(
                "Rejected %s at %s iteration %d | reason=empty_feature_frame | elapsed=%s",
                candidate_name,
                stage_label,
                iteration,
                _format_duration(time.perf_counter() - iteration_started_at),
            )
            return False

        try:
            score = score_feature_subset(
                trial_subset,
                feature_names,
                holdout_window,
                model_config=model_config,
            )
        except Exception as exc:
            if "No valid train/validation rows remained for subset scoring." in str(exc):
                append_rejection_row(
                    candidate_name,
                    feature_names=feature_names,
                    rejection_reason="no_valid_train_validation_rows",
                )
                LOGGER.info(
                    (
                        "Rejected %s at %s iteration %d | "
                        "reason=no_valid_train_validation_rows | elapsed=%s"
                    ),
                    candidate_name,
                    stage_label,
                    iteration,
                    _format_duration(time.perf_counter() - iteration_started_at),
                )
                return False
            if "All features are either constant or ignored" not in str(exc):
                raise
            append_rejection_row(
                candidate_name,
                feature_names=feature_names,
                rejection_reason="catboost_all_features_constant_or_ignored",
            )
            LOGGER.info(
                (
                    "Rejected %s at %s iteration %d | "
                    "reason=catboost_all_features_constant_or_ignored | elapsed=%s"
                ),
                candidate_name,
                stage_label,
                iteration,
                _format_duration(time.perf_counter() - iteration_started_at),
            )
            return False
        trial_rmse = cast(float, score["mean_rmse"])
        marginal_gain = current_best_rmse - trial_rmse
        candidate_selected = bool((current_best_rmse - trial_rmse) > MIN_RMSE_IMPROVEMENT)
        evaluation_rows.append({
            "iteration": iteration,
            CANDIDATE_FEATURE_NAME_COLUMN: candidate_name,
            "subset_size": len(feature_names),
            "trial_feature_names": "|".join(feature_names),
            "trial_rmse": trial_rmse,
            "baseline_rmse": cast(float, score["baseline_rmse"]),
            "trial_r2": cast(float, score["mean_r2"]),
            "previous_best_rmse": current_best_rmse,
            "marginal_rmse_gain": marginal_gain,
            "improves_over_previous": candidate_selected,
            "selected_at_iteration": candidate_selected,
            "train_row_count": cast(int, score["train_row_count"]),
            "validation_row_count": cast(int, score["validation_row_count"]),
            "rejection_reason": "" if candidate_selected else "no_rmse_improvement",
        })

        if not candidate_selected:
            LOGGER.info(
                (
                    "Rejected %s at %s iteration %d | trial_rmse=%.6f | "
                    "current_best_rmse=%.6f | elapsed=%s"
                ),
                candidate_name,
                stage_label,
                iteration,
                trial_rmse,
                current_best_rmse,
                _format_duration(time.perf_counter() - iteration_started_at),
            )
            return False

        selected_feature_names.append(candidate_name)
        current_best_rmse = trial_rmse
        current_feature_frame[candidate_name] = candidate_series.to_numpy()
        LOGGER.info(
            (
                "Selected %s at %s iteration %d | new_best_rmse=%.6f | "
                "gain=%.6f | selected_features=%d | iteration_elapsed=%s"
            ),
            candidate_name,
            stage_label,
            iteration,
            current_best_rmse,
            marginal_gain,
            len(selected_feature_names),
            _format_duration(time.perf_counter() - iteration_started_at),
        )
        return True

    bootstrap_candidates = list(candidate_features)
    while bootstrap_candidates and not selected_feature_names:
        candidate_name = bootstrap_candidates.pop(0)
        evaluate_candidate(candidate_name, stage_label="bootstrap")

    remaining_candidate_features = [
        feature_name for feature_name in candidate_features if feature_name not in selected_feature_names
    ]
    run_index = 1
    while remaining_candidate_features:
        run_candidates = list(remaining_candidate_features)
        remaining_candidate_features = []
        selected_count_before_run = len(selected_feature_names)
        LOGGER.info(
            (
                "Starting greedy run %d: %d remaining candidates | "
                "selected=%d | current_best_rmse=%.6f"
            ),
            run_index,
            len(run_candidates),
            len(selected_feature_names),
            current_best_rmse,
        )
        for candidate_name in run_candidates:
            candidate_selected = evaluate_candidate(candidate_name, stage_label=f"run_{run_index}")
            if not candidate_selected:
                remaining_candidate_features.append(candidate_name)
        new_survivors = len(selected_feature_names) - selected_count_before_run
        LOGGER.info(
            (
                "Completed greedy run %d | new_survivors=%d | "
                "still_unselected=%d | current_best_rmse=%.6f"
            ),
            run_index,
            new_survivors,
            len(remaining_candidate_features),
            current_best_rmse,
        )
        if new_survivors == 0:
            break
        run_index += 1

    evaluation_scores = _as_dataframe(
        pd.DataFrame(evaluation_rows)
        .sort_values(
            ["iteration", "trial_rmse", CANDIDATE_FEATURE_NAME_COLUMN],
            ascending=[True, True, True],
            na_position="last",
        )
        .reset_index(drop=True),
    )
    selected_features = create_selected_features_summary(evaluation_scores)
    retained_features = create_retained_features_summary(selected_features)
    filtered_features = build_filtered_feature_dataset(
        feature_parquet_path,
        retained_features,
        scaffold,
        metadata_columns_to_keep=metadata_columns_to_keep,
    )
    _save_greedy_forward_selection_outputs(
        evaluation_scores,
        selected_features,
        retained_features,
        filtered_features,
        scores_parquet_path=scores_parquet_path,
        scores_csv_path=scores_csv_path,
        selected_features_parquet_path=selected_features_parquet_path,
        selected_features_csv_path=selected_features_csv_path,
        filtered_features_parquet_path=filtered_features_parquet_path,
        filtered_features_csv_path=filtered_features_csv_path,
    )
    LOGGER.info(
        "Feature greedy forward selection completed in %s with %d selected features.",
        _format_duration(time.perf_counter() - pipeline_started_at),
        len(selected_features),
    )
    return evaluation_scores, selected_features


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_greedy_forward_selection()
    LOGGER.info("Feature greedy forward selection pipeline completed.")


__all__ = [
    "build_candidate_feature_columns",
    "build_filtered_feature_dataset",
    "create_retained_features_summary",
    "create_selected_features_summary",
    "load_ranked_candidate_feature_columns",
    "load_selection_scaffold",
    "load_train_feature_series",
    "main",
    "run_greedy_forward_selection",
    "score_feature_subset",
]


if __name__ == "__main__":
    main()
