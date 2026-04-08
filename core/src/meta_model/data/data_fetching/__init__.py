"""Data-fetching pipelines: S&P 500 prices, macro, cross-asset, sentiment, calendar."""

from __future__ import annotations

from .calendar_pipeline import CalendarConfig, build_calendar_dataset
from .cross_asset_pipeline import CrossAssetConfig, build_cross_asset_dataset
from .macro_pipeline import MacroConfig, build_macro_dataset
from .sentiment_pipeline import SentimentConfig, build_sentiment_dataset
from .sp500_pipeline import PipelineConfig, build_dataset, run_pipeline, save_outputs

__all__ = [
    "CalendarConfig",
    "CrossAssetConfig",
    "MacroConfig",
    "PipelineConfig",
    "SentimentConfig",
    "build_calendar_dataset",
    "build_cross_asset_dataset",
    "build_dataset",
    "build_macro_dataset",
    "build_sentiment_dataset",
    "run_pipeline",
    "save_outputs",
]
