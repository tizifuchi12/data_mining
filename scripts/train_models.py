#!/usr/bin/env python3
"""Train baseline models for vehicle energy consumption estimation."""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_ingestion import IngestionConfig, load_dataset
from src.features import build_feature_matrix
from src.modeling import (
    SplitConfig,
    build_model_registry,
    chronological_trip_split,
    save_metrics,
    train_and_evaluate_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-glob",
        type=str,
        default="../fuel_data/*.csv",
        help="Glob pattern to locate Munic CSV exports.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional cap on number of CSV files to ingest (useful for quick runs).",
    )
    parser.add_argument(
        "--resample-rate",
        type=str,
        default="1s",
        help="Resampling frequency passed to IngestionConfig (e.g., '1S', '2S').",
    )
    parser.add_argument(
        "--include-rpm",
        action="store_true",
        help="Include RPM-derived features if available.",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="../outputs/metrics/model_metrics.json",
        help="Path to save evaluation metrics (JSON).",
    )
    return parser.parse_args()


def resolve_paths(glob_pattern: str, limit: int | None) -> List[Path]:
    paths = sorted(Path(p).resolve() for p in glob(glob_pattern))
    if limit is not None:
        paths = paths[:limit]
    return paths


def main() -> None:
    args = parse_args()
    csv_paths = resolve_paths(args.data_glob, args.limit_files)
    if not csv_paths:
        raise SystemExit("No CSV files found; check --data-glob pattern.")

    cfg = IngestionConfig(resample_rate=args.resample_rate)
    dataset = load_dataset(csv_paths, cfg)
    if dataset.empty:
        raise SystemExit("Dataset is empty after ingestion; verify input files.")

    sampling_period_seconds = max(pd.to_timedelta(cfg.resample_rate).total_seconds(), 1.0)
    X, y, processed = build_feature_matrix(
        dataset,
        include_rpm=args.include_rpm,
        sampling_period_seconds=sampling_period_seconds,
    )
    if "time" not in processed.columns:
        processed = processed.reset_index().rename(columns={"index": "time"})
    if "trip_id" not in processed.columns:
        raise SystemExit("Processed dataframe missing 'trip_id'; check ingestion pipeline.")
    train_mask, val_mask, test_mask = chronological_trip_split(
        processed,
        trip_column="trip_id",
        time_column="time",
        config=SplitConfig(),
    )

    registry = build_model_registry()
    metrics = train_and_evaluate_models(registry, X, y, train_mask, val_mask, test_mask)
    output_path = Path(args.output_metrics).resolve()
    save_metrics(metrics, output_path)

    print(f"Saved metrics to {output_path}")
    for model_name, splits in metrics.items():
        val_metrics = splits["validation"]
        test_metrics = splits["test"]
        print(f"{model_name}: val MAE={val_metrics['mae']:.3f}, test MAE={test_metrics['mae']:.3f}")


if __name__ == "__main__":
    main()

