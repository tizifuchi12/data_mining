"""Model training utilities for vehicle consumption estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None


@dataclass(slots=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15


def chronological_trip_split(
    df: pd.DataFrame,
    trip_column: str = "trip_id",
    time_column: str = "time",
    config: Optional[SplitConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = config or SplitConfig()
    trip_order = (
        df.groupby(trip_column)[time_column]
        .min()
        .sort_values()
        .index.tolist()
    )
    n_trips = len(trip_order)
    train_cut = int(n_trips * config.train_ratio)
    val_cut = int(n_trips * (config.train_ratio + config.val_ratio))
    train_trips = set(trip_order[:train_cut])
    val_trips = set(trip_order[train_cut:val_cut])
    test_trips = set(trip_order[val_cut:])
    trip_values = df[trip_column].to_numpy()
    train_mask = np.isin(trip_values, list(train_trips))
    val_mask = np.isin(trip_values, list(val_trips))
    test_mask = np.isin(trip_values, list(test_trips))
    return train_mask, val_mask, test_mask


def build_model_registry(random_state: int = 42) -> Dict[str, Pipeline]:
    registry: Dict[str, Pipeline] = {}
    registry["ridge"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", RidgeCV(alphas=(0.1, 1.0, 10.0), scoring="neg_mean_absolute_error")),
        ]
    )
    registry["elastic_net"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=50, cv=5)),
        ]
    )
    registry["random_forest"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1)),
        ]
    )
    registry["hist_gbrt"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(random_state=random_state, max_depth=8, learning_rate=0.05)),
        ]
    )
    if LGBMRegressor is not None:
        registry["lightgbm"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                    ),
                )
            ]
        )
    return registry


def maape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    denominator = np.maximum(np.abs(y_true), eps)
    return np.mean(np.arctan(np.abs((y_true - y_pred) / denominator)))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    maape_score = maape(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "maape": float(maape_score),
    }


def train_and_evaluate_models(
    registry: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    for name, pipeline in registry.items():
        pipeline.fit(X_train, y_train)
        metrics[name] = {
            "validation": evaluate_predictions(y_val, pipeline.predict(X_val)),
            "test": evaluate_predictions(y_test, pipeline.predict(X_test)),
        }
    return metrics


def save_metrics(metrics: Dict[str, Dict[str, Dict[str, float]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


