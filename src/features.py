"""Feature engineering for vehicle energy consumption modeling."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_SPEED_WINDOWS: Sequence[int] = (3, 10, 30, 60)
DEFAULT_RPM_WINDOWS: Sequence[int] = (3, 10, 30)


def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" not in df.columns:
            raise ValueError("Dataframe must contain a 'time' column.")
        df = df.set_index(pd.DatetimeIndex(df["time"]))
    return df.sort_index()


def _diff_series(series: pd.Series, dt_seconds: float) -> pd.Series:
    return series.diff() / dt_seconds


def add_speed_features(
    df: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_SPEED_WINDOWS,
    sampling_period_seconds: float = 1.0,
) -> pd.DataFrame:
    df = _ensure_time_index(df.copy())
    speed_kmh = df["speed_kmh"].fillna(0)
    speed_mps = speed_kmh * (1000.0 / 3600.0)
    df["speed_mps"] = speed_mps
    df["acceleration_mps2"] = _diff_series(speed_mps, sampling_period_seconds).fillna(0)
    df["brake_flag"] = (df["acceleration_mps2"] < -0.7).astype(float)
    df["throttle_flag"] = (df["acceleration_mps2"] > 0.7).astype(float)

    for window in windows:
        rolling = speed_mps.rolling(window=window, min_periods=1)
        df[f"speed_mps_mean_{window}s"] = rolling.mean()
        df[f"speed_mps_std_{window}s"] = rolling.std(ddof=0)
        df[f"accel_mps2_mean_{window}s"] = df["acceleration_mps2"].rolling(window=window, min_periods=1).mean()

    return df


def add_rpm_features(
    df: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_RPM_WINDOWS,
    sampling_period_seconds: float = 1.0,
) -> pd.DataFrame:
    df = _ensure_time_index(df.copy())
    if "rpm" not in df.columns:
        df["rpm"] = np.nan
    rpm = df["rpm"].ffill()
    df["rpm"] = rpm
    df["rpm_change_per_s"] = _diff_series(rpm, sampling_period_seconds).fillna(0)
    for window in windows:
        rolling = rpm.rolling(window=window, min_periods=1)
        df[f"rpm_mean_{window}s"] = rolling.mean()
        df[f"rpm_std_{window}s"] = rolling.std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["speed_per_rpm"] = (df["speed_mps"]) / rpm.replace(0, np.nan)
    return df


def add_target_columns(
    df: pd.DataFrame,
    consumption_col: str = "consumption_ml_per_s",
) -> pd.DataFrame:
    df = df.copy()
    df["target_ml_per_s"] = df[consumption_col]
    df["target_l_per_100km"] = (
        df["target_ml_per_s"] * 3.6 / df["speed_kmh"].replace(0, np.nan)
    )
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    include_rpm: bool = False,
    sampling_period_seconds: float = 1.0,
    drop_na_target: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df.copy()
    df = add_speed_features(df, sampling_period_seconds=sampling_period_seconds)
    if include_rpm:
        df = add_rpm_features(df, sampling_period_seconds=sampling_period_seconds)
    df = add_target_columns(df)
    feature_columns = [
        col
        for col in df.columns
        if any(
            col.startswith(prefix)
            for prefix in (
                "speed_mps",
                "speed_per_rpm",
                "accel_mps2",
                "brake_flag",
                "throttle_flag",
                "rpm",
            )
        )
        or col
        in (
            "speed_kmh",
        )
    ]
    feature_columns = sorted(set(feature_columns))
    X = df[feature_columns]
    y = df["target_ml_per_s"]
    processed = df.copy()
    mask = y.notna()
    if drop_na_target:
        X = X[mask]
        y = y[mask]
        processed = processed.loc[mask]
    return X, y, processed


