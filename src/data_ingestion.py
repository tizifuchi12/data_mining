"""Utilities for loading and cleaning Munic telematics datasets.

The helpers in this module consolidate heterogeneous CSV exports into a common
schema, align timestamps, and derive core signals (speed, RPM, cumulative fuel
consumption). Downstream code can rely on a tidy dataframe indexed by time with
uniform cadence and standardized column names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

DATE_COLUMNS: Sequence[str] = ("time", "received_at")
GPS_SPEED_COL = "TRACKS.MUNIC.GPS_SPEED (km/h)"
OBD_SPEED_COL = "TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"
FUEL_COLUMNS_PRIORITY: Sequence[str] = (
    "TRACKS.MUNIC.MDI_OBD_FUEL (ml)",
    "TRACKS.MUNIC.MDI_FUEL (ml)",
    "TRACKS.MUNIC.FUEL_USED (ml)",
    "TRACKS.MUNIC.MDI_DASHBOARD_FUEL (l)",
)
FUEL_LEVEL_COLUMNS: Sequence[str] = (
    "TRACKS.MUNIC.MDI_CC_DASHBOARD_FUEL_LEVEL_PERCENT (%)",
    "TRACKS.MUNIC.MDI_DASHBOARD_FUEL_LEVEL (%)",
)
RPM_COLUMNS_PRIORITY: Sequence[str] = (
    "TRACKS.MUNIC.MDI_OBD_ENGINE_SPEED (rpm)",
    "TRACKS.MUNIC.MDI_ENGINE_SPEED (rpm)",
    "TRACKS.MUNIC.MDI_ENGINE_RPM (rpm)",
    "TRACKS.MUNIC.MDI_OBD_ENGINE_RPM (rpm)",
)


@dataclass(slots=True)
class IngestionConfig:
    """Configuration toggles for preprocessing."""

    resample_rate: str = "1s"
    interpolation_limit_seconds: int = 5
    max_speed_kmh: float = 220.0
    min_speed_kmh: float = 0.0
    max_fuel_jump_factor: float = 3.0
    fuel_diff_smoothing_window: int = 5


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_telematics_csv(
    path: Union[str, Path],
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Read a Munic CSV file and perform basic parsing."""

    df = pd.read_csv(path, usecols=usecols)
    df.columns = [c.strip() for c in df.columns]
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    numeric_candidates = [c for c in df.columns if c not in DATE_COLUMNS]
    df[numeric_candidates] = df[numeric_candidates].apply(_coerce_numeric)
    df["source_file"] = Path(path).name
    return df


def _select_first_available(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)
    series = df[available[0]].copy()
    for col in available[1:]:
        series = series.fillna(df[col])
    return series


def _normalize_fuel(series: pd.Series, column_name: str) -> pd.Series:
    """Convert cumulative fuel to milliliters regardless of source units."""

    if series.isna().all():
        return series
    if column_name.strip().endswith("(l)"):
        return series * 1000.0
    return series


def standardize_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Produce canonical columns: speed_kmh, rpm, fuel_ml, fuel_level_pct."""

    df = df.copy()
    df["speed_kmh"] = _select_first_available(df, (GPS_SPEED_COL, OBD_SPEED_COL))
    df["speed_kmh"] = df["speed_kmh"].clip(lower=np.nan)

    rpm_series = _select_first_available(df, RPM_COLUMNS_PRIORITY)
    df["rpm"] = rpm_series

    fuel_series = pd.Series(np.nan, index=df.index)
    for col in FUEL_COLUMNS_PRIORITY:
        if col in df.columns:
            normalized = _normalize_fuel(df[col], col)
            fuel_series = fuel_series.fillna(normalized)
    df["fuel_cumulative_ml"] = fuel_series

    level_series = _select_first_available(df, FUEL_LEVEL_COLUMNS)
    df["fuel_level_pct"] = level_series

    cols_to_keep = [c for c in DATE_COLUMNS if c in df.columns] + [
        "source_file",
        "speed_kmh",
        "rpm",
        "fuel_cumulative_ml",
        "fuel_level_pct",
    ]
    return df[cols_to_keep]


def _remove_outliers(df: pd.DataFrame, cfg: IngestionConfig) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["speed_kmh"].isna()) | ((df["speed_kmh"] >= cfg.min_speed_kmh) & (df["speed_kmh"] <= cfg.max_speed_kmh))]
    return df


def _resample_uniform(
    df: pd.DataFrame,
    cfg: IngestionConfig,
) -> pd.DataFrame:
    df = df.sort_values("time").drop_duplicates(subset="time")
    df = df.set_index("time")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    freq = cfg.resample_rate.lower()
    resampled_numeric = pd.DataFrame(index=pd.DatetimeIndex([]))
    if len(numeric_cols) > 0:
        resampled_numeric = df[numeric_cols].resample(freq).mean()
        resampled_numeric = resampled_numeric.interpolate(
            method="time",
            limit=int(cfg.interpolation_limit_seconds),
        )
    else:
        resampled_numeric = df.resample(freq).asfreq()

    resampled = resampled_numeric
    resampled["source_file"] = df["source_file"].resample(freq).first()
    if "received_at" in df.columns:
        resampled["received_at"] = df["received_at"].resample(freq).first()
    return resampled.reset_index()


def _smooth_diff(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values
    return values.rolling(window=window, min_periods=1, center=True).median()


def derive_consumption_rate(
    df: pd.DataFrame,
    cfg: IngestionConfig,
) -> pd.DataFrame:
    df = df.copy()
    if "fuel_cumulative_ml" not in df.columns:
        df["consumption_ml_per_s"] = np.nan
        return df

    cumulative = df["fuel_cumulative_ml"].ffill()
    diff_raw = cumulative.diff()
    cumulative = cumulative.mask(diff_raw < 0).ffill()
    df["fuel_cumulative_ml"] = cumulative
    diff = cumulative.diff()
    positive_diff = diff.clip(lower=0)
    smoothed = _smooth_diff(positive_diff, cfg.fuel_diff_smoothing_window)
    freq_seconds = max(pd.to_timedelta(cfg.resample_rate).total_seconds(), 1.0)
    rate = smoothed / freq_seconds
    df["consumption_ml_per_s"] = rate
    df["consumption_ml_per_s"] = df["consumption_ml_per_s"].replace([np.inf, -np.inf], np.nan)
    df["consumption_ml_per_s"] = df["consumption_ml_per_s"].clip(lower=0)
    return df


def process_file(path: Union[str, Path], cfg: Optional[IngestionConfig] = None) -> pd.DataFrame:
    cfg = cfg or IngestionConfig()
    raw = read_telematics_csv(path)
    standardized = standardize_signals(raw)
    cleaned = _remove_outliers(standardized, cfg)
    cleaned = cleaned.dropna(subset=["time"])
    resampled = _resample_uniform(cleaned, cfg)
    enriched = derive_consumption_rate(resampled, cfg)
    enriched["trip_id"] = Path(path).stem
    return enriched


def load_dataset(
    paths: Iterable[Union[str, Path]],
    cfg: Optional[IngestionConfig] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = process_file(path, cfg)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    dataset = pd.concat(frames, ignore_index=True).sort_values(["trip_id", "time"])
    return dataset.reset_index(drop=True)


