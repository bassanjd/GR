"""
loaders.py — Schema-translating data loaders for Great Race practice parquet files.

Each loader normalises a source-specific format to the standard internal schema
defined in telemetry.py:

    Elapsed time (sec)  — time axis (seconds from first record)
    Distance (mi)       — cumulative GPS odometer (haversine)
    Speed (mph)         — vehicle speed
    Latitude, Longitude — GPS position (decimal degrees)
    Time                — timestamp string (HH:MM:SS)
    Date                — date string (YYYY-MM-DD)

Adding a new source
-------------------
1. Write load_<source>(path: str) -> pd.DataFrame that produces the schema above.
   Extra source-specific columns may be kept; the analysis layer ignores them.
2. Add a detection branch in load_any() based on the filename pattern.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from telemetry import haversine_cumulative_mi


def load_racebox(path: str) -> pd.DataFrame:
    """Load a RaceBox parquet file and normalise to the standard internal schema.

    RaceBox columns used:
        Time      — ISO-8601 string (e.g. '2026-03-10T18:03:09.600')
        Latitude  — decimal degrees
        Longitude — decimal degrees
        Speed     — mph

    Derived columns added:
        Elapsed time (sec), Distance (mi), Speed (mph), Date
    All original columns are preserved.
    """
    df = pd.read_parquet(Path(path))

    df["_ts"] = pd.to_datetime(df["Time"])
    df = df.sort_values("_ts").reset_index(drop=True)
    df["Elapsed time (sec)"] = (df["_ts"] - df["_ts"].iloc[0]).dt.total_seconds()
    df = df.drop_duplicates(subset="Elapsed time (sec)", keep="first").reset_index(drop=True)

    df.loc[df["Latitude"] == 0, "Latitude"] = np.nan
    df.loc[df["Longitude"] == 0, "Longitude"] = np.nan
    df["Latitude"]  = df["Latitude"].interpolate()
    df["Longitude"] = df["Longitude"].interpolate()

    df["Distance (mi)"] = haversine_cumulative_mi(
        df["Latitude"].to_numpy(), df["Longitude"].to_numpy()
    )
    df["Speed (mph)"] = df["Speed"]
    df["Date"]        = df["_ts"].dt.strftime("%Y-%m-%d")
    df["Time"]        = df["_ts"].dt.strftime("%H:%M:%S")
    df = df.drop(columns=["_ts", "Speed"])
    return df.reset_index(drop=True)


def load_speed_tracker(path: str) -> pd.DataFrame:
    """Load a speed_tracker parquet or CSV and normalise to the standard schema.

    Speed-tracker files are pre-processed and already carry the standard columns
    (Elapsed time (sec), Distance (mi), Speed (mph), Latitude, Longitude).
    This function only sorts and deduplicates.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, skipinitialspace=True)
        df.columns = df.columns.str.strip()
    df = df.sort_values("Elapsed time (sec)")
    df = df.drop_duplicates(subset="Elapsed time (sec)", keep="first")
    return df.reset_index(drop=True)


def load_generic(path: str) -> pd.DataFrame:
    """Load a Generic datalogger parquet and normalise to the standard internal schema.

    Column renaming applied (originals replaced by standard names):
        GPS Speed (mph)     → Speed (mph)
        GPS Latitude (deg)  → Latitude
        GPS Longitude (deg) → Longitude
        GPS Altitude (ft)   → Altitude (ft)
        Datetime            → Time (HH:MM:SS string) + Date (YYYY-MM-DD string)

    Distance (mi) is computed via haversine from the GPS coordinates.
    All other columns are preserved as-is.
    """
    df = pd.read_parquet(Path(path))
    df = (
        df.sort_values("Elapsed time (sec)")
          .drop_duplicates(subset="Elapsed time (sec)", keep="first")
          .reset_index(drop=True)
    )

    rename_map: dict[str, str] = {
        "GPS Speed (mph)":    "Speed (mph)",
        "GPS Latitude (deg)": "Latitude",
        "GPS Longitude (deg)":"Longitude",
    }
    if "GPS Altitude (ft)" in df.columns:
        rename_map["GPS Altitude (ft)"] = "Altitude (ft)"
    df = df.rename(columns=rename_map)

    df["Time"] = df["Datetime"].dt.strftime("%H:%M:%S")
    df["Date"] = df["Datetime"].dt.strftime("%Y-%m-%d")

    df.loc[df["Latitude"] == 0, "Latitude"] = np.nan
    df.loc[df["Longitude"] == 0, "Longitude"] = np.nan
    df["Latitude"]  = df["Latitude"].interpolate()
    df["Longitude"] = df["Longitude"].interpolate()

    df["Distance (mi)"] = haversine_cumulative_mi(
        df["Latitude"].to_numpy(), df["Longitude"].to_numpy()
    )
    return df.reset_index(drop=True)


def load_any(path: str) -> pd.DataFrame:
    """Dispatch to the correct loader based on the filename.

    RaceBox*        → load_racebox
    *speed_tracker* → load_speed_tracker
    Everything else → load_generic
    """
    name = Path(path).name
    if name.startswith("RaceBox"):
        return load_racebox(path)
    if "speed_tracker" in name.lower():
        return load_speed_tracker(path)
    return load_generic(path)
