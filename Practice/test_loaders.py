"""
Tests for loaders.py — schema-translating data loaders.

Run from the repo root:
    pytest Practice/test_loaders.py
Or from Practice/:
    pytest test_loaders.py
"""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import loaders

DATA_DIR = Path(__file__).parent / "DataParquet"

# Standard internal schema columns that every loader must produce.
REQUIRED_COLS = {
    "Elapsed time (sec)", "Distance (mi)", "Speed (mph)",
    "Latitude", "Longitude", "Time", "Date",
}

# All parquet files — used for parametrized contract tests.
ALL_PARQUETS = sorted(DATA_DIR.glob("*.parquet"))


# ── Individual loader tests ───────────────────────────────────────────────────

class TestLoadSpeedTracker:
    PATH = DATA_DIR / "2026-03-07 speed_tracker_07_Mar_2026_11_23_00.parquet"

    def test_standard_schema_present(self):
        df = loaders.load_speed_tracker(str(self.PATH))
        assert REQUIRED_COLS.issubset(df.columns)

    def test_sorted_and_no_duplicates(self):
        df = loaders.load_speed_tracker(str(self.PATH))
        assert df["Elapsed time (sec)"].is_monotonic_increasing
        assert df["Elapsed time (sec)"].duplicated().sum() == 0

    def test_extra_datalogger_columns_preserved(self):
        # First speed_tracker file has four attached-datalogger temperature channels.
        df = loaders.load_speed_tracker(str(self.PATH))
        assert "Mechanical fuel pump" in df.columns


class TestLoadRaceBox:
    PATH = DATA_DIR / "RaceBox 10-03-2026.parquet"

    def test_standard_schema_present(self):
        df = loaders.load_racebox(str(self.PATH))
        assert REQUIRED_COLS.issubset(df.columns)

    def test_elapsed_starts_at_zero(self):
        df = loaders.load_racebox(str(self.PATH))
        assert df["Elapsed time (sec)"].iloc[0] == pytest.approx(0.0)

    def test_elapsed_monotonic_no_duplicates(self):
        df = loaders.load_racebox(str(self.PATH))
        assert df["Elapsed time (sec)"].is_monotonic_increasing
        assert df["Elapsed time (sec)"].duplicated().sum() == 0

    def test_speed_mph_is_numeric_and_non_negative(self):
        df = loaders.load_racebox(str(self.PATH))
        assert pd.api.types.is_numeric_dtype(df["Speed (mph)"])
        assert (df["Speed (mph)"] >= 0).all()

    def test_distance_non_decreasing(self):
        df = loaders.load_racebox(str(self.PATH))
        assert (df["Distance (mi)"].diff().dropna() >= -1e-9).all()

    def test_gforce_columns_preserved(self):
        df = loaders.load_racebox(str(self.PATH))
        assert {"GForceX", "GForceY", "GForceZ"}.issubset(df.columns)

    def test_raw_speed_column_removed(self):
        # 'Speed' is the raw RaceBox column; it should be replaced by 'Speed (mph)'.
        df = loaders.load_racebox(str(self.PATH))
        assert "Speed" not in df.columns


class TestLoadGeneric:
    PATH = DATA_DIR / "Dan_190SL_Generic testing_a_0035.parquet"

    def test_standard_schema_present(self):
        df = loaders.load_generic(str(self.PATH))
        assert REQUIRED_COLS.issubset(df.columns)

    def test_gps_columns_renamed(self):
        df = loaders.load_generic(str(self.PATH))
        assert "GPS Speed (mph)"     not in df.columns
        assert "GPS Latitude (deg)"  not in df.columns
        assert "GPS Longitude (deg)" not in df.columns
        assert "GPS Altitude (ft)"   not in df.columns

    def test_time_date_string_format(self):
        df = loaders.load_generic(str(self.PATH))
        assert df["Time"].str.match(r"^\d{2}:\d{2}:\d{2}$").all()
        assert df["Date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()

    def test_imu_channels_preserved(self):
        df = loaders.load_generic(str(self.PATH))
        assert {"InlineAcc (g)", "LateralAcc (g)", "VerticalAcc (g)"}.issubset(df.columns)
        assert "RPM (rpm)" in df.columns

    def test_distance_computed_and_positive(self):
        df = loaders.load_generic(str(self.PATH))
        assert df["Distance (mi)"].max() > 0

    def test_elapsed_monotonic_no_duplicates(self):
        df = loaders.load_generic(str(self.PATH))
        assert df["Elapsed time (sec)"].is_monotonic_increasing
        assert df["Elapsed time (sec)"].duplicated().sum() == 0


# ── load_any dispatch tests ───────────────────────────────────────────────────

class TestLoadAnyDispatch:
    def test_dispatches_racebox_by_prefix(self):
        df = loaders.load_any(str(DATA_DIR / "RaceBox 10-03-2026.parquet"))
        assert "GForceX" in df.columns

    def test_dispatches_speed_tracker_by_name(self):
        df = loaders.load_any(str(
            DATA_DIR / "2026-03-07 speed_tracker_07_Mar_2026_11_23_00.parquet"
        ))
        assert "Accuracy (ft)" in df.columns

    def test_dispatches_generic_as_fallback(self):
        df = loaders.load_any(str(DATA_DIR / "Dan_190SL_Generic testing_a_0035.parquet"))
        assert "InlineAcc (g)" in df.columns

    def test_all_dispatches_produce_standard_schema(self):
        sample_files = [
            DATA_DIR / "RaceBox 10-03-2026.parquet",
            DATA_DIR / "2026-03-07 speed_tracker_07_Mar_2026_11_23_00.parquet",
            DATA_DIR / "Dan_190SL_Generic testing_a_0035.parquet",
        ]
        for fpath in sample_files:
            df = loaders.load_any(str(fpath))
            missing = REQUIRED_COLS - set(df.columns)
            assert not missing, f"{fpath.name} missing: {missing}"


# ── Parametrized contract tests across all files ──────────────────────────────

@pytest.mark.parametrize("fpath", ALL_PARQUETS, ids=lambda p: p.name)
def test_required_columns_present(fpath):
    df = loaders.load_any(str(fpath))
    missing = REQUIRED_COLS - set(df.columns)
    assert not missing, f"Missing standard schema columns: {missing}"


@pytest.mark.parametrize("fpath", ALL_PARQUETS, ids=lambda p: p.name)
def test_elapsed_monotonic_no_duplicates(fpath):
    df = loaders.load_any(str(fpath))
    if not df["Elapsed time (sec)"].is_monotonic_increasing:
        pytest.fail("Elapsed time (sec) is not monotonically increasing")
    dup_count = int(df["Elapsed time (sec)"].duplicated().sum())
    if dup_count:
        pytest.fail(f"{dup_count} duplicate Elapsed time (sec) values found")


@pytest.mark.parametrize("fpath", ALL_PARQUETS, ids=lambda p: p.name)
def test_speed_non_negative(fpath):
    df = loaders.load_any(str(fpath))
    neg_count = int((df["Speed (mph)"].dropna() < 0).sum())
    if neg_count:
        pytest.fail(f"{neg_count} negative Speed (mph) values found")


@pytest.mark.parametrize("fpath", ALL_PARQUETS, ids=lambda p: p.name)
def test_distance_non_decreasing(fpath):
    df = loaders.load_any(str(fpath))
    violations = int((df["Distance (mi)"].diff().dropna() < -1e-9).sum())
    if violations:
        pytest.fail(f"{violations} decreasing Distance (mi) steps found")


@pytest.mark.parametrize("fpath", ALL_PARQUETS, ids=lambda p: p.name)
def test_lat_lon_in_valid_range(fpath):
    df = loaders.load_any(str(fpath))
    lat = df["Latitude"].dropna()
    lon = df["Longitude"].dropna()
    if not bool(lat.between(-90, 90).all()):
        pytest.fail("Latitude out of [-90, 90]")
    if not bool(lon.between(-180, 180).all()):
        pytest.fail("Longitude out of [-180, 180]")


@pytest.mark.parametrize("fpath", ALL_PARQUETS, ids=lambda p: p.name)
def test_non_empty(fpath):
    df = loaders.load_any(str(fpath))
    assert len(df) > 0
