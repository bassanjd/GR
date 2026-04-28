"""Tests for telemetry.py — GPS telemetry analysis module."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from telemetry import (
    G_FT_S2,
    MPH_S_TO_FT_S2,
    TrapConfig,
    _pct_range,
    _sym_range,
    compute_derivatives,
    find_crossing_dist,
    get_run_timing_refs,
    haversine_cumulative_mi,
    load_racebox,
    load_speed_tracker,
    run_finish_type,
    run_start_type,
    segment_exercise_runs,
    smooth_series,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_trap():
    """East-west trap at 35°N — ux=1.0, uy=0.0, clean geometry."""
    return TrapConfig(
        east_lon=-97.000,
        west_lon=-97.020,
        east_lat=35.000,
        west_lat=35.000,
        dist_mi=1.0,
    )


def _make_run_df(lons, lats, speeds, elapsed=None, dist=None):
    """Build a minimal telemetry DataFrame."""
    n = len(lons)
    if elapsed is None:
        elapsed = np.arange(n, dtype=float)
    if dist is None:
        dist = np.linspace(0.0, 1.5, n)
    return pd.DataFrame({
        "Longitude":          np.asarray(lons, dtype=float),
        "Latitude":           np.asarray(lats, dtype=float),
        "Speed (mph)":        np.asarray(speeds, dtype=float),
        "Elapsed time (sec)": np.asarray(elapsed, dtype=float),
        "Distance (mi)":      np.asarray(dist, dtype=float),
        "Time":               [f"T{i}" for i in range(n)],
    })


# ── Physics constants ─────────────────────────────────────────────────────────

def test_mph_s_to_ft_s2():
    assert MPH_S_TO_FT_S2 == pytest.approx(5280.0 / 3600.0)


def test_g_ft_s2():
    assert G_FT_S2 == pytest.approx(32.174)


# ── TrapConfig ────────────────────────────────────────────────────────────────

class TestTrapConfig:
    def test_east_west_trap_unit_vector(self, flat_trap):
        assert flat_trap.ux == pytest.approx(1.0)
        assert flat_trap.uy == pytest.approx(0.0)

    def test_chord_mi_positive(self, flat_trap):
        assert flat_trap.chord_mi > 0

    def test_mi_per_deg_lon_at_35n(self, flat_trap):
        expected = 69.172 * math.cos(math.radians(35.0))
        assert flat_trap.mi_per_deg_lon == pytest.approx(expected, rel=1e-6)

    def test_diagonal_trap_unit_vector_is_normalised(self):
        trap = TrapConfig(
            east_lon=-97.000, west_lon=-97.010,
            east_lat=35.001,  west_lat=35.000,
        )
        magnitude = math.sqrt(trap.ux ** 2 + trap.uy ** 2)
        assert magnitude == pytest.approx(1.0, abs=1e-9)

    def test_degenerate_same_coordinates(self):
        """Coincident gates fall back to ux=1, uy=0 without divide-by-zero."""
        trap = TrapConfig(
            east_lon=-97.0, west_lon=-97.0,
            east_lat=35.0,  west_lat=35.0,
        )
        assert trap.chord_mi == pytest.approx(0.0)
        assert trap.ux == pytest.approx(1.0)
        assert trap.uy == pytest.approx(0.0)

    def test_chord_mi_matches_manual_calculation(self, flat_trap):
        expected_chord = 0.020 * flat_trap.mi_per_deg_lon
        assert flat_trap.chord_mi == pytest.approx(expected_chord, rel=1e-6)


# ── smooth_series ─────────────────────────────────────────────────────────────

class TestSmoothSeries:
    def test_window_one_is_identity(self):
        arr = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
        np.testing.assert_array_almost_equal(smooth_series(arr, 1), arr)

    def test_smoothing_reduces_variance(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(100)
        assert smooth_series(noise, 9).std() < noise.std()

    def test_output_length_matches_input(self):
        arr = np.linspace(0, 10, 50)
        assert len(smooth_series(arr, 5)) == 50

    def test_single_element(self):
        assert smooth_series(np.array([7.0]), 5)[0] == pytest.approx(7.0)

    def test_constant_signal_unchanged(self):
        arr = np.full(20, 42.0)
        np.testing.assert_allclose(smooth_series(arr, 7), arr)


# ── compute_derivatives ───────────────────────────────────────────────────────

class TestComputeDerivatives:
    def _df(self, speeds, times=None):
        n = len(speeds)
        if times is None:
            times = np.arange(n, dtype=float)
        return pd.DataFrame({
            "Speed (mph)":        np.asarray(speeds, dtype=float),
            "Elapsed time (sec)": np.asarray(times, dtype=float),
        })

    def test_output_columns_present(self):
        df = compute_derivatives(self._df([30.0] * 20), smooth_window=3)
        for col in ("speed_smooth", "accel_ft_s2", "accel_g", "jerk_ft_s3"):
            assert col in df.columns

    def test_constant_speed_accel_near_zero(self):
        df = compute_derivatives(self._df([35.0] * 40), smooth_window=3)
        assert np.nanmean(np.abs(df["accel_ft_s2"])) == pytest.approx(0.0, abs=1e-6)

    def test_linear_ramp_accel_consistent(self):
        """1 mph/s ramp → interior accel_ft_s2 ≈ MPH_S_TO_FT_S2."""
        times = np.arange(50, dtype=float)
        speeds = times * 1.0
        df = compute_derivatives(self._df(speeds, times), smooth_window=3)
        interior = df["accel_ft_s2"].iloc[5:-5]
        assert interior.values == pytest.approx(MPH_S_TO_FT_S2, rel=0.05)

    def test_accel_g_consistent_with_accel_ft_s2(self):
        speeds = np.linspace(0, 50, 40)
        df = compute_derivatives(self._df(speeds), smooth_window=3)
        ratio = df["accel_ft_s2"] / df["accel_g"]
        finite = ratio[np.isfinite(ratio)]
        assert finite.values == pytest.approx(G_FT_S2, rel=1e-5)

    def test_time_gap_sets_accel_nan(self):
        """Time gap > 5 s should produce NaN accel at the gap row."""
        times = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        df = compute_derivatives(self._df([30.0] * 6, times), smooth_window=1)
        assert np.isnan(df["accel_ft_s2"].iloc[3])

    def test_time_gap_sets_jerk_nan(self):
        """Jerk must also be NaN at the gap row."""
        times = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        df = compute_derivatives(self._df([30.0, 31.0, 32.0, 33.0, 34.0, 35.0], times),
                                 smooth_window=1)
        assert np.isnan(df["jerk_ft_s3"].iloc[3])

    def test_isolated_zero_speed_interpolated(self):
        """A zero-speed spike between non-zero values should be smoothed away."""
        df = compute_derivatives(self._df([30.0, 30.0, 0.0, 30.0, 30.0]), smooth_window=1)
        assert df["speed_smooth"].iloc[2] > 0.0

    def test_input_df_not_mutated(self):
        original = self._df([35.0] * 20)
        cols_before = set(original.columns)
        compute_derivatives(original, smooth_window=3)
        assert set(original.columns) == cols_before


# ── haversine_cumulative_mi ───────────────────────────────────────────────────

class TestHaversineCumulativeMi:
    def test_starts_at_zero(self):
        result = haversine_cumulative_mi(
            np.array([35.0, 35.1, 35.2]),
            np.array([-97.0, -97.0, -97.0]),
        )
        assert result[0] == pytest.approx(0.0, abs=1e-9)

    def test_one_degree_latitude_is_approx_69_miles(self):
        result = haversine_cumulative_mi(
            np.array([35.0, 36.0]),
            np.array([-97.0, -97.0]),
        )
        assert result[-1] == pytest.approx(69.0, rel=0.01)

    def test_monotone_increasing_for_forward_path(self):
        result = haversine_cumulative_mi(
            np.array([35.00, 35.01, 35.02, 35.03]),
            np.array([-97.00, -97.01, -97.02, -97.03]),
        )
        assert np.all(np.diff(result) >= 0)

    def test_nan_dropout_does_not_poison_cumsum(self):
        """Interior NaN (GPS lock lost) must not make downstream distances NaN."""
        result = haversine_cumulative_mi(
            np.array([35.00, np.nan, np.nan, 35.03]),
            np.array([-97.00, np.nan, np.nan, -97.03]),
        )
        assert np.all(np.isfinite(result))

    def test_dropout_steps_are_zero_not_advancing(self):
        """NaN steps contribute zero distance; caller (load_racebox) interpolates first."""
        result = haversine_cumulative_mi(
            np.array([35.00, np.nan, 35.02]),
            np.array([-97.00, np.nan, -97.02]),
        )
        # Both NaN steps produce 0 distance — cumsum stalls through the gap
        assert result[1] == pytest.approx(result[0])
        assert result[2] == pytest.approx(result[0])

    def test_single_point(self):
        result = haversine_cumulative_mi(np.array([35.0]), np.array([-97.0]))
        assert result[0] == pytest.approx(0.0)


# ── find_crossing_dist ────────────────────────────────────────────────────────

class TestFindCrossingDist:
    def test_west_crossing_interpolated(self, flat_trap):
        """Car straddles the east gate going west → crossing at midpoint distance."""
        grp = _make_run_df(
            lons=[-96.999, -97.001],
            lats=[35.0, 35.0],
            speeds=[35.0, 35.0],
            elapsed=[0.0, 1.0],
            dist=[0.0, 0.1],
        )
        result = find_crossing_dist(grp, flat_trap.east_lon, going_west=True, trap=flat_trap)
        assert result == pytest.approx(0.05, abs=1e-6)

    def test_east_crossing_interpolated(self, flat_trap):
        """Car straddles the west gate going east → crossing at midpoint distance."""
        grp = _make_run_df(
            lons=[-97.021, -97.019],
            lats=[35.0, 35.0],
            speeds=[35.0, 35.0],
            elapsed=[0.0, 1.0],
            dist=[0.0, 0.1],
        )
        result = find_crossing_dist(grp, flat_trap.west_lon, going_west=False, trap=flat_trap)
        assert result == pytest.approx(0.05, abs=1e-6)

    def test_returns_none_when_no_crossing(self, flat_trap):
        """Car stays entirely east of the gate — crossing never reached."""
        grp = _make_run_df(
            lons=[-96.990, -96.995],
            lats=[35.0, 35.0],
            speeds=[35.0, 35.0],
            elapsed=[0.0, 1.0],
            dist=[0.0, 0.1],
        )
        result = find_crossing_dist(grp, flat_trap.east_lon, going_west=True, trap=flat_trap)
        assert result is None

    def test_point_on_gate_returns_first_dist(self, flat_trap):
        """First GPS point exactly on the gate (s=0) returns dist[0]."""
        grp = _make_run_df(
            lons=[-97.000, -97.001],
            lats=[35.0, 35.0],
            speeds=[35.0, 35.0],
            elapsed=[0.0, 1.0],
            dist=[0.5, 0.6],
        )
        result = find_crossing_dist(grp, flat_trap.east_lon, going_west=True, trap=flat_trap)
        assert result == pytest.approx(0.5)

    def test_crossing_at_quarter_point(self, flat_trap):
        """Three-point path where crossing is 1/4 of the way between two samples."""
        # s values: [+3, -1] → frac = 3/(3+1) = 0.75
        # crossing = 0.0 + 0.75 * 0.2 = 0.15
        m = flat_trap.mi_per_deg_lon
        lon_offset = 3.0 / m   # +3 mi east of gate
        lon_past = -1.0 / m    # -1 mi west of gate
        grp = _make_run_df(
            lons=[-97.000 + lon_offset, -97.000 + lon_past],
            lats=[35.0, 35.0],
            speeds=[35.0, 35.0],
            elapsed=[0.0, 1.0],
            dist=[0.0, 0.2],
        )
        result = find_crossing_dist(grp, flat_trap.east_lon, going_west=True, trap=flat_trap)
        assert result == pytest.approx(0.15, abs=1e-6)


# ── segment_exercise_runs ─────────────────────────────────────────────────────

class TestSegmentExerciseRuns:
    LAT_THRESH = 35.05

    def _south_df(self, n=40, lat=35.0, lon_start=-97.020, lon_end=-97.000,
                  speed=35.0, elapsed_start=0.0):
        elapsed = elapsed_start + np.arange(n, dtype=float)
        return pd.DataFrame({
            "Latitude":           np.full(n, lat),
            "Longitude":          np.linspace(lon_start, lon_end, n),
            "Speed (mph)":        np.full(n, float(speed)),
            "Elapsed time (sec)": elapsed,
            "Distance (mi)":      np.linspace(0.0, 1.2, n),
            "Time":               [f"T{i}" for i in range(n)],
        })

    def test_empty_df_returns_empty_list(self):
        df = pd.DataFrame(columns=["Latitude", "Longitude", "Speed (mph)",
                                   "Elapsed time (sec)", "Distance (mi)", "Time"])
        assert segment_exercise_runs(df, self.LAT_THRESH, 0.2, 5) == []

    def test_all_points_north_of_threshold_returns_empty(self):
        df = self._south_df(lat=35.10)
        assert segment_exercise_runs(df, self.LAT_THRESH, 0.2, 5) == []

    def test_single_eastbound_run(self):
        df = self._south_df(lon_start=-97.020, lon_end=-97.000)
        runs = segment_exercise_runs(df, self.LAT_THRESH, 0.2, 5)
        assert len(runs) == 1
        assert runs[0]["direction"] == "East"

    def test_single_westbound_run(self):
        df = self._south_df(lon_start=-97.000, lon_end=-97.020)
        runs = segment_exercise_runs(df, self.LAT_THRESH, 0.2, 5)
        assert len(runs) == 1
        assert runs[0]["direction"] == "West"

    def test_time_gap_splits_into_two_runs(self):
        df1 = self._south_df(n=30, elapsed_start=0.0)
        df2 = self._south_df(n=30, elapsed_start=100.0)  # 70 s gap
        combined = pd.concat([df1, df2]).reset_index(drop=True)
        assert len(segment_exercise_runs(combined, self.LAT_THRESH, 0.2, 5)) == 2

    def test_segment_below_min_rows_excluded(self):
        df = self._south_df(n=3)
        assert segment_exercise_runs(df, self.LAT_THRESH, 0.2, 5) == []

    def test_target_speed_is_mode_rounded_to_5(self):
        # 37 mph rounds to 35 (nearest 5)
        df = self._south_df(speed=37.0)
        runs = segment_exercise_runs(df, self.LAT_THRESH, 0.2, 5)
        assert runs[0]["target_speed"] == 35

    def test_target_speed_zero_segment_skipped(self):
        # speed=0 → target=0 → should be filtered out
        df = self._south_df(speed=0.0)
        assert segment_exercise_runs(df, self.LAT_THRESH, 0.2, 5) == []

    def test_run_dict_has_required_keys(self):
        run = segment_exercise_runs(self._south_df(), self.LAT_THRESH, 0.2, 5)[0]
        for key in ("seg_id", "start_elapsed", "end_elapsed", "direction",
                    "mean_speed", "target_speed", "max_speed", "distance_mi",
                    "n_rows", "data"):
            assert key in run

    def test_run_data_is_dataframe(self):
        run = segment_exercise_runs(self._south_df(), self.LAT_THRESH, 0.2, 5)[0]
        assert isinstance(run["data"], pd.DataFrame)

    def test_with_trap_filters_by_longitude(self, flat_trap):
        """Points outside trap longitude bounds should be excluded."""
        df_outside = self._south_df(lon_start=-96.900, lon_end=-96.850)  # east of trap
        assert segment_exercise_runs(df_outside, self.LAT_THRESH, 0.2, 5, trap=flat_trap) == []


# ── _sym_range / _pct_range ───────────────────────────────────────────────────

class TestRangeHelpers:
    def test_sym_range_lo_below_min_hi_above_max(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = _sym_range(arr)
        assert lo < 1.0
        assert hi > 5.0

    def test_sym_range_zero_flag_includes_zero(self):
        arr = np.array([1.0, 2.0, 3.0])
        lo, _ = _sym_range(arr, zero=True)
        assert lo <= 0.0

    def test_sym_range_all_non_finite_returns_min_half(self):
        lo, hi = _sym_range(np.array([np.nan, np.inf, -np.inf]))
        assert lo == pytest.approx(-0.5)
        assert hi == pytest.approx(0.5)

    def test_pct_range_clips_outliers(self):
        # 48 values near 1.0, two extreme outliers
        arr = np.concatenate([np.ones(48), [1000.0, -1000.0]])
        lo, hi = _pct_range(arr, pct=2, zero=False)
        assert hi < 100.0
        assert lo > -100.0

    def test_pct_range_all_non_finite_returns_min_half(self):
        lo, hi = _pct_range(np.array([np.nan, np.inf]))
        assert lo == pytest.approx(-1.0)
        assert hi == pytest.approx(1.0)

    def test_pct_range_zero_flag_includes_zero(self):
        arr = np.array([1.0, 2.0, 3.0])
        lo, _ = _pct_range(arr, zero=True)
        assert lo <= 0.0


# ── run_start_type / run_finish_type ──────────────────────────────────────────

class TestRunClassification:
    # n=500 ensures the flying_window_mi (100 ft ≈ 0.019 mi) contains several
    # samples at the approach/departure speeds (sample spacing ≈ 0.003 mi).
    N = 500
    # First/last 125 samples cover the approach/departure regions on each side
    # of both trap gates.
    FLANK = 125

    def _make_run(self, direction, target_speed, approach_speed, exit_speed):
        n = self.N
        if direction == "West":
            lons = np.linspace(-96.990, -97.030, n)
        else:
            lons = np.linspace(-97.030, -96.990, n)
        speeds = np.full(n, float(target_speed))
        speeds[:self.FLANK] = approach_speed
        speeds[-self.FLANK:] = exit_speed
        data = pd.DataFrame({
            "Longitude":          lons,
            "Latitude":           np.full(n, 35.0),
            "Speed (mph)":        speeds,
            "Elapsed time (sec)": np.arange(n, dtype=float),
            "Distance (mi)":      np.linspace(0.0, 1.5, n),
            "Time":               [f"T{i}" for i in range(n)],
        })
        return {"direction": direction, "target_speed": target_speed, "data": data}

    def test_flying_start(self, flat_trap):
        run = self._make_run("West", 35, approach_speed=33, exit_speed=35)
        assert run_start_type(run, flat_trap, smooth_window=3) == "Flying"

    def test_standing_start(self, flat_trap):
        run = self._make_run("West", 35, approach_speed=5, exit_speed=35)
        assert run_start_type(run, flat_trap, smooth_window=3) == "Standing"

    def test_flying_finish(self, flat_trap):
        run = self._make_run("West", 35, approach_speed=35, exit_speed=33)
        assert run_finish_type(run, flat_trap, smooth_window=3) == "Flying"

    def test_standing_finish(self, flat_trap):
        run = self._make_run("West", 35, approach_speed=35, exit_speed=5)
        assert run_finish_type(run, flat_trap, smooth_window=3) == "Standing"

    def test_eastbound_flying_start(self, flat_trap):
        run = self._make_run("East", 35, approach_speed=33, exit_speed=35)
        assert run_start_type(run, flat_trap, smooth_window=3) == "Flying"

    def test_eastbound_standing_start(self, flat_trap):
        run = self._make_run("East", 35, approach_speed=5, exit_speed=35)
        assert run_start_type(run, flat_trap, smooth_window=3) == "Standing"


# ── get_run_timing_refs ───────────────────────────────────────────────────────

class TestGetRunTimingRefs:
    def _flying_run(self):
        """West-going run at constant speed that crosses both trap gates."""
        n = 500
        return {
            "direction":    "West",
            "target_speed": 35,
            "data": pd.DataFrame({
                "Longitude":          np.linspace(-96.990, -97.030, n),
                "Latitude":           np.full(n, 35.0),
                "Speed (mph)":        np.full(n, 35.0),
                "Elapsed time (sec)": np.arange(n, dtype=float),
                "Distance (mi)":      np.linspace(0.0, 1.5, n),
                "Time":               [f"T{i}" for i in range(n)],
            }),
        }

    def test_flying_returns_two_distances(self, flat_trap):
        entry_d, exit_d = get_run_timing_refs(
            self._flying_run(), flat_trap, smooth_window=3,
            start_type="Flying", finish_type="Flying",
        )
        assert entry_d is not None
        assert exit_d is not None

    def test_flying_exit_greater_than_entry(self, flat_trap):
        entry_d, exit_d = get_run_timing_refs(
            self._flying_run(), flat_trap, smooth_window=3,
            start_type="Flying", finish_type="Flying",
        )
        assert exit_d > entry_d

    def test_flying_span_matches_odometer_between_crossings(self, flat_trap):
        """Entry/exit crossings land at 25%/75% of the run → span ≈ 0.75 mi."""
        # Run spans lon -96.990 to -97.030 (0.040°); east gate at -97.000 is
        # 25% along, west gate at -97.020 is 75% along.  Odometer is 0–1.5 mi.
        entry_d, exit_d = get_run_timing_refs(
            self._flying_run(), flat_trap, smooth_window=3,
            start_type="Flying", finish_type="Flying",
        )
        assert exit_d - entry_d == pytest.approx(0.75, abs=0.05)

    def test_standing_finish_span_equals_dist_mi(self, flat_trap):
        """Standing finish anchors on exit, entry = exit - dist_mi exactly."""
        entry_d, exit_d = get_run_timing_refs(
            self._flying_run(), flat_trap, smooth_window=3,
            start_type="Flying", finish_type="Standing",
        )
        assert (exit_d - entry_d) == pytest.approx(flat_trap.dist_mi)

    def test_standing_start_span_equals_dist_mi(self, flat_trap):
        """Standing start anchors on entry, exit = entry + dist_mi exactly."""
        entry_d, exit_d = get_run_timing_refs(
            self._flying_run(), flat_trap, smooth_window=3,
            start_type="Standing", finish_type="Flying",
        )
        assert (exit_d - entry_d) == pytest.approx(flat_trap.dist_mi)


# ── load_racebox ──────────────────────────────────────────────────────────────

class TestLoadRacebox:
    def _write_parquet(self, tmp_path, n=10, zero_row=None):
        times = pd.date_range("2026-03-10T10:00:00", periods=n, freq="1s")
        lat = np.linspace(35.00, 35.02, n)
        lon = np.linspace(-97.00, -97.02, n)
        if zero_row is not None:
            lat[zero_row] = 0.0
            lon[zero_row] = 0.0
        df = pd.DataFrame({
            "Time":      times.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3],
            "Latitude":  lat,
            "Longitude": lon,
            "Speed":     np.full(n, 35.0),
        })
        path = tmp_path / "RaceBox_test.parquet"
        df.to_parquet(path, index=False)
        return str(path)

    def test_standard_columns_present(self, tmp_path):
        df = load_racebox(self._write_parquet(tmp_path))
        for col in ("Elapsed time (sec)", "Distance (mi)", "Speed (mph)",
                    "Latitude", "Longitude", "Date"):
            assert col in df.columns

    def test_elapsed_time_starts_at_zero(self, tmp_path):
        df = load_racebox(self._write_parquet(tmp_path))
        assert df["Elapsed time (sec)"].iloc[0] == pytest.approx(0.0)

    def test_distance_starts_at_zero(self, tmp_path):
        df = load_racebox(self._write_parquet(tmp_path))
        assert df["Distance (mi)"].iloc[0] == pytest.approx(0.0)

    def test_distance_monotone_increasing(self, tmp_path):
        df = load_racebox(self._write_parquet(tmp_path))
        assert np.all(np.diff(df["Distance (mi)"].to_numpy()) >= 0)

    def test_zero_latlon_row_interpolated_not_nan(self, tmp_path):
        """A GPS dropout row (0,0) must be interpolated, not left as NaN or zero."""
        df = load_racebox(self._write_parquet(tmp_path, zero_row=3))
        assert df["Latitude"].iloc[3] != 0.0
        assert not np.isnan(df["Latitude"].iloc[3])
        assert df["Longitude"].iloc[3] != 0.0
        assert not np.isnan(df["Longitude"].iloc[3])

    def test_distance_finite_despite_dropout(self, tmp_path):
        df = load_racebox(self._write_parquet(tmp_path, zero_row=3))
        assert np.all(np.isfinite(df["Distance (mi)"]))

    def test_no_duplicate_elapsed_times(self, tmp_path):
        df = load_racebox(self._write_parquet(tmp_path))
        assert df["Elapsed time (sec)"].nunique() == len(df)

    def test_date_column_format(self, tmp_path):
        df = load_racebox(self._write_parquet(tmp_path))
        assert df["Date"].iloc[0] == "2026-03-10"

    def test_speed_mph_column_matches_speed_source(self, tmp_path):
        df = load_racebox(self._write_parquet(tmp_path))
        assert (df["Speed (mph)"] == 35.0).all()


# ── load_speed_tracker ────────────────────────────────────────────────────────

class TestLoadSpeedTracker:
    COLS = ["Elapsed time (sec)", "Distance (mi)", "Speed (mph)", "Latitude", "Longitude"]

    def _write_parquet(self, tmp_path, shuffle=False, duplicate=False):
        elapsed = [0.0, 1.0, 1.0, 2.0, 3.0] if duplicate else list(range(5))
        n = len(elapsed)
        df = pd.DataFrame({
            "Elapsed time (sec)": elapsed,
            "Distance (mi)":      np.linspace(0, 1, n),
            "Speed (mph)":        np.full(n, 30.0),
            "Latitude":           np.linspace(35.0, 35.1, n),
            "Longitude":          np.linspace(-97.0, -97.1, n),
        })
        if shuffle:
            df = df.sample(frac=1, random_state=42)
        path = tmp_path / "speed_tracker.parquet"
        df.to_parquet(path, index=False)
        return str(path)

    def test_standard_columns_present(self, tmp_path):
        df = load_speed_tracker(self._write_parquet(tmp_path))
        for col in self.COLS:
            assert col in df.columns

    def test_sorted_on_elapsed_time(self, tmp_path):
        df = load_speed_tracker(self._write_parquet(tmp_path, shuffle=True))
        assert df["Elapsed time (sec)"].is_monotonic_increasing

    def test_no_duplicate_elapsed_times(self, tmp_path):
        df = load_speed_tracker(self._write_parquet(tmp_path, duplicate=True))
        assert df["Elapsed time (sec)"].nunique() == len(df)

    def test_load_csv(self, tmp_path):
        path = tmp_path / "speed_tracker.csv"
        path.write_text(
            "Elapsed time (sec),Distance (mi),Speed (mph),Latitude,Longitude\n"
            "0.0,0.0,30.0,35.0,-97.0\n"
            "1.0,0.1,30.0,35.01,-97.01\n"
        )
        df = load_speed_tracker(str(path))
        assert list(df["Elapsed time (sec)"]) == [0.0, 1.0]

    def test_csv_with_leading_spaces_in_headers(self, tmp_path):
        path = tmp_path / "speed_tracker_spaces.csv"
        path.write_text(
            " Elapsed time (sec), Distance (mi), Speed (mph), Latitude, Longitude\n"
            "0.0,0.0,30.0,35.0,-97.0\n"
        )
        df = load_speed_tracker(str(path))
        assert "Elapsed time (sec)" in df.columns
