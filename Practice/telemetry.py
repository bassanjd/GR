"""
telemetry.py — Data-source-agnostic GPS telemetry analysis for Great Race practice apps.

All functions here operate on any DataFrame normalised to the standard internal schema:

    Elapsed time (sec)  — time axis
    Distance (mi)       — cumulative GPS odometer (haversine)
    Speed (mph)         — vehicle speed
    Latitude, Longitude — GPS position (decimal degrees)
    Time                — timestamp string (HH:MM:SS)
    Date                — date string (YYYY-MM-DD)

Schema translation (converting source parquet files to this schema) lives in
loaders.py.  Call loaders.load_any(path) in apps; wrap with @st.cache_data.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd


# ── Physics constants ─────────────────────────────────────────────────────────

MPH_S_TO_FT_S2: float = 5280.0 / 3600.0   # 1 mph/s → ft/s²
G_FT_S2: float = 32.174                     # 1 g in ft/s²


# ── Trap geometry ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class TrapConfig:
    """All geometric and behavioural state describing the measured-section traps.

    Derived fields (ux, uy, mi_per_deg_lon, chord_mi) are computed automatically
    from the four endpoint coordinates in __post_init__.  Do not set them
    manually.
    """
    east_lon: float
    west_lon: float
    east_lat: float
    west_lat: float
    dist_mi: float = 1.0                      # Nominal measured-section length (5280 ft)
    flying_window_mi: float = 100.0 / 5280.0  # Half-width of speed-sampling window (ft → mi)
    min_speed: float = 0.2                    # Speed (mph) below which the car is stopped
    mi_per_deg_lat: float = 69.0              # Approximately constant at US latitudes

    # Derived — populated by __post_init__
    ux: float = dataclasses.field(init=False)
    uy: float = dataclasses.field(init=False)
    mi_per_deg_lon: float = dataclasses.field(init=False)
    chord_mi: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        ref_lat_rad = np.radians((self.east_lat + self.west_lat) / 2)
        self.mi_per_deg_lon = 69.172 * float(np.cos(ref_lat_rad))
        dx = (self.east_lon - self.west_lon) * self.mi_per_deg_lon   # miles east
        dy = (self.east_lat - self.west_lat) * self.mi_per_deg_lat   # miles north
        self.chord_mi = float(np.sqrt(dx ** 2 + dy ** 2))
        if self.chord_mi > 0:
            self.ux = dx / self.chord_mi
            self.uy = dy / self.chord_mi
        else:
            self.ux, self.uy = 1.0, 0.0


# ── Signal processing ─────────────────────────────────────────────────────────

def smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with edge handling."""
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().to_numpy()


def compute_derivatives(df: pd.DataFrame, smooth_window: int) -> pd.DataFrame:
    """Add smoothed speed, acceleration (ft/s²), acceleration (g), and jerk (ft/s³).

    Uses np.gradient for non-uniform time spacing.  Gaps > 5 s are masked to
    avoid spurious derivative spikes at GPS dropouts.
    """
    df = df.copy()
    t = df["Elapsed time (sec)"].to_numpy(dtype=float)
    v = df["Speed (mph)"].to_numpy(dtype=float)

    # Interpolate isolated zero-speed GPS dropout spikes linearly
    mask_zero = (v == 0)
    v_interp = pd.Series(np.where(mask_zero, np.nan, v)).interpolate().to_numpy()

    v_smooth = smooth_series(v_interp, smooth_window)
    df["speed_smooth"] = v_smooth

    # Gap mask: flag rows where the time step to the NEXT sample is > 5 s
    dt = np.diff(t, prepend=t[0])
    gap_mask = dt > 5

    # Acceleration: mph/s → ft/s² and g
    dv_dt = np.gradient(v_smooth, t)
    accel_raw = dv_dt * MPH_S_TO_FT_S2
    accel_raw[gap_mask] = np.nan
    accel_smooth = smooth_series(accel_raw, smooth_window)
    df["accel_ft_s2"] = accel_smooth
    df["accel_g"] = accel_smooth / G_FT_S2

    # Jerk in ft/s³
    jerk_raw = np.gradient(np.nan_to_num(accel_smooth), t)
    jerk_raw[gap_mask] = np.nan
    jerk_raw[np.isnan(accel_smooth)] = np.nan  # Null boundary samples adjacent to gaps
    df["jerk_ft_s3"] = smooth_series(jerk_raw, smooth_window)

    return df


# ── GPS distance ──────────────────────────────────────────────────────────────

def haversine_cumulative_mi(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Compute cumulative great-circle distance (miles) from sequential lat/lon arrays."""
    R_MI = 3958.8
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    dlat = np.diff(lat, prepend=lat[0])
    dlon = np.diff(lon, prepend=lon[0])
    lat_prev = np.concatenate([[lat[0]], lat[:-1]])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat_prev) * np.sin(dlon / 2) ** 2
    step_mi = R_MI * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.cumsum(np.nan_to_num(step_mi))  # NaN steps (GPS dropout) → 0 distance


# ── Trap crossing ─────────────────────────────────────────────────────────────

def find_crossing_dist(
    grp: pd.DataFrame,
    trap_lon: float,
    going_west: bool,
    trap: TrapConfig,
) -> float | None:
    """Return cumulative odometer distance (mi) at which the run crosses the trap gate.

    The trap is treated as an oblique line (not pure east-west) using Cartesian
    geometry from *trap*.  The crossing is where the GPS track's signed
    projection along the trap direction past the trap endpoint changes sign —
    i.e. the car passes the gate perpendicular to the trap at that endpoint.

    Returns None if the crossing is never reached.
    """
    lat  = grp["Latitude"].to_numpy()
    lon  = grp["Longitude"].to_numpy()
    dist = grp["Distance (mi)"].to_numpy()

    # Identify which endpoint latitude to use
    trap_lat = (
        trap.east_lat
        if trap_lon >= (trap.east_lon + trap.west_lon) / 2
        else trap.west_lat
    )

    # Signed projection along the trap direction from the target endpoint.
    # Positive = car is on the east side; negative = west side.
    s = (
        ((lon - trap_lon) * trap.mi_per_deg_lon * trap.ux) +
        ((lat - trap_lat) * trap.mi_per_deg_lat * trap.uy)
    )

    mask = s <= 0 if going_west else s >= 0
    if not mask.any():
        return None

    idx = int(np.argmax(mask))
    if idx == 0:
        return float(dist[0])

    # Linear interpolation between the bracketing samples
    s0, s1 = s[idx - 1], s[idx]
    d0, d1 = dist[idx - 1], dist[idx]
    frac = (-s0) / (s1 - s0) if abs(s1 - s0) > 1e-12 else 0.0
    return float(d0 + frac * (d1 - d0))


# ── Run segmentation ──────────────────────────────────────────────────────────

def segment_exercise_runs(
    df: pd.DataFrame,
    lat_thresh: float,
    min_speed: float,
    min_rows: int,
    trap: TrapConfig | None = None,
) -> list[dict]:
    """Isolate the run zone and split into individual runs.

    The run zone is south of *lat_thresh* AND (when *trap* is provided) within
    *trap.flying_window_mi* beyond each trap endpoint in longitude.  Everything
    outside is Transit.

    A new run boundary is declared only on a time gap > 30 s — speed
    transitions within a pass (e.g. standing start/stop) stay in one run.

    Target speed is the mode of per-sample speeds (rounded to the nearest
    5 mph) inside the measured section; falls back to the full moving portion
    when trap positions are unknown.
    """
    if trap is not None:
        window_deg = trap.flying_window_mi / trap.mi_per_deg_lon
        lon_lo = trap.west_lon - window_deg
        lon_hi = trap.east_lon + window_deg
        mask = (
            (df["Latitude"] < lat_thresh) &
            (df["Longitude"] >= lon_lo) &
            (df["Longitude"] <= lon_hi)
        )
    else:
        mask = df["Latitude"] < lat_thresh

    south = df[mask].copy().reset_index(drop=True)
    if south.empty:
        return []

    south["is_moving"] = south["Speed (mph)"] >= min_speed
    south["seg_id"] = (
        south["Elapsed time (sec)"].diff().fillna(0) > 30
    ).cumsum()

    runs: list[dict] = []
    for seg_id, grp in south.groupby("seg_id"):
        moving = grp[grp["is_moving"]]
        if len(moving) < min_rows:
            continue

        lon_delta  = grp["Longitude"].iloc[-1] - grp["Longitude"].iloc[0]
        going_west = lon_delta < 0

        # Restrict speed pool to the measured section when trap is known
        speed_pool = moving
        if trap is not None:
            en_lon = trap.east_lon if going_west else trap.west_lon
            ex_lon = trap.west_lon if going_west else trap.east_lon
            en_d   = find_crossing_dist(grp, en_lon, going_west, trap)
            ex_d   = find_crossing_dist(grp, ex_lon, going_west, trap)
            if en_d is not None and ex_d is not None:
                d_lo, d_hi = min(en_d, ex_d), max(en_d, ex_d)
                trap_section = moving[
                    (moving["Distance (mi)"] >= d_lo) &
                    (moving["Distance (mi)"] <= d_hi)
                ]
                if not trap_section.empty:
                    speed_pool = trap_section

        rounded   = (speed_pool["Speed (mph)"] / 5).round() * 5
        target    = int(rounded.mode().iloc[0])
        if target == 0:
            continue
        direction = "West" if going_west else "East"
        distance  = abs(grp["Distance (mi)"].iloc[-1] - grp["Distance (mi)"].iloc[0])
        _start_time = str(grp["Time"].iloc[0]) if "Time" in grp.columns else ""
        runs.append({
            "seg_id":        seg_id,
            "start_elapsed": float(grp["Elapsed time (sec)"].iloc[0]),
            "end_elapsed":   float(grp["Elapsed time (sec)"].iloc[-1]),
            "start_time":    _start_time,
            "direction":     direction,
            "mean_speed":    float(moving["Speed (mph)"].mean()),
            "target_speed":  target,
            "max_speed":     float(grp["Speed (mph)"].max()),
            "distance_mi":   distance,
            "n_rows":        len(grp),
            "data":          grp.reset_index(drop=True),
        })
    return runs


# ── Plot range helpers ────────────────────────────────────────────────────────

def _sym_range(arr: np.ndarray, pad: float = 0.15,
               min_half: float = 0.5, zero: bool = False) -> list:
    """Min/max range with fractional padding; optionally force zero in range."""
    vals = arr[np.isfinite(arr)]
    if len(vals) == 0:
        return [-min_half, min_half]
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = max(hi - lo, min_half * 2)
    p = span * pad
    return [lo - p, hi + p]


def _pct_range(arr: np.ndarray, pct: float = 2, pad: float = 0.15,
               min_half: float = 1.0, zero: bool = True) -> list:
    """Percentile-clipped range (clips derivative spikes); optionally force zero."""
    vals = arr[np.isfinite(arr)]
    if len(vals) == 0:
        return [-min_half, min_half]
    lo = float(np.percentile(vals, pct))
    hi = float(np.percentile(vals, 100 - pct))
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = max(hi - lo, min_half * 2)
    p = span * pad
    return [lo - p, hi + p]


# ── Run classification ────────────────────────────────────────────────────────

def _grp_derivs(r: dict, smooth_window: int) -> pd.DataFrame:
    """Return the derivative-enriched DataFrame for a run, computing if needed."""
    g = r.get("_grp_with_derivs")
    return g if g is not None else compute_derivatives(r["data"].copy(), smooth_window)


def _min_speed_near(
    grp: pd.DataFrame,
    center_d: float,
    v_col: str,
    side: str,
    flying_window_mi: float,
) -> float:
    """Return minimum smoothed speed in the standing-detection window outside the trap.

    side='before': [center_d - window, center_d] — approach to entry trap.
    side='after' : [center_d, center_d + window] — departure from exit trap.
    Falls back to an interpolated point if the window contains no samples.
    """
    _d = grp["Distance (mi)"].to_numpy()
    if side == "before":
        mask = (_d >= center_d - flying_window_mi) & (_d <= center_d)
    else:
        mask = (_d >= center_d) & (_d <= center_d + flying_window_mi)
    if mask.any():
        return float(grp.loc[mask, v_col].min())
    return float(np.interp(center_d, _d, grp[v_col].to_numpy()))


def _last_stopped_before(grp: pd.DataFrame, before_d: float, min_speed: float) -> float | None:
    """Return Distance (mi) at the last sample with Speed < min_speed before before_d."""
    candidates = grp[
        (grp["Distance (mi)"] < before_d) & (grp["Speed (mph)"] < min_speed)
    ]
    if candidates.empty:
        return None
    return float(candidates["Distance (mi)"].iloc[-1])


def _first_stopped_after(grp: pd.DataFrame, after_d: float, min_speed: float) -> float | None:
    """Return Distance (mi) at the first sample with Speed < min_speed after after_d."""
    candidates = grp[
        (grp["Distance (mi)"] > after_d) & (grp["Speed (mph)"] < min_speed)
    ]
    if candidates.empty:
        return None
    return float(candidates["Distance (mi)"].iloc[0])


def _standing_anchor(
    grp: pd.DataFrame,
    gate_lon: float,
    going_west: bool,
    trap: TrapConfig,
    side: str,
) -> float:
    """Return the zero-velocity anchor distance for a standing start or finish.

    Locates the GPS crossing of *gate_lon* and uses it as a rough positional
    guide, then finds the nearest stationary sample.  Zero-velocity readings
    are more accurate than GPS-derived crossing positions, so the search
    window extends one flying_window_mi past the crossing to accommodate
    GPS position error (the stationary sample may appear slightly on the
    'wrong' side of the computed crossing in cumulative-distance space).

    side='before': last stopped sample at or before crossing + flying_window_mi
                   (standing start — car was stationary just before trap entry).
    side='after' : first stopped sample at or after crossing - flying_window_mi
                   (standing finish — car stopped just after trap exit).

    Fallback chain: stationary search → crossing distance → segment edge.
    """
    d_vals  = grp["Distance (mi)"]
    d_start = float(d_vals.iloc[0])
    d_end   = float(d_vals.iloc[-1])
    d_mid   = (d_start + d_end) / 2

    crossing = find_crossing_dist(grp, gate_lon, going_west, trap)

    if side == "before":
        limit  = (crossing + trap.flying_window_mi) if crossing is not None else d_mid
        anchor = _last_stopped_before(grp, limit, trap.min_speed)
        return anchor if anchor is not None else (crossing if crossing is not None else d_start)
    else:
        limit  = (crossing - trap.flying_window_mi) if crossing is not None else d_mid
        anchor = _first_stopped_after(grp, limit, trap.min_speed)
        return anchor if anchor is not None else (crossing if crossing is not None else d_end)


def run_start_type(r: dict, trap: TrapConfig, smooth_window: int) -> str:
    """Return 'Standing' or 'Flying' based on min speed in window around trap entry.

    Flying  = approach speed is within 5 mph of target (car entered at speed).
    Standing = approach speed is more than 5 mph below target (standing start).
    """
    grp   = _grp_derivs(r, smooth_window)
    gw    = r["direction"] == "West"
    e_lon = trap.east_lon if gw else trap.west_lon
    e_d   = find_crossing_dist(grp, e_lon, gw, trap)
    v_col = "speed_smooth" if "speed_smooth" in grp.columns else "Speed (mph)"
    if e_d is None:
        e_d = grp["Distance (mi)"].iloc[0]
    return (
        "Flying"
        if _min_speed_near(grp, e_d, v_col, "before", trap.flying_window_mi) >= r["target_speed"] - 5
        else "Standing"
    )


def run_finish_type(r: dict, trap: TrapConfig, smooth_window: int) -> str:
    """Return 'Standing' or 'Flying' based on min speed in window around trap exit.

    Flying  = departure speed is within 5 mph of target (car left at speed).
    Standing = departure speed is more than 5 mph below target (car stopped).
    """
    grp    = _grp_derivs(r, smooth_window)
    gw     = r["direction"] == "West"
    en_lon = trap.east_lon if gw else trap.west_lon
    ex_lon = trap.west_lon if gw else trap.east_lon
    en_d   = find_crossing_dist(grp, en_lon, gw, trap)
    ex_d   = find_crossing_dist(grp, ex_lon, gw, trap)
    if ex_d is None:
        ex_d = (en_d + trap.dist_mi) if en_d is not None else grp["Distance (mi)"].iloc[-1]
    v_col = "speed_smooth" if "speed_smooth" in grp.columns else "Speed (mph)"
    return (
        "Flying"
        if _min_speed_near(grp, ex_d, v_col, "after", trap.flying_window_mi) >= r["target_speed"] - 5
        else "Standing"
    )


def get_run_timing_refs(
    r: dict,
    trap: TrapConfig,
    smooth_window: int,
    start_type: str | None = None,
    finish_type: str | None = None,
) -> tuple[float | None, float | None]:
    """Return (entry_d, exit_d) cumulative-odometer distances (mi) for timing.

    Standing finish → exit_d  = first stationary sample after the trap exit
                                 crossing; entry_d = exit_d - trap.dist_mi.
    Standing start  → entry_d = last stationary sample before the trap entry
                                 crossing; exit_d = entry_d + trap.dist_mi.
    Flying          → GPS trap-crossing geometry, trap.dist_mi fallback.
    """
    if start_type is None:
        start_type = run_start_type(r, trap, smooth_window)
    if finish_type is None:
        finish_type = run_finish_type(r, trap, smooth_window)
    grp = _grp_derivs(r, smooth_window)
    gw  = r["direction"] == "West"

    # ── Standing finish: anchor on trap exit crossing, first stop after it ──────
    if finish_type == "Standing":
        ex_lon = trap.west_lon if gw else trap.east_lon
        exit_d = _standing_anchor(grp, ex_lon, gw, trap, "after")
        return exit_d - trap.dist_mi, exit_d

    # ── Standing start: anchor on trap entry crossing, last stop before it ─────
    if start_type == "Standing":
        en_lon  = trap.east_lon if gw else trap.west_lon
        entry_d = _standing_anchor(grp, en_lon, gw, trap, "before")
        return entry_d, entry_d + trap.dist_mi

    # ── Flying start and finish: GPS trap-crossing geometry ───────────────────
    en_lon  = trap.east_lon if gw else trap.west_lon
    ex_lon  = trap.west_lon if gw else trap.east_lon
    entry_d = find_crossing_dist(grp, en_lon, gw, trap)
    exit_d  = find_crossing_dist(grp, ex_lon, gw, trap)
    if entry_d is None and exit_d is not None:
        entry_d = exit_d - trap.dist_mi
    if exit_d is None and entry_d is not None:
        exit_d = entry_d + trap.dist_mi
    return entry_d, exit_d
