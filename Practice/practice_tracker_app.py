"""
Speed Tracker — Acceleration & Jerk Analysis
Practice run evaluation
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Speed Tracker Analysis", layout="wide", page_icon="🏎")

DATALOGS = Path(__file__).parent / "DataLogs"

# ── Physics constants ─────────────────────────────────────────────────────────
MPH_S_TO_FT_S2 = 5280.0 / 3600.0   # 1 mph/s = 1.4667 ft/s²
G_FT_S2 = 32.174                     # 1 g in ft/s²


# ── Helpers ───────────────────────────────────────────────────────────────────

def smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with edge handling."""
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().to_numpy()


def compute_derivatives(df: pd.DataFrame, smooth_window: int) -> pd.DataFrame:
    """
    Add smoothed speed, acceleration (ft/s²), acceleration (g), and jerk (ft/s³)
    columns.  Uses np.gradient which handles non-uniform time spacing.
    Large time gaps (>5 s) are masked to avoid spurious spikes.
    """
    df = df.copy()
    t = df["Elapsed time (sec)"].to_numpy(dtype=float)
    v = df["Speed (mph)"].to_numpy(dtype=float)

    # Replace zero-speed GPS dropout spikes: if a single sample is 0 surrounded
    # by non-zero values, interpolate it linearly.
    mask_zero = (v == 0)
    v_interp = pd.Series(np.where(mask_zero, np.nan, v)).interpolate().to_numpy()

    v_smooth = smooth_series(v_interp, smooth_window)
    df["speed_smooth"] = v_smooth

    # Gap mask: flag rows where the time step to the NEXT sample is large
    dt = np.diff(t, prepend=t[0])
    gap_mask = dt > 5  # seconds

    # Acceleration in mph/s → convert to ft/s² and g
    dv_dt = np.gradient(v_smooth, t)          # mph/s (handles non-uniform t)
    accel_raw = dv_dt * MPH_S_TO_FT_S2        # ft/s²
    accel_raw[gap_mask] = np.nan
    accel_smooth = smooth_series(accel_raw, smooth_window)
    df["accel_ft_s2"] = accel_smooth
    df["accel_g"] = accel_smooth / G_FT_S2

    # Jerk in ft/s³
    jerk_raw = np.gradient(np.nan_to_num(accel_smooth), t)
    jerk_raw[gap_mask] = np.nan
    df["jerk_ft_s3"] = smooth_series(jerk_raw, smooth_window)

    return df


def segment_exercise_runs(df: pd.DataFrame, lat_thresh: float,
                          min_speed: float, min_rows: int,
                          trap_east: float | None = None,
                          trap_west: float | None = None) -> list[dict]:
    """
    Isolate the southern exercise area and split into individual runs.
    A new run boundary is declared when speed drops below min_speed OR when
    there is a time gap > 30 s between consecutive GPS fixes.

    Target speed: each GPS sample between the traps is rounded to the nearest
    5 mph; the most frequent (mode) value is used as the target.  Falls back
    to the mode over the full moving portion when trap positions are unknown.
    """
    south = df[df["Latitude"] < lat_thresh].copy().reset_index(drop=True)
    if south.empty:
        return []

    south["is_moving"] = south["Speed (mph)"] >= min_speed
    south["time_gap"] = south["Elapsed time (sec)"].diff().fillna(0) > 30

    # New segment: transition from moving→stopped, or a time gap
    stopped = ~south["is_moving"]
    prev_moving = south["is_moving"].shift(1, fill_value=True)
    south["seg_id"] = (
        (stopped & prev_moving) | south["time_gap"]
    ).cumsum()

    runs = []
    for seg_id, grp in south.groupby("seg_id"):
        moving = grp[grp["is_moving"]]
        if len(moving) < min_rows:
            continue
        mean_spd = moving["Speed (mph)"].mean()

        # Determine direction so we know which trap is entry vs exit
        lon_delta = grp["Longitude"].iloc[-1] - grp["Longitude"].iloc[0]
        going_west = lon_delta < 0

        # Restrict to measured section when trap positions are available
        speed_pool = moving
        if trap_east is not None and trap_west is not None:
            en_lon = trap_east if going_west else trap_west
            ex_lon = trap_west if going_west else trap_east
            en_d = find_crossing_dist(grp, en_lon, going_west)
            ex_d = find_crossing_dist(grp, ex_lon, going_west)
            if en_d is not None and ex_d is not None:
                d_lo, d_hi = min(en_d, ex_d), max(en_d, ex_d)
                trap_section = moving[
                    (moving["Distance (mi)"] >= d_lo) &
                    (moving["Distance (mi)"] <= d_hi)
                ]
                if not trap_section.empty:
                    speed_pool = trap_section

        # Mode of per-sample speeds rounded to nearest 5 mph
        rounded = (speed_pool["Speed (mph)"] / 5).round() * 5
        target = int(rounded.mode().iloc[0])
        direction = "West" if going_west else "East"
        distance = abs(grp["Distance (mi)"].iloc[-1] - grp["Distance (mi)"].iloc[0])
        _start_time = str(grp["Time"].iloc[0]) if "Time" in grp.columns else ""
        runs.append({
            "seg_id": seg_id,
            "start_elapsed": int(grp["Elapsed time (sec)"].iloc[0]),
            "end_elapsed": int(grp["Elapsed time (sec)"].iloc[-1]),
            "start_time": _start_time,
            "direction": direction,
            "mean_speed": mean_spd,
            "target_speed": target,
            "max_speed": grp["Speed (mph)"].max(),
            "distance_mi": distance,
            "n_rows": len(grp),
            "data": grp.reset_index(drop=True),
        })
    return runs


# ── Trap crossing helper ──────────────────────────────────────────────────────

def find_crossing_dist(grp: pd.DataFrame, trap_lon: float, going_west: bool) -> float | None:
    """
    Return the cumulative odometer distance (mi) at which the run crosses the
    perpendicular to the trap line at the specified trap endpoint (trap_lon).

    The trap is treated as an oblique line (not pure east-west) using the
    module-level geometry (_TRAP_UX, _TRAP_UY, _MI_PER_DEG_LON, etc.).
    The "crossing" is where the GPS track's signed projection along the trap
    direction past the trap endpoint changes sign — i.e. the car passes the
    gate perpendicular to the trap at that endpoint.

    Returns None if the crossing is never reached.
    """
    lat  = grp["Latitude"].to_numpy()
    lon  = grp["Longitude"].to_numpy()
    dist = grp["Distance (mi)"].to_numpy()

    # Identify which endpoint we are looking for
    trap_lat = trap_east_lat if trap_lon >= (trap_east + trap_west) / 2 else trap_west_lat

    # Signed projection along the trap direction from the target endpoint.
    # Positive = car is on the east side; negative = west side.
    s = (((lon - trap_lon) * _MI_PER_DEG_LON * _TRAP_UX) +
         ((lat - trap_lat) * _MI_PER_DEG_LAT * _TRAP_UY))

    if going_west:
        # Westbound: starts east of this endpoint (s > 0), crosses when s <= 0
        mask = s <= 0
    else:
        # Eastbound: starts west of this endpoint (s < 0), crosses when s >= 0
        mask = s >= 0

    if not mask.any():
        return None

    idx = int(np.argmax(mask))   # first True
    if idx == 0:
        return float(dist[0])

    # Linear interpolation between the bracketing samples
    s0, s1 = s[idx - 1], s[idx]
    d0, d1 = dist[idx - 1], dist[idx]
    frac = (-s0) / (s1 - s0) if s1 != s0 else 0.0
    return float(d0 + frac * (d1 - d0))


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_track(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, skipinitialspace=True)
        df.columns = df.columns.str.strip()
    df = df.sort_values("Elapsed time (sec)")
    df = df.drop_duplicates(subset="Elapsed time (sec)", keep="first")
    return df.reset_index(drop=True)


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Practice run evaluation")
st.caption("Analyze GPS logs from the Speed Tracker app to evaluate acceleration, jerk, and run quality.")

pq_files  = sorted(DATALOGS.glob("*.parquet"), reverse=True)
csv_files = sorted(DATALOGS.glob("*.csv"),     reverse=True)
track_files = pq_files or csv_files
if not track_files:
    st.error(f"No parquet or CSV files found in {DATALOGS}")
    st.stop()

# ── Settings defaults via session state (Settings tab widgets write here) ─────
for _k, _v in [("cfg_lat", 29.34), ("cfg_smooth", 9),
               ("cfg_min_speed", 3), ("cfg_min_rows", 40)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

lat_threshold = float(st.session_state.cfg_lat)
smooth_window = int(st.session_state.cfg_smooth)
min_speed     = float(st.session_state.cfg_min_speed)
min_rows      = int(st.session_state.cfg_min_rows)

with st.sidebar:
    st.header("Trap Setup")
    selected_file = st.selectbox(
        "Log file", track_files, format_func=lambda p: p.name
    )

# ── Load data early so we can estimate trap locations before rendering sliders ─
df_raw = load_track(str(selected_file))

# Estimate trap locations from at-rest GPS points in the exercise area.
# When the car waits between runs it is sitting at one of the two traps, so
# the stationary positions cluster tightly at exactly the trap longitudes.
_df_ex = df_raw[df_raw["Latitude"] < lat_threshold]
_rest  = _df_ex[_df_ex["Speed (mph)"] < 2]
if len(_rest) >= 6:
    _lons  = _rest["Longitude"].values
    _mid   = float(np.median(_lons))
    _e_lon = _lons[_lons > _mid]
    _w_lon = _lons[_lons <= _mid]
    _trap_east_def = round(float(np.median(_e_lon)), 4) if len(_e_lon) >= 2 else -95.641
    _trap_west_def = round(float(np.median(_w_lon)), 4) if len(_w_lon) >= 2 else -95.656
else:
    _trap_east_def = -95.641
    _trap_west_def = -95.656

_TRAP_MARGIN = 0.002   # slider range ± ~660 ft around auto-detected default

with st.sidebar:
    st.divider()
    st.subheader("Measured section traps")
    st.caption(
        f"Auto-detected from {len(_rest)} at-rest GPS points.  \n"
        f"East: **{_trap_east_def:.4f}**, West: **{_trap_west_def:.4f}**  \n"
        f"Slider step = 0.0001° ≈ 33 ft"
    )
    trap_east = st.slider(
        "East trap lon",
        float(round(_trap_east_def - _TRAP_MARGIN, 4)),
        float(round(_trap_east_def + _TRAP_MARGIN, 4)),
        float(_trap_east_def),
        step=0.0001, format="%.4f",
        help="Eastern boundary of the measured distance (~33 ft per step)."
    )
    _te_exact = st.text_input(
        "East exact lon", value="", placeholder=f"{trap_east:.4f}",
        help="Type an exact longitude to override the slider.",
        label_visibility="collapsed",
    )
    try:
        trap_east = float(_te_exact) if _te_exact.strip() else trap_east
    except ValueError:
        pass

    trap_west = st.slider(
        "West trap lon",
        float(round(_trap_west_def - _TRAP_MARGIN, 4)),
        float(round(_trap_west_def + _TRAP_MARGIN, 4)),
        float(_trap_west_def),
        step=0.0001, format="%.4f",
        help="Western boundary of the measured distance (~33 ft per step)."
    )
    _tw_exact = st.text_input(
        "West exact lon", value="", placeholder=f"{trap_west:.4f}",
        help="Type an exact longitude to override the slider.",
        label_visibility="collapsed",
    )
    try:
        trap_west = float(_tw_exact) if _tw_exact.strip() else trap_west
    except ValueError:
        pass

# ── Precise trap geometry ──────────────────────────────────────────────────────
# The trap is exactly 5280 ft (1 mile).  We estimate each endpoint's latitude
# from the at-rest GPS cluster at that trap so that crossing detection uses
# the actual (slightly oblique) trap line rather than a pure longitude gate.

TRAP_DIST_MI = 1.0   # 5280 ft exactly

_fallback_lat = 29.337   # used if at-rest data is insufficient
if len(_rest) >= 6:
    _e_rest = _rest[_rest["Longitude"] > _mid]
    _w_rest = _rest[_rest["Longitude"] <= _mid]
    trap_east_lat = float(np.median(_e_rest["Latitude"])) if len(_e_rest) >= 2 else _fallback_lat
    trap_west_lat = float(np.median(_w_rest["Latitude"])) if len(_w_rest) >= 2 else _fallback_lat
else:
    trap_east_lat = trap_west_lat = _fallback_lat

# Convert the trap endpoints to approximate Cartesian (x east, y north) in miles
_trap_ref_lat_rad = np.radians((trap_east_lat + trap_west_lat) / 2)
_MI_PER_DEG_LON   = 69.172 * float(np.cos(_trap_ref_lat_rad))
_MI_PER_DEG_LAT   = 69.0
_dx = (trap_east - trap_west) * _MI_PER_DEG_LON   # miles east
_dy = (trap_east_lat - trap_west_lat) * _MI_PER_DEG_LAT  # miles north
_trap_chord = float(np.sqrt(_dx**2 + _dy**2))

# Trap unit vector (west → east), Cartesian miles
if _trap_chord > 0:
    _TRAP_UX, _TRAP_UY = _dx / _trap_chord, _dy / _trap_chord
else:
    _TRAP_UX, _TRAP_UY = 1.0, 0.0

with st.sidebar:
    st.caption(
        f"Trap lats estimated from at-rest GPS — "
        f"East: **{trap_east_lat:.5f}**, West: **{trap_west_lat:.5f}** | "
        f"Chord: **{_trap_chord * 5280:.0f} ft** (locked to 1.0 mi for calculations)"
    )

runs = segment_exercise_runs(df_raw, lat_threshold, min_speed, min_rows,
                             trap_east=trap_east, trap_west=trap_west)

# ── Transit segments (pre-compute before tabs) ────────────────────────────────
ex_mask = df_raw["Latitude"] < lat_threshold
if ex_mask.any():
    ex_start_elapsed = float(df_raw.loc[ex_mask, "Elapsed time (sec)"].min())
    ex_end_elapsed   = float(df_raw.loc[ex_mask, "Elapsed time (sec)"].max())
else:
    ex_start_elapsed = ex_end_elapsed = None

df_outbound = df_raw[df_raw["Elapsed time (sec)"] <  (ex_start_elapsed or 0)].copy()
df_inbound  = df_raw[df_raw["Elapsed time (sec)"] >  (ex_end_elapsed   or float("inf"))].copy()
df_outbound["Leg"] = "To Exercise"
df_inbound["Leg"]  = "From Exercise"
df_transit = pd.concat([df_outbound, df_inbound]).reset_index(drop=True)

# ── Range helpers (used in multiple tabs) ─────────────────────────────────────

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


# ── Tabs ──────────────────────────────────────────────────────────────────────

def run_start_type(r: dict) -> str:
    """Return 'Standing' or 'Flying' based on smoothed speed at trap entry."""
    grp = r.get("_grp_with_derivs")
    if grp is None:
        grp = r["data"]
    gw      = r["direction"] == "West"
    e_lon   = trap_east if gw else trap_west
    e_d     = find_crossing_dist(grp, e_lon, gw)
    if e_d is None:
        return "?"
    v_col   = "speed_smooth" if "speed_smooth" in grp.columns else "Speed (mph)"
    v_entry = float(np.interp(e_d, grp["Distance (mi)"].to_numpy(),
                              grp[v_col].to_numpy()))
    return "Flying" if v_entry >= 0.5 * r["target_speed"] else "Standing"


def run_finish_type(r: dict) -> str:
    """Return 'Standing', 'Flying', or 'Unknown' based on smoothed speed at trap exit.

    Uses the same geometric fallback as the detail tab: if the GPS track ends
    before the exit trap longitude is reached, estimates the exit position as
    entry + TRAP_DIST_MI (1 mile) rather than falling back to tail speed.
    """
    grp = r.get("_grp_with_derivs")
    if grp is None:
        grp = r["data"]
    gw     = r["direction"] == "West"
    en_lon = trap_east if gw else trap_west
    ex_lon = trap_west if gw else trap_east
    en_d   = find_crossing_dist(grp, en_lon, gw)
    ex_d   = find_crossing_dist(grp, ex_lon, gw)
    # Geometric fallback: if exit not found but entry was, exit = entry + 1 mi
    if ex_d is None and en_d is not None:
        ex_d = en_d + TRAP_DIST_MI
    v_col  = "speed_smooth" if "speed_smooth" in grp.columns else "Speed (mph)"
    _v     = grp[v_col].to_numpy()
    _d     = grp["Distance (mi)"].to_numpy()
    if ex_d is not None:
        v_exit = float(np.interp(ex_d, _d, _v, left=_v[0], right=_v[-1]))
    else:
        v_exit = float(_v[-min(5, len(_v)):].mean())
    if v_exit < 5:
        return "Standing"
    if v_exit >= r["target_speed"] - 5:
        return "Flying"
    return "Unknown"


tab_map, tab_overview, tab_detail, tab_transit, tab_settings = st.tabs(
    ["Track Map", "Runs Overview", "Run Detail", "Transit Analysis", "Settings"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Track map
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    col_map, col_stats = st.columns([4, 1])

    df_raw["Zone"] = np.where(
        df_raw["Latitude"] < lat_threshold, "Exercise Area", "Transit"
    )
    df_plot = df_raw.iloc[::5].copy()  # downsample for render speed

    # Compute exercise track lat bounds for drawing trap lines
    ex_rows = df_raw[
        (df_raw["Latitude"] < lat_threshold) &
        (df_raw["Longitude"] >= trap_west) &
        (df_raw["Longitude"] <= trap_east)
    ]
    if not ex_rows.empty:
        trap_lat_mid  = float(ex_rows["Latitude"].median())
        trap_lat_lo   = float(ex_rows["Latitude"].quantile(0.05))
        trap_lat_hi   = float(ex_rows["Latitude"].quantile(0.95))
    else:
        trap_lat_mid  = 29.337
        trap_lat_lo   = 29.334
        trap_lat_hi   = 29.340

    with col_map:
        map_style_choice = st.radio(
            "Map style", ["Street (carto)", "Satellite (ESRI)"],
            horizontal=True, label_visibility="collapsed"
        )
        use_satellite = map_style_choice == "Satellite (ESRI)"

        _map_center_lat = (trap_east_lat + trap_west_lat) / 2
        _map_center_lon = (trap_east + trap_west) / 2
        fig_map = px.scatter_map(
            df_plot,
            lat="Latitude", lon="Longitude",
            color="Zone",
            color_discrete_map={"Exercise Area": "#e63946", "Transit": "#457b9d"},
            hover_data={"Speed (mph)": ":.1f", "Time": True, "Date": True,
                        "Elapsed time (sec)": True, "Distance (mi)": ":.2f"},
            zoom=14,
            center={"lat": _map_center_lat, "lon": _map_center_lon},
            map_style="white-bg" if use_satellite else "carto-positron",
            title="Full Track — Exercise Area & Measured Section Highlighted",
        )
        fig_map.update_traces(marker=dict(size=3))

        # Overlay satellite tiles when requested
        if use_satellite:
            fig_map.update_layout(map=dict(
                layers=[{
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/services/"
                        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ],
                    "sourceattribution": "Tiles &copy; Esri",
                }]
            ))

        # East trap boundary line
        fig_map.add_trace(go.Scattermap(
            lat=[trap_lat_lo, trap_lat_hi],
            lon=[trap_east, trap_east],
            mode="lines",
            line=dict(color="dodgerblue", width=3),
            name="East trap",
            hoverinfo="name",
        ))
        # West trap boundary line
        fig_map.add_trace(go.Scattermap(
            lat=[trap_lat_lo, trap_lat_hi],
            lon=[trap_west, trap_west],
            mode="lines",
            line=dict(color="dodgerblue", width=3),
            name="West trap",
            hoverinfo="name",
        ))
        # Measured section centre line connecting the two traps
        fig_map.add_trace(go.Scattermap(
            lat=[trap_lat_mid, trap_lat_mid],
            lon=[trap_east, trap_west],
            mode="lines",
            line=dict(color="yellow", width=2),
            name="Measured section",
            hoverinfo="name",
        ))

        fig_map.update_layout(height=580, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    with col_stats:
        st.metric("Total distance", f"{df_raw['Distance (mi)'].max():.1f} mi")
        elapsed_h = df_raw["Elapsed time (sec)"].max() / 3600
        st.metric("Session duration", f"{elapsed_h:.2f} hrs")
        st.metric("Exercise runs found", len(runs))
        st.metric("Max speed (all)", f"{df_raw['Speed (mph)'].max():.1f} mph")
        south_mask = df_raw["Latitude"] < lat_threshold
        if south_mask.any():
            st.metric("Max speed (exercise)", f"{df_raw.loc[south_mask, 'Speed (mph)'].max():.1f} mph")

        if runs:
            target_speeds = sorted({r["target_speed"] for r in runs})
            st.markdown("**Target speeds detected:**")
            for t in target_speeds:
                n = sum(1 for r in runs if r["target_speed"] == t)
                st.markdown(f"- {t} mph ({n} run{'s' if n != 1 else ''})")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Runs overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    if not runs:
        st.warning("No exercise runs found. Try adjusting the settings in the sidebar.")
        st.stop()

    # Summary table
    st.subheader("Run Summary (measured section only)")

    # Pass 1: compute derivatives and cache on every run
    for r in runs:
        if "_grp_with_derivs" not in r:
            grp = r["data"].copy()
            grp = compute_derivatives(grp, smooth_window)
            r["_grp_with_derivs"] = grp

    # Build the peer acceleration model once from all standing-start runs.
    # Normalised speed-vs-distance profiles aligned at trap entry, pointwise median.
    _ov_standing = [r for r in runs if run_start_type(r) == "Standing"]
    _ov_accel_profs = []
    for _r_p in _ov_standing:
        _grp_p  = _r_p["_grp_with_derivs"]
        _gw_p   = _r_p["direction"] == "West"
        _en_p   = find_crossing_dist(_grp_p, trap_east if _gw_p else trap_west, _gw_p)
        if _en_p is None:
            continue
        _d_p  = _grp_p["Distance (mi)"].to_numpy()
        _v_p  = _grp_p["speed_smooth"].to_numpy()
        _vt_p = float(_r_p["target_speed"])
        _daln = _d_p - _en_p
        _ph   = (_daln >= -0.05) & (_v_p < 1.01 * _vt_p)
        if _ph.sum() < 5:
            continue
        _ds = _daln[_ph]; _vn = np.clip(_v_p[_ph] / _vt_p, 0.0, 1.05)
        _si = np.argsort(_ds)
        _ov_accel_profs.append((_ds[_si], _vn[_si]))

    if _ov_accel_profs:
        _ov_d_end  = max(p[0][-1] for p in _ov_accel_profs)
        _OV_DGRID  = np.linspace(0.0, _ov_d_end, 400)
        _ov_all_v  = np.vstack([
            np.clip(np.interp(_OV_DGRID, d, v, left=v[0], right=1.0), 0.0, 1.05)
            for d, v in _ov_accel_profs
        ])
        _ov_v_nrm  = np.median(_ov_all_v, axis=0)   # normalised [0,1]
    else:
        _OV_DGRID = _ov_v_nrm = None

    # Pass 2: per-run summary stats + cumulative time error
    summary_rows = []
    for i, r in enumerate(runs):
        grp = r["_grp_with_derivs"]

        # Trap crossings with geometric fallback (same as detail tab)
        gw     = r["direction"] == "West"
        en_lon = trap_east if gw else trap_west
        ex_lon = trap_west if gw else trap_east
        en_d   = find_crossing_dist(grp, en_lon, gw)
        ex_d   = find_crossing_dist(grp, ex_lon, gw)
        if en_d is None and ex_d is not None:
            en_d = ex_d - TRAP_DIST_MI
        if ex_d is None and en_d is not None:
            ex_d = en_d + TRAP_DIST_MI

        if en_d is not None and ex_d is not None:
            d_lo, d_hi = min(en_d, ex_d), max(en_d, ex_d)
            section = grp[
                (grp["Distance (mi)"] >= d_lo) & (grp["Distance (mi)"] <= d_hi)
            ]
        else:
            section = grp

        section    = section[section["Speed (mph)"] >= min_speed]
        spd_err    = section["Speed (mph)"] - r["target_speed"]
        measured_dist = (d_hi - d_lo) if (en_d is not None and ex_d is not None) else r["distance_mi"]

        if section.empty:
            continue

        # ── Cumulative time error at exit (same method as detail tab) ─────────
        time_err_s = None
        if en_d is not None and ex_d is not None:
            _d_arr    = grp["Distance (mi)"].to_numpy()
            _v_arr    = grp["speed_smooth"].to_numpy()
            _d_aln    = _d_arr - en_d
            _trap_len = ex_d - en_d
            _tmask    = (_d_aln >= 0) & (_d_aln <= _trap_len)
            if _tmask.sum() >= 3:
                _si       = np.argsort(_d_aln[_tmask])
                _d_s      = _d_aln[_tmask][_si]
                _v_act    = np.where(_v_arr[_tmask][_si] > 1.0,
                                     _v_arr[_tmask][_si], np.nan)
                _target_f = float(r["target_speed"])

                if run_start_type(r) == "Flying" or _OV_DGRID is None:
                    _v_id = np.full(len(_d_s), _target_f)
                else:
                    _v_id = np.clip(
                        np.interp(_d_s, _OV_DGRID, _ov_v_nrm * _target_f,
                                  left=np.nan, right=np.nan),
                        0.0, _target_f,
                    )

                _v_id  = np.where(_v_id > 1.0, _v_id, np.nan)
                _dd    = np.diff(_d_s, prepend=_d_s[0])
                _dt    = np.nan_to_num(
                    (1.0 / _v_id - 1.0 / _v_act) * _dd * 3600.0, nan=0.0
                )
                time_err_s = round(float(np.sum(_dt)), 2)

        summary_rows.append({
            "Run": i + 1,
            "Time": r["start_time"],
            "Direction": r["direction"],
            "Target (mph)": r["target_speed"],
            "Start": run_start_type(r),
            "Finish": run_finish_type(r),
            "Time err at exit (s)": time_err_s,
            "Mean speed (mph)": round(float(section["Speed (mph)"].mean()), 1),
            "Max speed (mph)": round(float(section["Speed (mph)"].max()), 1),
            "Speed RMS err (mph)": round(float(np.sqrt((spd_err ** 2).mean())), 2),
            "Measured dist (mi)": round(measured_dist, 4),
            "Max |Accel| (ft/s²)": round(float(section["accel_ft_s2"].abs().max()), 2),
            "RMS Accel (ft/s²)": round(float(np.sqrt((section["accel_ft_s2"] ** 2).mean())), 2),
            "Max |Accel| (g)": round(float(section["accel_g"].abs().max()), 3),
            "Max |Jerk| (ft/s³)": round(float(section["jerk_ft_s3"].abs().max()), 2),
            "RMS Jerk (ft/s³)": round(float(np.sqrt((section["jerk_ft_s3"] ** 2).mean())), 2),
        })
        r["_measured_section"] = section   # cache for accel distribution

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()

    # All runs speed overlay
    st.subheader("Speed Overlay — All Runs (time from run start)")
    colors = px.colors.qualitative.Set2
    fig_all = go.Figure()
    for i, r in enumerate(runs):
        grp = r["data"]
        t_rel = grp["Elapsed time (sec)"] - grp["Elapsed time (sec)"].iloc[0]
        fig_all.add_trace(go.Scatter(
            x=t_rel,
            y=grp["Speed (mph)"],
            mode="lines",
            name=f"R{i+1} {r['direction']} ~{r['target_speed']} mph",
            line=dict(color=colors[i % len(colors)], width=1.5),
            opacity=0.8,
        ))
    fig_all.update_layout(
        xaxis_title="Time from run start (s)",
        yaxis_title="Speed (mph)",
        height=420,
        legend=dict(orientation="v", x=1.01, font_size=11),
        margin=dict(r=180),
    )
    st.plotly_chart(fig_all, use_container_width=True)

    st.divider()

    # Trap-aligned speed comparison
    st.subheader("Measured Section Comparison — Aligned at Trap Entry")
    st.caption(
        f"X = 0 at trap entry | dashed red line = trap exit (median length). "
        f"East trap {trap_east:.4f}, west trap {trap_west:.4f}. "
        f"Exit distance uses geometric fallback when GPS data ends before exit trap."
    )
    fig_trap = go.Figure()
    for i, r in enumerate(runs):
        grp = r["data"].copy()
        going_west = r["direction"] == "West"
        entry_lon = trap_east if going_west else trap_west

        entry_dist = find_crossing_dist(grp, entry_lon, going_west)
        if entry_dist is None:
            continue

        d_aligned = grp["Distance (mi)"] - entry_dist
        fig_trap.add_trace(go.Scatter(
            x=d_aligned,
            y=grp["Speed (mph)"],
            mode="lines",
            name=f"R{i+1} {r['direction']} ~{r['target_speed']} mph",
            line=dict(color=colors[i % len(colors)], width=1.5),
            opacity=0.8,
        ))

    # Mark trap boundaries — exit is always exactly TRAP_DIST_MI from entry
    fig_trap.add_vline(x=0, line_dash="dash", line_color="green",
                       annotation_text="Trap entry", annotation_position="top right")
    fig_trap.add_vline(x=TRAP_DIST_MI, line_dash="dash", line_color="red",
                       annotation_text=f"Trap exit ({TRAP_DIST_MI:.3f} mi / 5280 ft)",
                       annotation_position="top left")
    fig_trap.add_vrect(x0=0, x1=TRAP_DIST_MI, fillcolor="rgba(0,200,0,0.06)",
                       line_width=0, annotation_text="Measured section",
                       annotation_position="top left")

    fig_trap.update_layout(
        xaxis_title="Distance from trap entry (mi)  [negative = run-in, positive = run-out]",
        yaxis_title="Speed (mph)",
        height=420,
        legend=dict(orientation="v", x=1.01, font_size=11),
        margin=dict(r=180),
    )
    st.plotly_chart(fig_trap, use_container_width=True)

    st.divider()

    # Acceleration and jerk distributions per run (measured section only)
    st.subheader("Acceleration Distribution by Run (measured section only)")
    accel_fig = go.Figure()
    for i, r in enumerate(runs):
        if "_measured_section" not in r:
            continue
        moving = r["_measured_section"]
        accel_fig.add_trace(go.Box(
            y=moving["accel_ft_s2"],
            name=f"R{i+1} ~{r['target_speed']} mph",
            marker_color=colors[i % len(colors)],
            boxmean="sd",
        ))
    accel_fig.update_layout(
        yaxis_title="Acceleration (ft/s²)",
        height=380,
        showlegend=False,
    )
    st.plotly_chart(accel_fig, use_container_width=True)

    st.subheader("Jerk Distribution by Run (measured section only)")
    jerk_fig = go.Figure()
    for i, r in enumerate(runs):
        if "_measured_section" not in r:
            continue
        moving = r["_measured_section"]
        jerk_fig.add_trace(go.Box(
            y=moving["jerk_ft_s3"],
            name=f"R{i+1} ~{r['target_speed']} mph",
            marker_color=colors[i % len(colors)],
            boxmean="sd",
        ))
    jerk_fig.update_layout(
        yaxis_title="Jerk (ft/s³)",
        height=380,
        showlegend=False,
    )
    st.plotly_chart(jerk_fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Run detail
# ═══════════════════════════════════════════════════════════════════════════════
with tab_detail:
    if not runs:
        st.warning("No exercise runs found. Try adjusting the settings in the sidebar.")
        st.stop()

    run_labels = [
        f"Run {i+1}: {r['direction']}  ~{r['target_speed']} mph  "
        f"[{run_start_type(r)} start | {run_finish_type(r)} finish]  "
        f"@ {r['start_time']}  ({r['start_elapsed']}–{r['end_elapsed']} s)"
        for i, r in enumerate(runs)
    ]
    sel_idx = st.selectbox("Select run", range(len(runs)), format_func=lambda i: run_labels[i])
    r = runs[sel_idx]

    # Use cached derivatives if available, else compute
    if "_grp_with_derivs" in r:
        grp = r["_grp_with_derivs"].copy()
    else:
        grp = compute_derivatives(r["data"].copy(), smooth_window)

    t0 = grp["Elapsed time (sec)"].iloc[0]
    d0 = grp["Distance (mi)"].iloc[0]
    grp["t_rel"] = grp["Elapsed time (sec)"] - t0
    grp["d_rel_mi"] = grp["Distance (mi)"] - d0

    moving = grp[grp["Speed (mph)"] >= min_speed]
    spd_err = moving["Speed (mph)"] - r["target_speed"]

    # Precompute metrics
    mean_spd = moving["Speed (mph)"].mean()
    rms_err = float(np.sqrt((spd_err ** 2).mean()))
    max_accel_fps2 = float(moving["accel_ft_s2"].abs().max())
    max_accel_g = float(moving["accel_g"].abs().max())
    max_jerk = float(moving["jerk_ft_s3"].abs().max())

    # Metrics row
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Start time", r["start_time"])
    c2.metric("Target speed", f"{r['target_speed']} mph")
    c3.metric("Mean speed", f"{mean_spd:.1f} mph")
    c4.metric("Speed RMS error", f"{rms_err:.2f} mph")
    c5.metric("Max |accel|", f"{max_accel_fps2:.2f} ft/s² ({max_accel_g:.3f} g)")
    c6.metric("Max |jerk|", f"{max_jerk:.2f} ft/s3")
    c7.metric("Distance", f"{r['distance_mi']:.3f} mi")

    use_time = st.radio(
        "X axis", ["Distance from trap entry", "Time from trap entry"],
        horizontal=True,
    ) == "Time from trap entry"

    # ── Trap crossing positions for this run ──────────────────────────────────
    going_west  = r["direction"] == "West"
    entry_lon_r = trap_east if going_west else trap_west
    exit_lon_r  = trap_west if going_west else trap_east
    entry_d_raw = find_crossing_dist(grp, entry_lon_r, going_west)
    exit_d_raw  = find_crossing_dist(grp, exit_lon_r,  going_west)

    # Geometric fallback: trap is exactly TRAP_DIST_MI (5280 ft = 1 mile).
    if entry_d_raw is None and exit_d_raw is not None:
        entry_d_raw = exit_d_raw - TRAP_DIST_MI
    if exit_d_raw is None and entry_d_raw is not None:
        exit_d_raw = entry_d_raw + TRAP_DIST_MI


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Transit Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab_transit:
    if df_transit.empty:
        st.info("No transit data found outside the exercise area.")
        st.stop()

    colors_transit = {"To Exercise": "#457b9d", "From Exercise": "#e76f51"}

    # ── Summary table ─────────────────────────────────────────────────────────
    def leg_stats(leg_df: pd.DataFrame, label: str) -> dict:
        moving = leg_df[leg_df["Speed (mph)"] > 2]
        if moving.empty:
            return {}
        dist  = leg_df["Distance (mi)"].iloc[-1] - leg_df["Distance (mi)"].iloc[0]
        dur_s = leg_df["Elapsed time (sec)"].iloc[-1] - leg_df["Elapsed time (sec)"].iloc[0]
        return {
            "Leg": label,
            "Distance (mi)": round(abs(dist), 2),
            "Duration (min)": round(dur_s / 60, 1),
            "Avg speed (mph)": round(moving["Speed (mph)"].mean(), 1),
            "Max speed (mph)": round(moving["Speed (mph)"].max(), 1),
        }

    stats_rows = [s for s in [leg_stats(df_outbound, "To Exercise"),
                               leg_stats(df_inbound,  "From Exercise")] if s]
    if stats_rows:
        st.subheader("Transit Summary")
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # ── Compute derivatives per leg ───────────────────────────────────────────
    tr_legs = []
    for leg_label, leg_df, color in [
        ("To Exercise",   df_outbound, "#457b9d"),
        ("From Exercise", df_inbound,  "#e76f51"),
    ]:
        if leg_df.empty:
            continue
        sub = compute_derivatives(leg_df.copy(), smooth_window)
        t0  = sub["Elapsed time (sec)"].iloc[0]
        d0  = sub["Distance (mi)"].iloc[0]
        sub["t_rel"] = sub["Elapsed time (sec)"] - t0
        sub["d_rel"] = sub["Distance (mi)"] - d0
        tr_legs.append({"label": leg_label, "df": sub, "color": color})

    # ── X-axis selector ───────────────────────────────────────────────────────
    tr_use_time = st.radio(
        "X axis", ["Distance from leg start", "Time from leg start"],
        horizontal=True, key="transit_xaxis",
    ) == "Time from leg start"
    x_col   = "t_rel" if tr_use_time else "d_rel"
    x_label = ("Time from leg start (s)"
                if tr_use_time else "Distance from leg start (mi)")

    # ── 3-panel subplot: Speed / Acceleration / Jerk ──────────────────────────
    fig_tr = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.38, 0.31, 0.31],
        vertical_spacing=0.04,
        subplot_titles=["Speed (mph)", "Acceleration (ft/s²)", "Jerk (ft/s³)"],
    )

    _all_spd = _all_acc = _all_jrk = np.array([])
    for leg in tr_legs:
        sub   = leg["df"]
        color = leg["color"]
        label = leg["label"]
        x     = sub[x_col].to_numpy()
        moving_mask = sub["Speed (mph)"] > 2

        # Row 1 — Speed: raw (faint) + smoothed
        fig_tr.add_trace(go.Scatter(
            x=x, y=sub["Speed (mph)"],
            name=f"{label} (raw)", mode="lines",
            line=dict(color=color, width=1), opacity=0.35,
            showlegend=False,
        ), row=1, col=1)
        fig_tr.add_trace(go.Scatter(
            x=x, y=sub["speed_smooth"],
            name=label, mode="lines",
            line=dict(color=color, width=2),
        ), row=1, col=1)

        # Row 2 — Acceleration
        fig_tr.add_trace(go.Scatter(
            x=x, y=sub["accel_ft_s2"],
            name=f"{label} accel", mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
        ), row=2, col=1)

        # Row 3 — Jerk
        fig_tr.add_trace(go.Scatter(
            x=x, y=sub["jerk_ft_s3"],
            name=f"{label} jerk", mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
        ), row=3, col=1)

        mv = sub[moving_mask]
        _all_spd = np.concatenate([_all_spd, mv["speed_smooth"].to_numpy()])
        _all_acc = np.concatenate([_all_acc, mv["accel_ft_s2"].to_numpy()])
        _all_jrk = np.concatenate([_all_jrk, mv["jerk_ft_s3"].to_numpy()])

    fig_tr.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    fig_tr.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
    fig_tr.update_xaxes(title_text=x_label, row=3, col=1)
    fig_tr.update_yaxes(
        title_text="mph",
        range=_sym_range(_all_spd, pad=0.08, min_half=5.0),
        row=1, col=1,
    )
    fig_tr.update_yaxes(
        title_text="ft/s²",
        range=_pct_range(_all_acc, pct=1, pad=0.15, min_half=1.0, zero=True),
        row=2, col=1,
    )
    fig_tr.update_yaxes(
        title_text="ft/s³",
        range=_pct_range(_all_jrk, pct=1, pad=0.15, min_half=1.0, zero=True),
        row=3, col=1,
    )
    fig_tr.update_layout(height=820, legend=dict(orientation="h", y=-0.06))
    st.plotly_chart(fig_tr, use_container_width=True)

# ── Idealized Run comparison — continues inside the Run Detail tab ─────────────
with tab_detail:
    st.divider()
    st.subheader("Idealized Run Comparison")

    # Reuse variables already computed above for the selected run
    ri           = r
    grp_i        = grp
    going_west_i = going_west
    entry_d_i    = entry_d_raw
    exit_d_i     = exit_d_raw

    # Detect flying vs standstill start based on speed at trap entry
    if entry_d_i is not None:
        d_arr_i = grp_i["Distance (mi)"].to_numpy()
        v_arr_i = grp_i["speed_smooth"].to_numpy()
        idx_entry = int(np.argmax(d_arr_i >= entry_d_i))
        speed_at_entry = float(np.interp(entry_d_i, d_arr_i, v_arr_i))
    else:
        speed_at_entry = float(grp_i["Speed (mph)"].iloc[:5].mean())
    is_flying  = speed_at_entry >= 0.5 * ri["target_speed"]
    start_type = "Flying start" if is_flying else "Standstill start"

    # Finish type: speed at trap exit
    # Use exit_d_i which already has the geometric fallback (entry + 1 mi) applied,
    # matching the logic in run_finish_type().
    _d_i        = grp_i["Distance (mi)"].to_numpy()
    _v_smooth_i = grp_i["speed_smooth"].to_numpy()
    if exit_d_i is not None:
        speed_at_exit = float(np.interp(exit_d_i, _d_i, _v_smooth_i,
                                        left=_v_smooth_i[0], right=_v_smooth_i[-1]))
    else:
        speed_at_exit = float(_v_smooth_i[-min(5, len(_v_smooth_i)):].mean())
    if speed_at_exit < 5:
        finish_type = "Standstill finish"
    elif speed_at_exit >= ri["target_speed"] - 5:
        finish_type = "Flying finish"
    else:
        finish_type = "Unknown finish"
    is_flying_finish = finish_type == "Flying finish"

    # ── Selected run aligned at trap entry ────────────────────────────────────
    _d_arr = grp_i["Distance (mi)"].to_numpy()
    _t_arr = grp_i["Elapsed time (sec)"].to_numpy()

    if entry_d_i is not None:
        d_aligned_i = (_d_arr - entry_d_i)
        t_at_entry  = float(np.interp(entry_d_i, _d_arr, _t_arr))
    else:
        d_aligned_i = (_d_arr - _d_arr[0])
        t_at_entry  = float(_t_arr[0])
    t_aligned_i = _t_arr - t_at_entry   # seconds from trap entry

    v_actual     = grp_i["speed_smooth"].to_numpy()
    accel_actual = grp_i["accel_ft_s2"].to_numpy()
    jerk_actual  = grp_i["jerk_ft_s3"].to_numpy()

    GRID_N = 600
    d_run_min = float(d_aligned_i.min())
    d_run_max = float(d_aligned_i.max())

    if is_flying:
        # ── Flying start: ideal = constant target speed, zero accel/jerk ─────
        col_info1, col_info2 = st.columns(2)
        col_info1.info(
            f"Start: **{start_type}** (speed at entry ≈ {speed_at_entry:.1f} mph)  |  "
            f"Finish: **{finish_type}** (speed at exit ≈ {speed_at_exit:.1f} mph)"
        )
        col_info2.info(
            f"Flying start — ideal is constant **{ri['target_speed']} mph** through the "
            f"trap" + (" then decelerates to rest." if not is_flying_finish else " with zero acceleration and jerk.")
        )

        d_grid         = np.linspace(d_run_min, d_run_max, GRID_N)
        t_grid         = d_grid / float(ri["target_speed"]) * 3600
        v_ideal_smooth = np.full(GRID_N, float(ri["target_speed"]))
        accel_ideal    = np.zeros(GRID_N)
        jerk_ideal     = np.zeros(GRID_N)
        v_band_lo      = None
        v_band_hi      = None
        residual_label = f"Speed Residual — Actual minus Target ({ri['target_speed']} mph)"
        speed_residual = v_actual - float(ri["target_speed"])

    else:
        # ── Standing start: acceleration model from ALL standing-start runs ───
        all_standing = [r for r in runs if run_start_type(r) == "Standing"]
        peer_labels_str = ", ".join(f"R{runs.index(r)+1} {r['direction']}" for r in all_standing)

        col_info1, col_info2 = st.columns(2)
        col_info1.info(
            f"Start: **{start_type}** (speed at entry ≈ {speed_at_entry:.1f} mph)  |  "
            f"Finish: **{finish_type}** (speed at exit ≈ {speed_at_exit:.1f} mph)"
        )
        col_info2.info(
            f"Acceleration model built from {len(all_standing)} standing-start run(s): "
            f"{peer_labels_str}"
        )

        # ── Acceleration model: direct speed-profile median ───────────────────
        # Collect normalised speed-vs-distance profiles from every standing-start
        # peer run, aligned at trap entry (d=0).  Taking the pointwise median
        # avoids ODE integration and the noise that comes from differentiating
        # GPS speed data twice to get acceleration.
        target_f       = float(ri["target_speed"])
        trap_exit_d_ext = (exit_d_i - entry_d_i) \
                          if (entry_d_i is not None and exit_d_i is not None) else 0.25

        accel_profs = []   # list of (d_aligned, v_normalised) arrays
        for r_p in all_standing:
            grp_p = r_p.get("_grp_with_derivs")
            if grp_p is None:
                grp_p = compute_derivatives(r_p["data"].copy(), smooth_window)
            gw_p        = r_p["direction"] == "West"
            entry_lon_p = trap_east if gw_p else trap_west
            entry_d_p   = find_crossing_dist(grp_p, entry_lon_p, gw_p)
            if entry_d_p is None:
                continue
            d_p   = grp_p["Distance (mi)"].to_numpy()
            v_col = "speed_smooth" if "speed_smooth" in grp_p.columns else "Speed (mph)"
            v_p   = grp_p[v_col].to_numpy()
            vt_p  = float(r_p["target_speed"])

            d_aln = d_p - entry_d_p
            # Keep from slightly before entry through to target speed
            phase = (d_aln >= -0.05) & (v_p < 1.01 * vt_p)
            if phase.sum() < 5:
                continue
            d_sl  = d_aln[phase]
            v_nrm = np.clip(v_p[phase] / vt_p, 0.0, 1.05)
            s_idx = np.argsort(d_sl)
            accel_profs.append((d_sl[s_idx], v_nrm[s_idx]))

        if not accel_profs:
            st.warning("Not enough standing-start data to build acceleration model.")
            st.stop()

        # Common distance grid from trap entry to just past trap exit
        _d_end_acc = max(p[0][-1] for p in accel_profs)
        D_IDEAL    = np.linspace(0.0, min(_d_end_acc, trap_exit_d_ext * 1.5), 400)

        _all_v_acc = np.vstack([
            np.clip(np.interp(D_IDEAL, d, v, left=v[0], right=1.0), 0.0, 1.05)
            for d, v in accel_profs
        ])
        v_nrm_ideal    = np.median(_all_v_acc, axis=0)
        v_ideal_smooth = np.clip(v_nrm_ideal * target_f, 0.0, target_f)
        d_grid         = D_IDEAL

        # Distance → time via the ideal speed (avoids ODE integration noise)
        _dd_acc  = np.diff(d_grid, prepend=d_grid[0])
        _v_safe  = np.maximum(v_ideal_smooth, 0.5)         # mph, avoid /0
        t_grid   = np.cumsum(_dd_acc / _v_safe * 3600.0)   # seconds from trap entry

        # Smooth derivatives from the ideal speed profile
        accel_ideal = smooth_series(
            np.gradient(v_ideal_smooth * MPH_S_TO_FT_S2, t_grid), smooth_window
        )
        jerk_ideal  = smooth_series(np.gradient(accel_ideal, t_grid), smooth_window)

        v_band_lo = None
        v_band_hi = None

        residual_label = "Speed Residual — Actual minus Ideal (mph)"
        v_ideal_at_actual = np.interp(d_aligned_i, d_grid, v_ideal_smooth,
                                       left=np.nan, right=np.nan)
        speed_residual = v_actual - v_ideal_at_actual

    # ── Deceleration model (standing finish) ─────────────────────────────────
    # Same direct-profile approach as the acceleration model: collect normalised
    # speed-vs-distance profiles from all standing-finish runs, aligned at the
    # trap exit (d=0), then take the pointwise median.
    d_decel_ideal = v_decel_ideal = t_decel_ideal = None
    accel_decel_ideal = jerk_decel_ideal = None

    if not is_flying_finish:
        all_standing_finish = [r for r in runs if run_finish_type(r) == "Standing"]
        decel_profs = []
        for r_p in all_standing_finish:
            grp_p = r_p.get("_grp_with_derivs")
            if grp_p is None:
                grp_p = compute_derivatives(r_p["data"].copy(), smooth_window)
            gw_p        = r_p["direction"] == "West"
            entry_lon_p = trap_east if gw_p else trap_west
            exit_lon_p  = trap_west if gw_p else trap_east
            vt_p        = float(r_p["target_speed"])

            entry_d_p = find_crossing_dist(grp_p, entry_lon_p, gw_p)
            exit_d_p  = find_crossing_dist(grp_p, exit_lon_p,  gw_p)
            # Geometric fallback: trap is exactly TRAP_DIST_MI (5280 ft = 1 mile)
            if exit_d_p is None and entry_d_p is not None:
                exit_d_p = entry_d_p + TRAP_DIST_MI
            if exit_d_p is None:
                continue

            d_p   = grp_p["Distance (mi)"].to_numpy()
            v_col = "speed_smooth" if "speed_smooth" in grp_p.columns else "Speed (mph)"
            v_p   = grp_p[v_col].to_numpy()

            d_aln_exit = d_p - exit_d_p
            phase = d_aln_exit >= -0.01   # from exit onward
            if phase.sum() < 3:
                continue
            d_sl  = d_aln_exit[phase]
            v_nrm = np.clip(v_p[phase] / vt_p, 0.0, 1.1)
            s_idx = np.argsort(d_sl)
            d_sl, v_nrm = d_sl[s_idx], v_nrm[s_idx]

            # Skip if the car wasn't near target speed at exit (partial run)
            if v_nrm[0] < 0.5:
                continue
            decel_profs.append((d_sl, v_nrm))

        if decel_profs:
            _d_end_dec = max(p[0][-1] for p in decel_profs)
            D_DECEL    = np.linspace(0.0, _d_end_dec, 300)

            _all_v_dec = np.vstack([
                np.clip(np.interp(D_DECEL, d, v, left=v[0], right=0.0), 0.0, 1.1)
                for d, v in decel_profs
            ])
            _v_nrm_dec    = np.median(_all_v_dec, axis=0)
            _target_f     = float(ri["target_speed"])
            v_decel_ideal = np.clip(_v_nrm_dec * _target_f, 0.0, _target_f)

            # Align decel distance to trap-entry coordinate system
            _d0_dec       = (exit_d_i - entry_d_i) \
                            if (exit_d_i is not None and entry_d_i is not None) \
                            else float(d_grid[-1])
            d_decel_ideal = D_DECEL + _d0_dec

            # Distance → time via the ideal decel speed
            _t0_dec       = float(np.interp(_d0_dec, d_grid, t_grid))
            _dd_dec       = np.diff(d_decel_ideal, prepend=d_decel_ideal[0])
            _v_s_dec      = np.maximum(v_decel_ideal, 0.1)
            t_decel_ideal = _t0_dec + np.cumsum(_dd_dec / _v_s_dec * 3600.0)

            # Smooth derivatives
            accel_decel_ideal = smooth_series(
                np.gradient(v_decel_ideal * MPH_S_TO_FT_S2, t_decel_ideal), smooth_window
            )
            jerk_decel_ideal  = smooth_series(
                np.gradient(accel_decel_ideal, t_decel_ideal), smooth_window
            )

    # ── Cumulative time error (measured section only) ─────────────────────────
    # Integrate (1/v_ideal − 1/v_actual) × dd × 3600 s/hr between the traps.
    # Positive = ahead of ideal (running fast), negative = behind.
    trap_exit_d = (exit_d_i - entry_d_i) if (entry_d_i is not None and exit_d_i is not None) else None
    trap_mask   = (d_aligned_i >= 0) & (d_aligned_i <= (trap_exit_d if trap_exit_d is not None else np.inf))

    sort_idx_i  = np.argsort(d_aligned_i[trap_mask])
    d_sorted    = d_aligned_i[trap_mask][sort_idx_i]
    v_act_sort  = v_actual[trap_mask][sort_idx_i]
    v_act_sort  = np.where(v_act_sort > 1.0, v_act_sort, np.nan)
    v_id_at_d   = np.interp(d_sorted, d_grid, v_ideal_smooth, left=np.nan, right=np.nan)
    v_id_at_d   = np.where(v_id_at_d > 1.0, v_id_at_d, np.nan)
    dd          = np.diff(d_sorted, prepend=d_sorted[0])
    dt_diff     = (1.0 / v_id_at_d - 1.0 / v_act_sort) * dd * 3600.0  # s
    dt_diff     = np.nan_to_num(dt_diff, nan=0.0)
    cum_time_err = np.cumsum(dt_diff)   # positive = ahead of ideal

    # x values for plotting: the trap-section points in original order
    d_trap_plot = d_aligned_i[trap_mask]
    cum_time_err_plot = np.interp(d_trap_plot, d_sorted, cum_time_err)

    # ── Unified x-axis (controlled by radio) ─────────────────────────────────
    x_actual    = t_aligned_i      if use_time else d_aligned_i
    x_ideal     = t_grid           if use_time else d_grid
    x_trap_plot = t_aligned_i[trap_mask] if use_time else d_trap_plot
    x_label_str = ("Time from trap entry (s)"
                   if use_time else
                   "Distance from trap entry (mi)  [negative = run-in]")

    # Trap exit x position for vline/vrect
    if exit_d_i is not None and entry_d_i is not None:
        _exit_d_aln = exit_d_i - entry_d_i
        exit_x_plot = float(np.interp(_exit_d_aln, d_aligned_i, t_aligned_i)) \
                      if use_time else _exit_d_aln
    else:
        exit_x_plot = None

    # Decel ideal x-axis (None if no decel model)
    x_decel = (t_decel_ideal if use_time else d_decel_ideal) \
              if d_decel_ideal is not None else None

    # ── 5-panel subplot ───────────────────────────────────────────────────────
    fig_ideal = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        row_heights=[0.19, 0.25, 0.18, 0.18, 0.20],
        vertical_spacing=0.04,
        subplot_titles=[
            "Cumulative Time Error vs Ideal (s)  [+ = ahead, − = behind]",
            "Speed (mph)",
            "Acceleration (ft/s²)",
            "Jerk (ft/s³)",
            residual_label,
        ],
    )

    # Row 1: Cumulative time error (trap entry → trap exit only)
    fig_ideal.add_trace(go.Scatter(
        x=x_trap_plot, y=cum_time_err_plot,
        name="Cumulative time error", mode="lines",
        line=dict(color="crimson", width=2),
        fill="tozeroy",
        fillcolor="rgba(220,20,60,0.12)",
    ), row=1, col=1)
    fig_ideal.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    # Row 2: Speed
    fig_ideal.add_trace(go.Scatter(
        x=x_actual, y=grp_i["Speed (mph)"],
        name="Actual (raw)", mode="lines",
        line=dict(color="lightsteelblue", width=1), opacity=0.5,
    ), row=2, col=1)
    # Peer speed band (standing start only)
    if v_band_lo is not None and v_band_hi is not None:
        fig_ideal.add_trace(go.Scatter(
            x=np.concatenate([x_ideal, x_ideal[::-1]]),
            y=np.concatenate([v_band_hi, v_band_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(255,165,0,0.15)",
            line=dict(width=0),
            name="Peer range",
            showlegend=True,
        ), row=2, col=1)
    ideal_name = "Target speed" if is_flying else "Ideal accel profile"
    fig_ideal.add_trace(go.Scatter(
        x=x_ideal, y=v_ideal_smooth,
        name=ideal_name, mode="lines",
        line=dict(color="orange", width=2.5, dash="dash"),
    ), row=2, col=1)
    if x_decel is not None:
        fig_ideal.add_trace(go.Scatter(
            x=x_decel, y=v_decel_ideal,
            name="Ideal decel", mode="lines",
            line=dict(color="darkorange", width=2.5, dash="dot"),
            showlegend=True,
        ), row=2, col=1)
    fig_ideal.add_trace(go.Scatter(
        x=x_actual, y=v_actual,
        name="Actual (smooth)", mode="lines",
        line=dict(color="royalblue", width=2),
    ), row=2, col=1)
    fig_ideal.add_hline(
        y=ri["target_speed"], line_dash="dot", line_color="gray",
        annotation_text=f"Target {ri['target_speed']} mph",
        annotation_position="top right", row=2, col=1,
    )

    # Row 3: Acceleration
    fig_ideal.add_trace(go.Scatter(
        x=x_ideal, y=accel_ideal,
        name="Ideal accel", mode="lines",
        line=dict(color="darkorange", width=2, dash="dash"),
    ), row=3, col=1)
    if x_decel is not None:
        fig_ideal.add_trace(go.Scatter(
            x=x_decel, y=accel_decel_ideal,
            name="Ideal decel accel", mode="lines",
            line=dict(color="darkorange", width=2, dash="dot"),
            showlegend=False,
        ), row=3, col=1)
    fig_ideal.add_trace(go.Scatter(
        x=x_actual, y=accel_actual,
        name="Actual accel", mode="lines",
        line=dict(color="#e76f51", width=2),
    ), row=3, col=1)
    fig_ideal.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)

    # Row 4: Jerk
    fig_ideal.add_trace(go.Scatter(
        x=x_ideal, y=jerk_ideal,
        name="Ideal jerk", mode="lines",
        line=dict(color="darkcyan", width=2, dash="dash"),
    ), row=4, col=1)
    if x_decel is not None:
        fig_ideal.add_trace(go.Scatter(
            x=x_decel, y=jerk_decel_ideal,
            name="Ideal decel jerk", mode="lines",
            line=dict(color="darkcyan", width=2, dash="dot"),
            showlegend=False,
        ), row=4, col=1)
    fig_ideal.add_trace(go.Scatter(
        x=x_actual, y=jerk_actual,
        name="Actual jerk", mode="lines",
        line=dict(color="#2a9d8f", width=2),
    ), row=4, col=1)
    fig_ideal.add_hline(y=0, line_dash="dot", line_color="gray", row=4, col=1)

    # Row 5: Speed residual
    fig_ideal.add_trace(go.Scatter(
        x=x_actual, y=speed_residual,
        name="Speed residual", mode="lines",
        line=dict(color="mediumpurple", width=2),
        fill="tozeroy",
        fillcolor="rgba(147,112,219,0.15)",
    ), row=5, col=1)
    fig_ideal.add_hline(y=0, line_dash="dot", line_color="gray", row=5, col=1)

    # Trap markers on all rows (x=0 is always trap entry in both axis modes)
    fig_ideal.add_vline(x=0, line_dash="dash", line_color="green",
                        annotation_text="Trap entry", annotation_position="top right")
    if exit_x_plot is not None:
        fig_ideal.add_vline(x=exit_x_plot, line_dash="dash", line_color="red",
                            annotation_text="Trap exit", annotation_position="top left")
        fig_ideal.add_vrect(x0=0, x1=exit_x_plot,
                            fillcolor="rgba(0,200,0,0.06)", line_width=0)

    fig_ideal.update_xaxes(title_text=x_label_str, row=5, col=1)
    if len(cum_time_err_plot) > 0:
        _cte_lo = float(np.nanmin(cum_time_err_plot))
        _cte_hi = float(np.nanmax(cum_time_err_plot))
        _cte_pad = max(abs(_cte_hi - _cte_lo) * 0.15, 0.5)
        _cte_range = [_cte_lo - _cte_pad, _cte_hi + _cte_pad]
    else:
        _cte_range = [-1, 1]
    fig_ideal.update_yaxes(title_text="sec", range=_cte_range, row=1, col=1)
    # Mask ideal-grid arrays to trap section for range calculations
    _grid_trap = (d_grid >= 0) & (d_grid <= (trap_exit_d if trap_exit_d is not None else np.inf))

    # Row 2 — Speed: trap-section data only
    _spd_vals = np.concatenate([
        grp_i["Speed (mph)"].to_numpy()[trap_mask],
        v_actual[trap_mask],
        v_ideal_smooth[_grid_trap],
        v_band_lo[_grid_trap] if v_band_lo is not None else np.array([]),
        v_band_hi[_grid_trap] if v_band_hi is not None else np.array([]),
        np.array([float(ri["target_speed"])]),
    ])
    _spd_range = _sym_range(_spd_vals, pad=0.08, min_half=5.0)

    # Row 3 — Acceleration: percentile-clipped, trap section only
    _accel_vals = np.concatenate([accel_actual[trap_mask],
                                   accel_ideal[_grid_trap]])
    _accel_range = _pct_range(_accel_vals, pct=1, pad=0.15, min_half=1.0, zero=True)

    # Row 4 — Jerk: percentile-clipped, trap section only
    _jerk_vals = np.concatenate([jerk_actual[trap_mask],
                                  jerk_ideal[_grid_trap]])
    _jerk_range = _pct_range(_jerk_vals, pct=1, pad=0.15, min_half=1.0, zero=True)

    # Row 5 — Speed residual: trap section only
    _res_vals = speed_residual[trap_mask]
    _res_range = _sym_range(_res_vals, pad=0.15, min_half=1.0, zero=True)

    fig_ideal.update_yaxes(title_text="mph",   range=_spd_range,   row=2, col=1)
    fig_ideal.update_yaxes(title_text="ft/s²", range=_accel_range, row=3, col=1)
    fig_ideal.update_yaxes(title_text="ft/s³", range=_jerk_range,  row=4, col=1)
    fig_ideal.update_yaxes(title_text="Δ mph", range=_res_range,   row=5, col=1)
    fig_ideal.update_layout(height=1100, legend=dict(orientation="h", y=-0.04))
    st.plotly_chart(fig_ideal, use_container_width=True)

    # ── Comparison metrics table (measured section only) ──────────────────────
    st.subheader("Measured Section: Actual vs Ideal")
    if entry_d_i is not None and exit_d_i is not None:
        exit_d_aln = exit_d_i - entry_d_i
        ms_mask = (d_aligned_i >= 0) & (d_aligned_i <= exit_d_aln)
    else:
        ms_mask = np.ones(len(d_aligned_i), dtype=bool)

    d_ms          = d_aligned_i[ms_mask]
    v_actual_ms   = v_actual[ms_mask]
    accel_ms      = accel_actual[ms_mask]
    jerk_ms       = jerk_actual[ms_mask]
    v_ideal_ms    = np.interp(d_ms, d_grid, v_ideal_smooth)
    accel_ideal_ms = np.interp(d_ms, d_grid, accel_ideal)
    jerk_ideal_ms  = np.interp(d_ms, d_grid, jerk_ideal)

    def _cmp(metric, actual_val, ideal_val):
        delta = actual_val - ideal_val
        return {"Metric": metric, "Actual": actual_val, "Ideal": ideal_val,
                "Δ (Actual − Ideal)": delta}

    cmp_rows = [
        _cmp("Mean speed (mph)",
             round(float(np.nanmean(v_actual_ms)), 2),
             round(float(np.nanmean(v_ideal_ms)), 2)),
        _cmp("Speed vs target RMS (mph)",
             round(float(np.sqrt(np.nanmean((v_actual_ms - ri["target_speed"])**2))), 3),
             round(float(np.sqrt(np.nanmean((v_ideal_ms  - ri["target_speed"])**2))), 3)),
        _cmp("Max |Accel| (ft/s²)",
             round(float(np.nanmax(np.abs(accel_ms))), 2),
             round(float(np.nanmax(np.abs(accel_ideal_ms))), 2)),
        _cmp("RMS Accel (ft/s²)",
             round(float(np.sqrt(np.nanmean(accel_ms**2))), 3),
             round(float(np.sqrt(np.nanmean(accel_ideal_ms**2))), 3)),
        _cmp("Max |Jerk| (ft/s³)",
             round(float(np.nanmax(np.abs(jerk_ms))), 2),
             round(float(np.nanmax(np.abs(jerk_ideal_ms))), 2)),
        _cmp("RMS Jerk (ft/s³)",
             round(float(np.sqrt(np.nanmean(jerk_ms**2))), 3),
             round(float(np.sqrt(np.nanmean(jerk_ideal_ms**2))), 3)),
    ]
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    # g-force reference
    with st.expander("Reference: g-force scale"):
        st.markdown("""
| Acceleration | ft/s² | Example |
|---|---|---|
| 0.05 g | 1.6 | Gentle highway merge |
| 0.10 g | 3.2 | Moderate acceleration |
| 0.20 g | 6.4 | Spirited driving |
| 0.30 g | 9.7 | Firm braking |
| 0.50 g | 16.1 | Hard braking |
| 1.00 g | 32.2 | Maximum traction limit |

**Jerk** (rate of change of acceleration) matters for ride quality and the
ability to make smooth speed adjustments — critical in a time-distance rally
where abrupt throttle/brake inputs cause overshoot.
        """)

    # Raw data expander
    with st.expander("Raw data for this run"):
        show_cols = ["Elapsed time (sec)", "t_rel", "d_rel_mi",
                     "Speed (mph)", "speed_smooth",
                     "accel_ft_s2", "accel_g", "jerk_ft_s3",
                     "Latitude", "Longitude", "Accuracy (ft)"]
        st.dataframe(grp[[c for c in show_cols if c in grp.columns]].round(4),
                     use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Settings
# ═══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    if not pq_files:
        st.warning("No parquet files found. Run `prepare_track_data.py` to create filtered parquets.")

    st.subheader("Exercise Area")
    st.slider(
        "Southern boundary (lat)", 29.30, 29.42, step=0.005, format="%.3f",
        help="Rows south of this latitude are treated as the exercise area.",
        key="cfg_lat",
    )

    st.divider()
    st.subheader("Run Detection")
    st.slider(
        "Smoothing window (samples)", 3, 31, step=2,
        help="Larger = smoother acceleration/jerk curves but less temporal resolution.",
        key="cfg_smooth",
    )
    st.slider(
        "Min speed to count as 'moving' (mph)", 1, 15,
        key="cfg_min_speed",
    )
    st.slider(
        "Min GPS samples per run", 10, 100,
        key="cfg_min_rows",
    )
    st.info("Changes take effect immediately on the next interaction.")