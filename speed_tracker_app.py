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
                          min_speed: float, min_rows: int) -> list[dict]:
    """
    Isolate the southern exercise area and split into individual runs.
    A new run boundary is declared when speed drops below min_speed OR when
    there is a time gap > 30 s between consecutive GPS fixes.
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
        max_spd  = moving["Speed (mph)"].max()
        # Use steady-state (cruise) portion — majority where speed ≥ 75% of max
        cruise   = moving[moving["Speed (mph)"] >= 0.75 * max_spd]
        cruise_spd = cruise["Speed (mph)"].mean() if not cruise.empty else mean_spd
        target = round(cruise_spd / 5) * 5
        lon_delta = grp["Longitude"].iloc[-1] - grp["Longitude"].iloc[0]
        direction = "West" if lon_delta < 0 else "East"
        distance = abs(grp["Distance (mi)"].iloc[-1] - grp["Distance (mi)"].iloc[0])
        runs.append({
            "seg_id": seg_id,
            "start_elapsed": int(grp["Elapsed time (sec)"].iloc[0]),
            "end_elapsed": int(grp["Elapsed time (sec)"].iloc[-1]),
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
    Return the cumulative odometer distance (mi) at which the run crosses trap_lon.
    For westbound runs longitude decreases; for eastbound it increases.
    Uses linear interpolation between the two bracketing GPS samples.
    Returns None if the trap is never crossed.
    """
    lon = grp["Longitude"].to_numpy()
    dist = grp["Distance (mi)"].to_numpy()

    if going_west:
        # Looking for first point where lon <= trap_lon (crossed from east)
        mask = lon <= trap_lon
    else:
        # Looking for first point where lon >= trap_lon (crossed from west)
        mask = lon >= trap_lon

    if not mask.any():
        return None

    idx = int(np.argmax(mask))   # first True
    if idx == 0:
        return float(dist[0])

    # Linear interpolation
    lon0, lon1 = lon[idx - 1], lon[idx]
    d0, d1 = dist[idx - 1], dist[idx]
    frac = (trap_lon - lon0) / (lon1 - lon0) if lon1 != lon0 else 0.0
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

st.title("Speed Tracker — Acceleration & Jerk Analysis")
st.caption("Practice run evaluation")

pq_files  = sorted(DATALOGS.glob("*.parquet"), reverse=True)
csv_files = sorted(DATALOGS.glob("*.csv"),     reverse=True)
track_files = pq_files or csv_files
if not track_files:
    st.error(f"No parquet or CSV files found in {DATALOGS}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    if not pq_files:
        st.warning("No parquet files found. Run `prepare_track_data.py` to create filtered parquets.")
    selected_file = st.selectbox(
        "Log file", track_files, format_func=lambda p: p.name
    )
    st.divider()
    st.subheader("Exercise area")
    lat_threshold = st.slider(
        "Southern boundary (lat)", 29.30, 29.42, 29.34, step=0.005, format="%.3f",
        help="Rows south of this latitude are treated as the exercise area."
    )
    st.subheader("Run detection")
    smooth_window = st.slider(
        "Smoothing window (samples)", 3, 31, 9, step=2,
        help="Larger = smoother acceleration/jerk curves but less temporal resolution."
    )
    min_speed = st.slider("Min speed to count as 'moving' (mph)", 1, 15, 3)
    min_rows = st.slider("Min GPS samples per run", 10, 100, 40)

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

# Slider range spans all exercise-area longitudes with a small margin
_lon_min = float(_df_ex["Longitude"].min()) - 0.001 if not _df_ex.empty else -95.670
_lon_max = float(_df_ex["Longitude"].max()) + 0.001 if not _df_ex.empty else -95.630

with st.sidebar:
    st.divider()
    st.subheader("Measured section traps")
    st.caption(
        f"Defaults estimated from {len(_rest)} at-rest GPS points in exercise area. "
        f"East: **{_trap_east_def}**, West: **{_trap_west_def}**"
    )
    trap_east = st.slider(
        "East trap lon", _lon_min, _lon_max, _trap_east_def, step=0.0005, format="%.4f",
        help="Eastern boundary of the measured distance (higher/less-negative lon)."
    )
    trap_west = st.slider(
        "West trap lon", _lon_min, _lon_max, _trap_west_def, step=0.0005, format="%.4f",
        help="Western boundary of the measured distance (lower/more-negative lon)."
    )
runs = segment_exercise_runs(df_raw, lat_threshold, min_speed, min_rows)

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


tab_map, tab_overview, tab_detail, tab_transit, tab_ideal = st.tabs(
    ["Track Map", "Runs Overview", "Run Detail", "Transit Analysis", "Idealized Run"]
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

        fig_map = px.scatter_map(
            df_plot,
            lat="Latitude", lon="Longitude",
            color="Zone",
            color_discrete_map={"Exercise Area": "#e63946", "Transit": "#457b9d"},
            hover_data={"Speed (mph)": ":.1f", "Time": True, "Date": True,
                        "Elapsed time (sec)": True, "Distance (mi)": ":.2f"},
            zoom=10,
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
            line=dict(color="lime", width=3),
            name="East trap",
            hoverinfo="name",
        ))
        # West trap boundary line
        fig_map.add_trace(go.Scattermap(
            lat=[trap_lat_lo, trap_lat_hi],
            lon=[trap_west, trap_west],
            mode="lines",
            line=dict(color="red", width=3),
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
    summary_rows = []
    for i, r in enumerate(runs):
        grp = r["data"].copy()
        grp = compute_derivatives(grp, smooth_window)
        r["_grp_with_derivs"] = grp   # cache for reuse in detail tab

        # Clip to the measured section between the two traps
        gw = r["direction"] == "West"
        en_lon = trap_east if gw else trap_west
        ex_lon = trap_west if gw else trap_east
        en_d = find_crossing_dist(grp, en_lon, gw)
        ex_d = find_crossing_dist(grp, ex_lon, gw)

        if en_d is not None and ex_d is not None:
            d_lo, d_hi = min(en_d, ex_d), max(en_d, ex_d)
            section = grp[
                (grp["Distance (mi)"] >= d_lo) & (grp["Distance (mi)"] <= d_hi)
            ]
        else:
            section = grp   # fallback: no trap data, use whole run

        section = section[section["Speed (mph)"] >= min_speed]
        spd_err = section["Speed (mph)"] - r["target_speed"]
        measured_dist = (d_hi - d_lo) if (en_d is not None and ex_d is not None) else r["distance_mi"]

        if section.empty:
            continue

        summary_rows.append({
            "Run": i + 1,
            "Direction": r["direction"],
            "Target (mph)": r["target_speed"],
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
        f"X = 0 at trap entry | vertical line = trap exit "
        f"(east trap {trap_east:.4f}, west trap {trap_west:.4f})"
    )
    fig_trap = go.Figure()
    trap_lengths = []
    for i, r in enumerate(runs):
        grp = r["data"].copy()
        going_west = r["direction"] == "West"
        # Entry trap: first boundary the car crosses entering the measured section
        entry_lon = trap_east if going_west else trap_west
        exit_lon  = trap_west if going_west else trap_east

        entry_dist = find_crossing_dist(grp, entry_lon, going_west)
        exit_dist  = find_crossing_dist(grp, exit_lon,  going_west)

        if entry_dist is None:
            continue

        d_aligned = grp["Distance (mi)"] - entry_dist

        if exit_dist is not None:
            trap_lengths.append(abs(exit_dist - entry_dist))

        fig_trap.add_trace(go.Scatter(
            x=d_aligned,
            y=grp["Speed (mph)"],
            mode="lines",
            name=f"R{i+1} {r['direction']} ~{r['target_speed']} mph",
            line=dict(color=colors[i % len(colors)], width=1.5),
            opacity=0.8,
        ))

    # Mark trap boundaries
    fig_trap.add_vline(x=0, line_dash="dash", line_color="green",
                       annotation_text="Trap entry", annotation_position="top right")
    if trap_lengths:
        med_len = float(np.median(trap_lengths))
        fig_trap.add_vline(x=med_len, line_dash="dash", line_color="red",
                           annotation_text=f"Trap exit (~{med_len:.3f} mi)",
                           annotation_position="top left")
        # Shade the measured section
        fig_trap.add_vrect(x0=0, x1=med_len, fillcolor="rgba(0,200,0,0.06)",
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
        f"[{run_start_type(r)} start]  ({r['start_elapsed']}–{r['end_elapsed']} s)"
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
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Target speed", f"{r['target_speed']} mph")
    c2.metric("Mean speed", f"{mean_spd:.1f} mph")
    c3.metric("Speed RMS error", f"{rms_err:.2f} mph")
    c4.metric("Max |accel|", f"{max_accel_fps2:.2f} ft/s² ({max_accel_g:.3f} g)")
    c5.metric("Max |jerk|", f"{max_jerk:.2f} ft/s3")
    c6.metric("Distance", f"{r['distance_mi']:.3f} mi")

    x_choice = st.radio("X axis", ["Time (s from run start)", "Distance (mi from run start)"],
                        horizontal=True)
    x_col = "t_rel" if "Time" in x_choice else "d_rel_mi"
    x_label = "Time from run start (s)" if "Time" in x_choice else "Distance from run start (mi)"

    # ── Trap crossing positions for this run ──────────────────────────────────
    going_west = r["direction"] == "West"
    entry_lon_r = trap_east if going_west else trap_west
    exit_lon_r  = trap_west if going_west else trap_east

    entry_d_raw = find_crossing_dist(grp, entry_lon_r, going_west)
    exit_d_raw  = find_crossing_dist(grp, exit_lon_r,  going_west)

    def d_raw_to_x(d_raw):
        """Convert a cumulative odometer distance to the active x-axis value."""
        if d_raw is None:
            return None
        if "Distance" in x_choice:
            return d_raw - d0
        # For time x-axis: interpolate t_rel at the given odometer distance
        d_arr = grp["Distance (mi)"].to_numpy()
        t_arr = grp["t_rel"].to_numpy()
        mask = d_arr >= d_raw
        if not mask.any():
            return None
        idx = int(np.argmax(mask))
        if idx == 0:
            return float(t_arr[0])
        dd = d_arr[idx] - d_arr[idx - 1]
        frac = (d_raw - d_arr[idx - 1]) / dd if dd != 0 else 0.0
        return float(t_arr[idx - 1] + frac * (t_arr[idx] - t_arr[idx - 1]))

    entry_x = d_raw_to_x(entry_d_raw)
    exit_x  = d_raw_to_x(exit_d_raw)

    # 3-panel subplot: speed / acceleration / jerk
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.06,
        subplot_titles=["Speed (mph)", "Acceleration (ft/s²  |  g)", "Jerk (ft/s³)"],
    )

    x = grp[x_col]

    # Speed
    fig.add_trace(go.Scatter(
        x=x, y=grp["Speed (mph)"],
        name="Raw speed", mode="lines",
        line=dict(color="lightsteelblue", width=1), opacity=0.5,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=grp["speed_smooth"],
        name="Smoothed speed", mode="lines",
        line=dict(color="royalblue", width=2),
    ), row=1, col=1)
    fig.add_hline(y=r["target_speed"], line_dash="dash", line_color="gray",
                  annotation_text=f"Target {r['target_speed']} mph",
                  annotation_position="top right", row=1, col=1)

    # Acceleration
    fig.add_trace(go.Scatter(
        x=x, y=grp["accel_ft_s2"],
        name="Accel (ft/s²)", mode="lines",
        line=dict(color="#e76f51", width=2),
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    # Second y-axis annotation for g units alongside ft/s²
    # (add as scatter on same axis with right-side label via annotation)

    # Jerk
    fig.add_trace(go.Scatter(
        x=x, y=grp["jerk_ft_s3"],
        name="Jerk (ft/s³)", mode="lines",
        line=dict(color="#2a9d8f", width=2),
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)

    # Trap entry / exit markers (apply to all subplots simultaneously)
    if entry_x is not None:
        fig.add_vline(x=entry_x, line_dash="dash", line_color="green",
                      annotation_text="Trap entry", annotation_position="top right")
    if exit_x is not None:
        fig.add_vline(x=exit_x, line_dash="dash", line_color="red",
                      annotation_text="Trap exit", annotation_position="top left")
    if entry_x is not None and exit_x is not None:
        fig.add_vrect(x0=min(entry_x, exit_x), x1=max(entry_x, exit_x),
                      fillcolor="rgba(0,200,0,0.06)", line_width=0,
                      annotation_text="Measured section", annotation_position="top left")

    fig.update_xaxes(title_text=x_label, row=3, col=1)
    fig.update_yaxes(title_text="mph", row=1, col=1)
    fig.update_yaxes(title_text="ft/s²", row=2, col=1)
    fig.update_yaxes(title_text="ft/s³", row=3, col=1)
    fig.update_layout(height=700, showlegend=True,
                      legend=dict(orientation="h", y=-0.08))
    st.plotly_chart(fig, use_container_width=True)

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
# Tab 4 — Transit Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab_transit:
    if df_transit.empty:
        st.info("No transit data found outside the exercise area.")
        st.stop()

    colors_transit = {"To Exercise": "#457b9d", "From Exercise": "#e76f51"}

    # ── Summary metrics ───────────────────────────────────────────────────────
    def leg_stats(leg_df: pd.DataFrame, label: str) -> dict:
        moving = leg_df[leg_df["Speed (mph)"] > 2]
        if moving.empty:
            return {}
        dist = leg_df["Distance (mi)"].iloc[-1] - leg_df["Distance (mi)"].iloc[0]
        dur_s = leg_df["Elapsed time (sec)"].iloc[-1] - leg_df["Elapsed time (sec)"].iloc[0]
        return {
            "Leg": label,
            "Distance (mi)": round(abs(dist), 2),
            "Duration (min)": round(dur_s / 60, 1),
            "Avg speed (mph)": round(moving["Speed (mph)"].mean(), 1),
            "Max speed (mph)": round(moving["Speed (mph)"].max(), 1),
            "Rows": len(leg_df),
        }

    stats_rows = [s for s in [leg_stats(df_outbound, "To Exercise"),
                               leg_stats(df_inbound,  "From Exercise")] if s]
    if stats_rows:
        st.subheader("Transit Summary")
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # ── Map ───────────────────────────────────────────────────────────────────
    st.subheader("Transit Route")
    tr_map_style = st.radio(
        "Transit map style", ["Street (carto)", "Satellite (ESRI)"],
        horizontal=True, key="transit_map_style", label_visibility="collapsed"
    )
    tr_satellite = tr_map_style == "Satellite (ESRI)"

    df_transit_plot = df_transit.iloc[::3].copy()  # downsample
    fig_tr_map = px.scatter_map(
        df_transit_plot,
        lat="Latitude", lon="Longitude",
        color="Leg",
        color_discrete_map=colors_transit,
        hover_data={"Speed (mph)": ":.1f", "Time": True, "Date": True,
                    "Elapsed time (sec)": True},
        zoom=10,
        map_style="white-bg" if tr_satellite else "carto-positron",
        title="Transit Route — To Exercise & From Exercise",
    )
    fig_tr_map.update_traces(marker=dict(size=3))
    if tr_satellite:
        fig_tr_map.update_layout(map=dict(layers=[{
            "below": "traces", "sourcetype": "raster",
            "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/"
                       "World_Imagery/MapServer/tile/{z}/{y}/{x}"],
            "sourceattribution": "Tiles &copy; Esri",
        }]))
    fig_tr_map.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_tr_map, use_container_width=True)

    # ── Speed vs distance ─────────────────────────────────────────────────────
    st.subheader("Speed Profile")
    fig_tr_spd = go.Figure()
    for leg_label, leg_df in [("To Exercise", df_outbound), ("From Exercise", df_inbound)]:
        if leg_df.empty:
            continue
        d0 = leg_df["Distance (mi)"].iloc[0]
        fig_tr_spd.add_trace(go.Scatter(
            x=leg_df["Distance (mi)"] - d0,
            y=leg_df["Speed (mph)"],
            mode="lines",
            name=leg_label,
            line=dict(color=colors_transit[leg_label], width=1.5),
            opacity=0.8,
        ))
    fig_tr_spd.update_layout(
        xaxis_title="Distance from leg start (mi)",
        yaxis_title="Speed (mph)",
        height=380,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_tr_spd, use_container_width=True)

    # ── Speed histogram ───────────────────────────────────────────────────────
    st.subheader("Speed Distribution")
    fig_hist = go.Figure()
    for leg_label, leg_df in [("To Exercise", df_outbound), ("From Exercise", df_inbound)]:
        moving = leg_df[leg_df["Speed (mph)"] > 2]
        if moving.empty:
            continue
        fig_hist.add_trace(go.Histogram(
            x=moving["Speed (mph)"],
            name=leg_label,
            nbinsx=40,
            marker_color=colors_transit[leg_label],
            opacity=0.6,
        ))
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="Speed (mph)",
        yaxis_title="GPS samples",
        height=320,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Acceleration & jerk ───────────────────────────────────────────────────
    st.subheader("Acceleration & Jerk (Transit)")
    tr_with_derivs = compute_derivatives(df_transit, smooth_window)
    tr_moving = tr_with_derivs[tr_with_derivs["Speed (mph)"] > 2]

    col_a, col_b = st.columns(2)
    with col_a:
        fig_acc_box = go.Figure()
        for leg_label in ["To Exercise", "From Exercise"]:
            sub = tr_moving[tr_moving["Leg"] == leg_label]
            if sub.empty:
                continue
            fig_acc_box.add_trace(go.Box(
                y=sub["accel_ft_s2"],
                name=leg_label,
                marker_color=colors_transit[leg_label],
                boxmean="sd",
            ))
        fig_acc_box.update_layout(
            title="Acceleration (ft/s²)",
            yaxis_title="ft/s²",
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_acc_box, use_container_width=True)

    with col_b:
        fig_jrk_box = go.Figure()
        for leg_label in ["To Exercise", "From Exercise"]:
            sub = tr_moving[tr_moving["Leg"] == leg_label]
            if sub.empty:
                continue
            fig_jrk_box.add_trace(go.Box(
                y=sub["jerk_ft_s3"],
                name=leg_label,
                marker_color=colors_transit[leg_label],
                boxmean="sd",
            ))
        fig_jrk_box.update_layout(
            title="Jerk (ft/s³)",
            yaxis_title="ft/s³",
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_jrk_box, use_container_width=True)

    # Accel/jerk over time
    fig_tr_deriv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=["Acceleration (ft/s²)", "Jerk (ft/s³)"],
                                  vertical_spacing=0.1)
    for leg_label, leg_df in [("To Exercise", df_outbound), ("From Exercise", df_inbound)]:
        if leg_df.empty:
            continue
        sub = compute_derivatives(leg_df, smooth_window)
        d0 = sub["Distance (mi)"].iloc[0]
        x = sub["Distance (mi)"] - d0
        fig_tr_deriv.add_trace(go.Scatter(
            x=x, y=sub["accel_ft_s2"], mode="lines",
            name=leg_label, line=dict(color=colors_transit[leg_label], width=1.5),
            showlegend=True,
        ), row=1, col=1)
        fig_tr_deriv.add_trace(go.Scatter(
            x=x, y=sub["jerk_ft_s3"], mode="lines",
            name=leg_label, line=dict(color=colors_transit[leg_label], width=1.5),
            showlegend=False,
        ), row=2, col=1)
    fig_tr_deriv.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
    fig_tr_deriv.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    fig_tr_deriv.update_xaxes(title_text="Distance from leg start (mi)", row=2, col=1)
    fig_tr_deriv.update_yaxes(title_text="ft/s²", row=1, col=1)
    fig_tr_deriv.update_yaxes(title_text="ft/s³", row=2, col=1)
    fig_tr_deriv.update_layout(height=500, legend=dict(orientation="h"))
    st.plotly_chart(fig_tr_deriv, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Idealized Run
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ideal:
    if not runs:
        st.warning("No exercise runs found. Try adjusting the settings in the sidebar.")
        st.stop()

    run_labels_ideal = [
        f"Run {i+1}: {r['direction']}  ~{r['target_speed']} mph  "
        f"[{run_start_type(r)} start]  ({r['start_elapsed']}–{r['end_elapsed']} s)"
        for i, r in enumerate(runs)
    ]
    sel_idx_i = st.selectbox(
        "Select run to compare against ideal",
        range(len(runs)),
        format_func=lambda i: run_labels_ideal[i],
        key="ideal_run_sel",
    )
    ri = runs[sel_idx_i]

    # ── Set up selected run ────────────────────────────────────────────────────
    if "_grp_with_derivs" in ri:
        grp_i = ri["_grp_with_derivs"].copy()
    else:
        grp_i = compute_derivatives(ri["data"].copy(), smooth_window)

    going_west_i = ri["direction"] == "West"
    entry_lon_i  = trap_east if going_west_i else trap_west
    exit_lon_i   = trap_west if going_west_i else trap_east
    entry_d_i    = find_crossing_dist(grp_i, entry_lon_i, going_west_i)
    exit_d_i     = find_crossing_dist(grp_i, exit_lon_i,  going_west_i)

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

    # ── Selected run aligned at trap entry ────────────────────────────────────
    if entry_d_i is not None:
        d_aligned_i = (grp_i["Distance (mi)"] - entry_d_i).to_numpy()
    else:
        d_aligned_i = (grp_i["Distance (mi)"] - grp_i["Distance (mi)"].iloc[0]).to_numpy()

    v_actual     = grp_i["speed_smooth"].to_numpy()
    accel_actual = grp_i["accel_ft_s2"].to_numpy()
    jerk_actual  = grp_i["jerk_ft_s3"].to_numpy()

    GRID_N = 600
    d_run_min = float(d_aligned_i.min())
    d_run_max = float(d_aligned_i.max())

    if is_flying:
        # ── Flying start: ideal = constant target speed, zero accel/jerk ─────
        col_info1, col_info2 = st.columns(2)
        col_info1.info(f"Detected start type: **{start_type}** "
                       f"(speed at trap entry ≈ {speed_at_entry:.1f} mph)")
        col_info2.info(
            f"Flying start — ideal is constant **{ri['target_speed']} mph** "
            f"with zero acceleration and jerk."
        )

        d_grid      = np.linspace(d_run_min, d_run_max, GRID_N)
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
        col_info1.info(f"Detected start type: **{start_type}** "
                       f"(speed at trap entry ≈ {speed_at_entry:.1f} mph)")
        col_info2.info(
            f"Acceleration model built from {len(all_standing)} standing-start run(s): "
            f"{peer_labels_str}"
        )

        # Collect (v_ratio = v/v_target, accel_ft_s2) from the acceleration
        # phase of every standing-start run (trap entry → 95% of target speed)
        model_pts = []   # list of (v_ratio, accel_ft_s2)
        for r_p in all_standing:
            grp_p = r_p.get("_grp_with_derivs")
            if grp_p is None:
                grp_p = compute_derivatives(r_p["data"].copy(), smooth_window)
            gw_p        = r_p["direction"] == "West"
            entry_lon_p = trap_east if gw_p else trap_west
            entry_d_p   = find_crossing_dist(grp_p, entry_lon_p, gw_p)
            if entry_d_p is None:
                continue
            d_aln_p = (grp_p["Distance (mi)"] - entry_d_p).to_numpy()
            v_col   = "speed_smooth" if "speed_smooth" in grp_p.columns else "Speed (mph)"
            v_p     = grp_p[v_col].to_numpy()
            a_p     = grp_p["accel_ft_s2"].to_numpy() if "accel_ft_s2" in grp_p.columns else np.zeros(len(grp_p))
            vt_p    = float(r_p["target_speed"])
            # Acceleration phase: from trap entry until speed reaches 95% of target
            phase   = (d_aln_p >= 0) & (v_p < 0.95 * vt_p)
            if not phase.any():
                continue
            vr = v_p[phase] / vt_p
            ac = a_p[phase]
            valid = np.isfinite(vr) & np.isfinite(ac) & (ac > 0) & (vr >= 0) & (vr <= 1)
            if valid.sum() < 3:
                continue
            model_pts.append(np.column_stack([vr[valid], ac[valid]]))

        if not model_pts:
            st.warning("Not enough standing-start acceleration data to build model.")
            st.stop()

        all_pts = np.vstack(model_pts)   # shape (N, 2): [v_ratio, accel_ft_s2]

        # Bin by v_ratio, compute median accel per bin → a_model(v_ratio)
        N_BINS    = 20
        bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
        bin_ctrs  = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_ids   = np.clip(np.digitize(all_pts[:, 0], bin_edges[:-1]) - 1, 0, N_BINS - 1)
        med_accel = np.array([
            float(np.median(all_pts[bin_ids == b, 1])) if (bin_ids == b).any() else 0.0
            for b in range(N_BINS)
        ])
        med_accel = np.maximum(smooth_series(med_accel, 3), 0.0)   # smooth + non-negative

        def a_model(vr_val: float) -> float:
            return float(np.interp(min(vr_val, 0.9999), bin_ctrs, med_accel))

        # ── Integrate ideal profile from rest at trap entry ───────────────────
        DT        = 0.1   # seconds
        target_f  = float(ri["target_speed"])
        v_int     = 0.0   # mph
        t_pts_int, v_pts_int, d_pts_int = [0.0], [0.0], [0.0]
        for _ in range(int(600 / DT)):
            vr = v_int / target_f if target_f > 0 else 1.0
            if vr >= 0.99:
                break
            v_int += a_model(vr) * DT / MPH_S_TO_FT_S2          # ft/s² → mph
            t_pts_int.append(t_pts_int[-1] + DT)
            v_pts_int.append(min(v_int, target_f))
            d_pts_int.append(d_pts_int[-1] + v_int * DT / 3600)  # miles

        # Extend at constant target speed to cover the full trap section
        trap_exit_d_ext = (exit_d_i - entry_d_i) if (entry_d_i is not None and exit_d_i is not None) else d_pts_int[-1] * 1.1
        if d_pts_int[-1] < trap_exit_d_ext:
            n_ext = 50
            d_ext = np.linspace(d_pts_int[-1], trap_exit_d_ext, n_ext + 1)[1:]
            dt_ext = (d_ext / target_f * 3600)   # cumulative time for each extra point
            t_ext  = t_pts_int[-1] + (dt_ext - dt_ext[0] + (d_ext[0] - d_pts_int[-1]) / target_f * 3600)
            t_pts_int.extend(t_ext.tolist())
            v_pts_int.extend([target_f] * n_ext)
            d_pts_int.extend(d_ext.tolist())

        d_grid         = np.array(d_pts_int)
        v_ideal_smooth = np.array(v_pts_int)
        t_grid         = np.array(t_pts_int)

        # Accel from model; zero once target reached
        accel_ideal = smooth_series(np.array([
            a_model(v / target_f) if v < 0.99 * target_f else 0.0
            for v in v_ideal_smooth
        ]), smooth_window)

        # Jerk = d(accel)/dt along the ideal time axis
        jerk_ideal = smooth_series(np.gradient(accel_ideal, t_grid), smooth_window)

        v_band_lo = None
        v_band_hi = None

        residual_label = "Speed Residual — Actual minus Acceleration Model (mph)"
        v_ideal_at_actual = np.interp(d_aligned_i, d_grid, v_ideal_smooth,
                                       left=np.nan, right=np.nan)
        speed_residual = v_actual - v_ideal_at_actual

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

    # x values for plotting: the trap-section distance points in original order
    d_trap_plot = d_aligned_i[trap_mask]
    cum_time_err_plot = np.interp(d_trap_plot, d_sorted, cum_time_err)

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
        x=d_trap_plot, y=cum_time_err_plot,
        name="Cumulative time error", mode="lines",
        line=dict(color="crimson", width=2),
        fill="tozeroy",
        fillcolor="rgba(220,20,60,0.12)",
    ), row=1, col=1)
    fig_ideal.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    # Row 2: Speed
    fig_ideal.add_trace(go.Scatter(
        x=d_aligned_i, y=grp_i["Speed (mph)"],
        name="Actual (raw)", mode="lines",
        line=dict(color="lightsteelblue", width=1), opacity=0.5,
    ), row=2, col=1)
    # Peer speed band (standing start only)
    if v_band_lo is not None and v_band_hi is not None:
        fig_ideal.add_trace(go.Scatter(
            x=np.concatenate([d_grid, d_grid[::-1]]),
            y=np.concatenate([v_band_hi, v_band_lo[::-1]]),
            fill="toself",
            fillcolor="rgba(255,165,0,0.15)",
            line=dict(width=0),
            name="Peer range",
            showlegend=True,
        ), row=2, col=1)
    ideal_name = "Target speed" if is_flying else "Ideal (median peers)"
    fig_ideal.add_trace(go.Scatter(
        x=d_grid, y=v_ideal_smooth,
        name=ideal_name, mode="lines",
        line=dict(color="orange", width=2.5, dash="dash"),
    ), row=2, col=1)
    fig_ideal.add_trace(go.Scatter(
        x=d_aligned_i, y=v_actual,
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
        x=d_grid, y=accel_ideal,
        name="Ideal accel", mode="lines",
        line=dict(color="darkorange", width=2, dash="dash"),
    ), row=3, col=1)
    fig_ideal.add_trace(go.Scatter(
        x=d_aligned_i, y=accel_actual,
        name="Actual accel", mode="lines",
        line=dict(color="#e76f51", width=2),
    ), row=3, col=1)
    fig_ideal.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)

    # Row 4: Jerk
    fig_ideal.add_trace(go.Scatter(
        x=d_grid, y=jerk_ideal,
        name="Ideal jerk", mode="lines",
        line=dict(color="darkcyan", width=2, dash="dash"),
    ), row=4, col=1)
    fig_ideal.add_trace(go.Scatter(
        x=d_aligned_i, y=jerk_actual,
        name="Actual jerk", mode="lines",
        line=dict(color="#2a9d8f", width=2),
    ), row=4, col=1)
    fig_ideal.add_hline(y=0, line_dash="dot", line_color="gray", row=4, col=1)

    # Row 5: Speed residual
    fig_ideal.add_trace(go.Scatter(
        x=d_aligned_i, y=speed_residual,
        name="Speed residual", mode="lines",
        line=dict(color="mediumpurple", width=2),
        fill="tozeroy",
        fillcolor="rgba(147,112,219,0.15)",
    ), row=5, col=1)
    fig_ideal.add_hline(y=0, line_dash="dot", line_color="gray", row=5, col=1)

    # Trap markers on all rows
    if entry_d_i is not None:
        fig_ideal.add_vline(x=0, line_dash="dash", line_color="green",
                            annotation_text="Trap entry",
                            annotation_position="top right")
    if exit_d_i is not None:
        exit_x_i = exit_d_i - entry_d_i
        fig_ideal.add_vline(x=exit_x_i, line_dash="dash", line_color="red",
                            annotation_text="Trap exit",
                            annotation_position="top left")
        if entry_d_i is not None:
            fig_ideal.add_vrect(x0=0, x1=exit_x_i,
                                fillcolor="rgba(0,200,0,0.06)", line_width=0)

    fig_ideal.update_xaxes(
        title_text="Distance from trap entry (mi)  [negative = run-in]", row=5, col=1
    )
    if len(cum_time_err_plot) > 0:
        _cte_lo = float(np.nanmin(cum_time_err_plot))
        _cte_hi = float(np.nanmax(cum_time_err_plot))
        _cte_pad = max(abs(_cte_hi - _cte_lo) * 0.15, 0.5)
        _cte_range = [_cte_lo - _cte_pad, _cte_hi + _cte_pad]
    else:
        _cte_range = [-1, 1]
    fig_ideal.update_yaxes(title_text="sec", range=_cte_range, row=1, col=1)
    def _sym_range(arr, pad=0.15, min_half=0.5, zero=False):
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

    def _pct_range(arr, pct=2, pad=0.15, min_half=1.0, zero=True):
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