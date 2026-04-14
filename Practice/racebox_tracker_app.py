"""
Practice Run Analysis — Acceleration & Jerk
Supports RaceBox GPS logger files and speed-tracker parquet files.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

import telemetry
from telemetry import (
    TrapConfig,
    MPH_S_TO_FT_S2, G_FT_S2,
    smooth_series, compute_derivatives,
    segment_exercise_runs,
    _sym_range, _pct_range,
    _grp_derivs, _min_speed_near,
    run_start_type, run_finish_type, get_run_timing_refs,
)

st.set_page_config(page_title="Practice Run Analysis", layout="wide", page_icon="🏎")

# Reduce default top padding
st.markdown(
    """<style>
    [data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 1rem !important; }
    </style>""",
    unsafe_allow_html=True,
)

DATALOGS = Path(__file__).parent




# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_track(path: str) -> pd.DataFrame:
    """Load a parquet file and normalise to the standard internal schema.

    Files whose name starts with 'RaceBox' are loaded via telemetry.load_racebox
    (raw ISO-8601 timestamps, haversine distance computation).  All other files
    are treated as pre-processed speed-tracker output and loaded via
    telemetry.load_speed_tracker.
    """
    if Path(path).name.startswith("RaceBox"):
        return telemetry.load_racebox(path)
    return telemetry.load_speed_tracker(path)


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Practice Run Analysis")
st.caption("Analyze GPS logs from RaceBox or speed-tracker files to evaluate acceleration, jerk, and run quality.")

rb_files = sorted(DATALOGS.glob("RaceBox*.parquet"), reverse=True)
st_files = sorted(DATALOGS.glob("*speed_tracker*.parquet"), reverse=True)
track_files = sorted(rb_files + st_files, key=lambda p: p.name, reverse=True)
if not track_files:
    st.error(f"No parquet files found in {DATALOGS}")
    st.stop()

# ── Settings defaults via session state (Settings tab widgets write here) ─────
for _k, _v in [("cfg_lat", 29.34), ("cfg_smooth", 9),
               ("cfg_min_speed", 0.2), ("cfg_min_rows", 40)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

lat_threshold = float(st.session_state.cfg_lat)
smooth_window  = int(st.session_state.cfg_smooth)
min_speed      = float(st.session_state.cfg_min_speed)
min_rows       = int(st.session_state.cfg_min_rows)
flying_window_mi  = float(st.session_state.get("cfg_flying_window_ft", 100)) / 5280.0

# Reserve a sidebar slot at the top for run filters (populated after runs are computed)
_filter_container = st.sidebar.container()

with st.sidebar:
    st.markdown("**Log files**")
    selected_files = [
        f for f in track_files
        if st.checkbox(f.name, value=(f == track_files[0]), key=f"file__{f.name}")
    ]
    if not selected_files:
        st.warning("Select at least one log file.")
        st.stop()


# ── Load data early so we can estimate trap locations before rendering sliders ─
_dfs_per_file = [
    load_track(str(_tf)).dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    for _tf in selected_files
]
df_raw = pd.concat(_dfs_per_file, ignore_index=True)

# ── Default trap positions — edit to match your course ────────────────────────
_EAST_TRAP_LAT =  29.337290   # ← edit me
_EAST_TRAP_LON = -95.63952    # ← edit me
_WEST_TRAP_LAT =  29.33716    # ← edit me
_WEST_TRAP_LON = -95.65612    # ← edit me
# ──────────────────────────────────────────────────────────────────────────────

# Reset session state when file selection changes.
_src_key = str(sorted(f.name for f in selected_files))
if st.session_state.get("_trap_src_file") != _src_key:
    st.session_state["cfg_east_ctr_lat"] = _EAST_TRAP_LAT
    st.session_state["cfg_east_ctr_lon"] = _EAST_TRAP_LON
    st.session_state["cfg_west_ctr_lat"] = _WEST_TRAP_LAT
    st.session_state["cfg_west_ctr_lon"] = _WEST_TRAP_LON
    st.session_state["_trap_src_file"] = _src_key
    # Reset run filters so stale values don't hide runs from the new selection
    for _fk in ("flt_target", "flt_start", "flt_finish"):
        st.session_state.pop(_fk, None)

trap_east_lat = float(st.session_state.get("cfg_east_ctr_lat", _EAST_TRAP_LAT))
trap_east     = float(st.session_state.get("cfg_east_ctr_lon", _EAST_TRAP_LON))
trap_west_lat = float(st.session_state.get("cfg_west_ctr_lat", _WEST_TRAP_LAT))
trap_west     = float(st.session_state.get("cfg_west_ctr_lon", _WEST_TRAP_LON))

# ── Trap geometry and run segmentation ────────────────────────────────────────
# TrapConfig derives all Cartesian geometry from the four endpoint coordinates.

trap = TrapConfig(
    east_lon=trap_east, west_lon=trap_west,
    east_lat=trap_east_lat, west_lat=trap_west_lat,
    flying_window_mi=flying_window_mi,
    min_speed=min_speed,
)
TRAP_DIST_MI = trap.dist_mi   # kept as a local alias for readability in tabs

runs = []
for _df_f in _dfs_per_file:
    runs.extend(segment_exercise_runs(_df_f, lat_threshold, min_speed, min_rows, trap=trap))

# ── Shared map helpers ────────────────────────────────────────────────────────


def _esri_layer() -> list[dict]:
    """ESRI World Imagery raster tile layer for Plotly Scattermap."""
    return [{"sourcetype": "raster",
             "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/"
                        "World_Imagery/MapServer/tile/{z}/{y}/{x}"],
             "type": "raster", "below": "traces"}]


def _map_layout(center_lat: float, center_lon: float,
                satellite: bool, height: int = 580) -> dict:
    """Standard Scattermap layout dict (satellite or street basemap)."""
    return dict(
        map=dict(
            style="white-bg" if satellite else "carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=13,
            layers=_esri_layer() if satellite else [],
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )


def _add_trap_lines(fig: go.Figure) -> None:
    """Add East/West trap gate lines and measured-section centre line to a Scattermap."""
    for name, lon in [("East trap", trap_east), ("West trap", trap_west)]:
        fig.add_trace(go.Scattermap(
            lat=[trap_lat_lo, trap_lat_hi], lon=[lon, lon],
            mode="lines", line=dict(color="dodgerblue", width=3),
            name=name, showlegend=False,
            hovertemplate=f"{name}<br>lon={lon:.5f}<extra></extra>",
        ))
    fig.add_trace(go.Scattermap(
        lat=[trap_lat_mid, trap_lat_mid], lon=[trap_east, trap_west],
        mode="lines", line=dict(color="yellow", width=2),
        name="Measured section", showlegend=False,
        hovertemplate="Measured section<extra></extra>",
    ))


# ── Tabs ──────────────────────────────────────────────────────────────────────



tab_overview, tab_detail, tab_accel, tab_compare, tab_settings, tab_map = st.tabs(
    ["Runs Overview", "Run Detail", "Accel/Decel", "GPS vs Accel", "Settings", "Track Map"]
)

# ── Module-scope map state (used across multiple tabs) ────────────────────────

_window_deg_lon = trap.flying_window_mi / trap.mi_per_deg_lon
_run_zone_mask = (
    (df_raw["Latitude"] < lat_threshold) &
    (df_raw["Longitude"] >= min(trap_east, trap_west) - _window_deg_lon) &
    (df_raw["Longitude"] <= max(trap_east, trap_west) + _window_deg_lon)
)
df_raw["Zone"] = np.where(_run_zone_mask, "Exercise Area", "Transit")

df_plot = df_raw.iloc[::5].copy()  # downsample for render speed

# ── Run filters (sidebar — fills the top-of-sidebar container) ────────────────
with _filter_container:
    _all_targets = sorted({r["target_speed"] for r in runs}) if runs else []
    sel_target = st.selectbox(
        "Target speed",
        ["All"] + _all_targets,
        format_func=lambda x: "All" if x == "All" else f"{x} mph",
        key="flt_target",
    )
    _start_types = sorted({run_start_type(r, trap, smooth_window) for r in runs}) if runs else []
    sel_start = st.selectbox("Start type", ["All"] + _start_types, key="flt_start")
    _finish_types = sorted({run_finish_type(r, trap, smooth_window) for r in runs}) if runs else []
    sel_finish = st.selectbox("Finish type", ["All"] + _finish_types, key="flt_finish")
    st.divider()

# Apply run filters
if sel_target != "All":
    runs = [r for r in runs if r["target_speed"] == sel_target]
if sel_start != "All":
    runs = [r for r in runs if run_start_type(r, trap, smooth_window) == sel_start]
if sel_finish != "All":
    runs = [r for r in runs if run_finish_type(r, trap, smooth_window) == sel_finish]

# Trap gate lat bounds (used by _add_trap_lines and all maps)
_ex_rows = df_raw[
    (df_raw["Latitude"] < lat_threshold) &
    (df_raw["Longitude"] >= trap_west) &
    (df_raw["Longitude"] <= trap_east)
]
if not _ex_rows.empty:
    trap_lat_mid = float(_ex_rows["Latitude"].median())
    trap_lat_lo  = float(_ex_rows["Latitude"].quantile(0.05))
    trap_lat_hi  = float(_ex_rows["Latitude"].quantile(0.95))
else:
    trap_lat_mid, trap_lat_lo, trap_lat_hi = 29.337, 29.334, 29.340

_map_center_lat = (trap_east_lat + trap_west_lat) / 2
_map_center_lon = (trap_east + trap_west) / 2

_RUN_PALETTE = [
    "#e63946", "#f4a261", "#2a9d8f", "#e9c46a", "#a8dadc",
    "#ff6b6b", "#ffd166", "#06d6a0", "#118ab2", "#fb5607",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Track map
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    col_map, col_stats = st.columns([4, 1])

    with col_map:
        use_satellite = st.radio(
            "Map style", ["Street (carto)", "Satellite (ESRI)"],
            horizontal=True, label_visibility="collapsed"
        ) == "Satellite (ESRI)"

        _fig_map = go.Figure()
        _fig_map.add_trace(go.Scattermap(
            lat=df_plot.loc[df_plot["Zone"] == "Transit", "Latitude"],
            lon=df_plot.loc[df_plot["Zone"] == "Transit", "Longitude"],
            mode="lines", line=dict(color="#457b9d", width=2), name="Transit",
            hovertemplate="Transit<br>lat=%{lat:.5f}<br>lon=%{lon:.5f}<extra></extra>",
        ))
        _fig_map.add_trace(go.Scattermap(
            lat=df_plot.loc[df_plot["Zone"] == "Exercise Area", "Latitude"],
            lon=df_plot.loc[df_plot["Zone"] == "Exercise Area", "Longitude"],
            mode="lines", line=dict(color="#aaaaaa", width=1), name="Exercise area",
            hovertemplate="Exercise area<br>lat=%{lat:.5f}<br>lon=%{lon:.5f}<extra></extra>",
        ))
        for _ri, _r in enumerate(runs):
            _rdf = _r["data"].iloc[::5]  # downsample for render speed
            _col = _RUN_PALETTE[_ri % len(_RUN_PALETTE)]
            _lbl = f"R{_ri+1} {_r['direction']} {_r['target_speed']}mph"
            _fig_map.add_trace(go.Scattermap(
                lat=_rdf["Latitude"], lon=_rdf["Longitude"],
                mode="lines", line=dict(color=_col, width=3), name=_lbl,
                hovertemplate=f"{_lbl}<br>lat=%{{lat:.5f}}<br>lon=%{{lon:.5f}}<extra></extra>",
            ))
        _add_trap_lines(_fig_map)
        _fig_map.update_layout(**_map_layout(_map_center_lat, _map_center_lon, use_satellite))
        st.plotly_chart(_fig_map, use_container_width=True)

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
    _ov_standing = [r for r in runs if run_start_type(r, trap, smooth_window) == "Standing"]
    _ov_accel_profs = []
    for _r_p in _ov_standing:
        _grp_p  = _r_p["_grp_with_derivs"]
        _en_p, _ = get_run_timing_refs(_r_p, trap, smooth_window)
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

        # Timing references: see get_run_timing_refs for anchor logic per start/finish type.
        en_d, ex_d = get_run_timing_refs(r, trap, smooth_window)

        if en_d is not None and ex_d is not None:
            d_lo, d_hi = min(en_d, ex_d), max(en_d, ex_d)
            section = grp[
                (grp["Distance (mi)"] >= d_lo) & (grp["Distance (mi)"] <= d_hi)
            ]
        else:
            section = grp

        section    = section[section["Speed (mph)"] >= min_speed]
        if section.empty:
            section = grp  # fallback: use full run if measured section has no moving data
        spd_err    = section["Speed (mph)"] - r["target_speed"]
        measured_dist = (d_hi - d_lo) if (en_d is not None and ex_d is not None) else r["distance_mi"]

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

                if run_start_type(r, trap, smooth_window) == "Flying" or _OV_DGRID is None:
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
            "Start": run_start_type(r, trap, smooth_window),
            "Finish": run_finish_type(r, trap, smooth_window),
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
        f"Exit distance uses geometric fallback when GPS data ends before exit trap. "
        f"* = trap entry not detected, aligned to run start."
    )
    fig_trap = go.Figure()
    for i, r in enumerate(runs):
        grp = r["data"].copy()
        entry_dist, _ = get_run_timing_refs(r, trap, smooth_window)
        aligned = entry_dist is not None
        if entry_dist is None:
            entry_dist = grp["Distance (mi)"].iloc[0]

        d_aligned = grp["Distance (mi)"] - entry_dist
        fig_trap.add_trace(go.Scatter(
            x=d_aligned,
            y=grp["Speed (mph)"],
            mode="lines",
            name=f"R{i+1} {r['direction']} ~{r['target_speed']} mph" + ("" if aligned else " *"),
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
        f"[{run_start_type(r, trap, smooth_window)} start | {run_finish_type(r, trap, smooth_window)} finish]  "
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

    # ── Timing references for this run ────────────────────────────────────────
    # Standing → motion-start distance + GPS 1-mile mark.
    # Flying   → GPS trap-crossing geometry with TRAP_DIST_MI fallback.
    entry_d_raw, exit_d_raw = get_run_timing_refs(r, trap, smooth_window)



with tab_detail:
    # ── Idealized Run comparison ──────────────────────────────────────────────
    st.divider()
    st.subheader("Idealized Run Comparison")

    # Detect flying vs standing start/finish using min speed in window around each trap
    _center_en = entry_d_raw if entry_d_raw is not None else grp["Distance (mi)"].iloc[0]
    _center_ex = exit_d_raw  if exit_d_raw  is not None else grp["Distance (mi)"].iloc[-1]
    speed_at_entry = _min_speed_near(grp, _center_en, "speed_smooth", side="before", flying_window_mi=flying_window_mi)
    speed_at_exit  = _min_speed_near(grp, _center_ex, "speed_smooth", side="after",  flying_window_mi=flying_window_mi)
    is_flying        = speed_at_entry >= r["target_speed"] - 5
    is_flying_finish = speed_at_exit  >= r["target_speed"] - 5
    start_type  = "Flying start"  if is_flying        else "Standstill start"
    finish_type = "Flying finish" if is_flying_finish else "Standstill finish"

    # ── Selected run aligned at trap entry ────────────────────────────────────
    _d_arr = grp["Distance (mi)"].to_numpy()
    _t_arr = grp["Elapsed time (sec)"].to_numpy()

    if entry_d_raw is not None:
        d_aligned_i = (_d_arr - entry_d_raw)
        t_at_entry  = float(np.interp(entry_d_raw, _d_arr, _t_arr))
    else:
        d_aligned_i = (_d_arr - _d_arr[0])
        t_at_entry  = float(_t_arr[0])
    t_aligned_i = _t_arr - t_at_entry   # seconds from trap entry

    v_actual     = grp["speed_smooth"].to_numpy()
    accel_actual = grp["accel_ft_s2"].to_numpy()
    jerk_actual  = grp["jerk_ft_s3"].to_numpy()

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
            f"Flying start — ideal is constant **{r['target_speed']} mph** through the "
            f"trap" + (" then decelerates to rest." if not is_flying_finish else " with zero acceleration and jerk.")
        )

        d_grid         = np.linspace(d_run_min, d_run_max, GRID_N)
        t_grid         = d_grid / float(r["target_speed"]) * 3600
        v_ideal_smooth = np.full(GRID_N, float(r["target_speed"]))
        accel_ideal    = np.zeros(GRID_N)
        jerk_ideal     = np.zeros(GRID_N)
        residual_label = f"Speed Residual — Actual minus Target ({r['target_speed']} mph)"
        speed_residual = v_actual - float(r["target_speed"])

    else:
        # ── Standing start: acceleration model from ALL standing-start runs ───
        all_standing = [r for r in runs if run_start_type(r, trap, smooth_window) == "Standing"]

        col_info1, _ = st.columns(2)
        col_info1.info(
            f"Start: **{start_type}** (speed at entry ≈ {speed_at_entry:.1f} mph)  |  "
            f"Finish: **{finish_type}** (speed at exit ≈ {speed_at_exit:.1f} mph)"
        )

        # ── Acceleration model: direct speed-profile median ───────────────────
        # Collect normalised speed-vs-distance profiles from every standing-start
        # peer run, aligned at trap entry (d=0).  Taking the pointwise median
        # avoids ODE integration and the noise that comes from differentiating
        # GPS speed data twice to get acceleration.
        target_f       = float(r["target_speed"])
        trap_exit_d_ext = (exit_d_raw - entry_d_raw) \
                          if (entry_d_raw is not None and exit_d_raw is not None) else 0.25

        accel_profs = []   # list of (d_aligned, v_normalised) arrays
        for r_p in all_standing:
            grp_p = r_p.get("_grp_with_derivs")
            if grp_p is None:
                grp_p = compute_derivatives(r_p["data"].copy(), smooth_window)
            entry_d_p, _ = get_run_timing_refs(r_p, trap, smooth_window)
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
        all_standing_finish = [r for r in runs if run_finish_type(r, trap, smooth_window) == "Standing"]
        decel_profs = []
        for r_p in all_standing_finish:
            grp_p = r_p.get("_grp_with_derivs")
            if grp_p is None:
                grp_p = compute_derivatives(r_p["data"].copy(), smooth_window)
            vt_p = float(r_p["target_speed"])

            entry_d_p, exit_d_p = get_run_timing_refs(r_p, trap, smooth_window)
            if exit_d_p is None:
                continue

            d_p   = grp_p["Distance (mi)"].to_numpy()
            v_col = "speed_smooth" if "speed_smooth" in grp_p.columns else "Speed (mph)"
            v_p   = grp_p[v_col].to_numpy()

            d_aln_exit = d_p - exit_d_p
            # exit_d is the stop point (speed ≈ 0 at d_aln=0); capture the
            # braking zone leading up to the stop (up to 0.3 mi before it).
            phase = (d_aln_exit >= -0.3) & (d_aln_exit <= 0.02)
            if phase.sum() < 3:
                continue
            d_sl  = d_aln_exit[phase]
            v_nrm = np.clip(v_p[phase] / vt_p, 0.0, 1.1)
            s_idx = np.argsort(d_sl)
            d_sl, v_nrm = d_sl[s_idx], v_nrm[s_idx]

            # Skip if the car never reached target speed during this segment
            # (partial run or mis-classified).
            if np.max(v_nrm) < 0.8:
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
            _target_f     = float(r["target_speed"])
            v_decel_ideal = np.clip(_v_nrm_dec * _target_f, 0.0, _target_f)

            # Align decel distance to trap-entry coordinate system
            _d0_dec       = (exit_d_raw - entry_d_raw) \
                            if (exit_d_raw is not None and entry_d_raw is not None) \
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
    trap_exit_d = (exit_d_raw - entry_d_raw) if (entry_d_raw is not None and exit_d_raw is not None) else None
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
    if exit_d_raw is not None and entry_d_raw is not None:
        _exit_d_aln = exit_d_raw - entry_d_raw
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
        x=x_actual, y=grp["Speed (mph)"],
        name="Actual (raw)", mode="lines",
        line=dict(color="lightsteelblue", width=1), opacity=0.5,
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
        y=r["target_speed"], line_dash="dot", line_color="gray",
        annotation_text=f"Target {r['target_speed']} mph",
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
        grp["Speed (mph)"].to_numpy()[trap_mask],
        v_actual[trap_mask],
        v_ideal_smooth[_grid_trap],
        np.array([float(r["target_speed"])]),
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
    if entry_d_raw is not None and exit_d_raw is not None:
        exit_d_aln = exit_d_raw - entry_d_raw
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
             round(float(np.sqrt(np.nanmean((v_actual_ms - r["target_speed"])**2))), 3),
             round(float(np.sqrt(np.nanmean((v_ideal_ms  - r["target_speed"])**2))), 3)),
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
                     "Latitude", "Longitude"]
        st.dataframe(grp[[c for c in show_cols if c in grp.columns]].round(4),
                     use_container_width=True)

    # ── Run map ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Run Map")
    _det_sat = st.radio(
        "Map style", ["Street (carto)", "Satellite (ESRI)"],
        horizontal=True, label_visibility="collapsed", key="cfg_det_map_style",
    ) == "Satellite (ESRI)"

    _fig_det = go.Figure()
    _fig_det.add_trace(go.Scattermap(
        lat=grp["Latitude"], lon=grp["Longitude"],
        mode="lines+markers",
        line=dict(color="#e63946", width=2),
        marker=dict(size=4, color="#e63946"),
        name=f"R{sel_idx+1} {r['direction']}",
        hovertemplate="lat=%{lat:.5f}<br>lon=%{lon:.5f}<extra></extra>",
    ))
    _add_trap_lines(_fig_det)
    _fig_det.update_layout(**_map_layout(
        float(grp["Latitude"].mean()), float(grp["Longitude"].mean()), _det_sat, height=500
    ))
    st.plotly_chart(_fig_det, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Accel / Decel Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab_accel:
    if not runs:
        st.info("No exercise runs found.")
    else:
        _colors = px.colors.qualitative.Plotly

        def _run_short_label(r_p: dict) -> str:
            idx = runs.index(r_p)
            return f"R{idx+1} {r_p['direction']} {r_p['start_time']} (~{r_p['target_speed']} mph)"

        # ── Acceleration ─────────────────────────────────────────────────────
        all_ss = [r for r in runs if run_start_type(r, trap, smooth_window) == "Standing"]
        all_sf = [r for r in runs if run_finish_type(r, trap, smooth_window) == "Standing"]

        st.subheader(f"Acceleration — Standing Starts ({len(all_ss)} run{'s' if len(all_ss) != 1 else ''})")

        if not all_ss:
            st.info("No standing-start runs found.")
        else:
            fig_acc = make_subplots(
                rows=3, cols=2, shared_xaxes="columns",
                column_titles=["vs Distance from Entry", "vs Speed (shift consistency)"],
                row_titles=["Speed (mph)", "Accel (ft/s²)", "Jerk (ft/s³)"],
                vertical_spacing=0.08, horizontal_spacing=0.08,
            )

            for idx, r_p in enumerate(all_ss):
                grp_p = _grp_derivs(r_p, smooth_window)
                en_d, _ = get_run_timing_refs(r_p, trap, smooth_window)
                if en_d is None:
                    continue

                d_p   = grp_p["Distance (mi)"].to_numpy()
                v_p   = grp_p["speed_smooth"].to_numpy()
                a_p   = grp_p["accel_ft_s2"].to_numpy()
                j_p   = grp_p["jerk_ft_s3"].to_numpy()
                d_aln = d_p - en_d  # 0 = trap entry

                # Show from first movement to target speed (+ small margin)
                _tgt_p   = float(r_p["target_speed"])
                mv_idx   = int(np.argmax(v_p >= 1.0)) if (v_p >= 1.0).any() else 0
                tgt_mask = v_p >= _tgt_p * 0.97
                end_idx  = int(np.argmax(tgt_mask)) + 15 if tgt_mask.any() else len(v_p) - 1
                end_idx  = min(end_idx, len(v_p) - 1)
                sl       = slice(mv_idx, end_idx + 1)

                color = _colors[idx % len(_colors)]
                lbl   = _run_short_label(r_p)

                # Left column — vs distance
                fig_acc.add_trace(go.Scatter(
                    x=d_aln[sl], y=v_p[sl], mode="lines",
                    name=lbl, line=dict(color=color), showlegend=True, legendgroup=lbl,
                ), row=1, col=1)
                fig_acc.add_trace(go.Scatter(
                    x=d_aln[sl], y=a_p[sl], mode="lines",
                    name=lbl, line=dict(color=color), showlegend=False, legendgroup=lbl,
                ), row=2, col=1)
                fig_acc.add_trace(go.Scatter(
                    x=d_aln[sl], y=j_p[sl], mode="lines",
                    name=lbl, line=dict(color=color), showlegend=False, legendgroup=lbl,
                ), row=3, col=1)

                # Right column — accel & jerk vs speed (rows 2 & 3, matching left column)
                acc_phase = (v_p[sl] >= 2) & (v_p[sl] <= _tgt_p * 1.02)
                v_ap  = v_p[sl][acc_phase]
                a_ap  = a_p[sl][acc_phase]
                j_ap  = j_p[sl][acc_phase]
                s_ord = np.argsort(v_ap)
                fig_acc.add_trace(go.Scatter(
                    x=v_ap[s_ord], y=a_ap[s_ord], mode="lines",
                    name=lbl, line=dict(color=color), showlegend=False, legendgroup=lbl,
                ), row=2, col=2)
                fig_acc.add_trace(go.Scatter(
                    x=v_ap[s_ord], y=j_ap[s_ord], mode="lines",
                    name=lbl, line=dict(color=color), showlegend=False, legendgroup=lbl,
                ), row=3, col=2)

            # Reference lines — left column
            for _r in (2, 3):
                fig_acc.add_hline(y=0, line_dash="dot", line_color="gray", row=_r, col=1)
            fig_acc.add_vline(x=0, line_dash="dash", line_color="green",
                              annotation_text="Run start", row=1, col=1)
            # Reference lines — right column (rows 2 & 3)
            for _r in (2, 3):
                fig_acc.add_hline(y=0, line_dash="dot", line_color="gray", row=_r, col=2)

            fig_acc.update_xaxes(title_text="Distance from run start (mi)", row=3, col=1)
            fig_acc.update_xaxes(title_text="Speed (mph)", row=2, col=2)
            fig_acc.update_xaxes(title_text="Speed (mph)", row=3, col=2)
            fig_acc.update_layout(
                height=800,
                legend=dict(orientation="h", y=-0.08),
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            # ── Gear shift summary ────────────────────────────────────────────
            st.caption(
                "Gear shifts appear as **dips in the acceleration curve**. "
                "If shifts are consistent, the dips in the right-hand chart will overlap at the same speed. "
                "Braking spikes at the end appear as sharp negative acceleration."
            )

        st.divider()

        # ── Deceleration ─────────────────────────────────────────────────────
        st.subheader(f"Deceleration — Standing Stops ({len(all_sf)} run{'s' if len(all_sf) != 1 else ''})")

        if not all_sf:
            st.info("No standing-finish runs found.")
        else:
            fig_dec = make_subplots(
                rows=3, cols=2, shared_xaxes="columns",
                column_titles=["vs Distance from Exit", "vs Speed (braking consistency)"],
                row_titles=["Speed (mph)", "Accel (ft/s²)", "Jerk (ft/s³)"],
                vertical_spacing=0.08, horizontal_spacing=0.08,
            )

            for idx, r_p in enumerate(all_sf):
                grp_p    = _grp_derivs(r_p, smooth_window)
                _, ex_d  = get_run_timing_refs(r_p, trap, smooth_window)
                if ex_d is None:
                    continue

                d_p   = grp_p["Distance (mi)"].to_numpy()
                v_p   = grp_p["speed_smooth"].to_numpy()
                a_p   = grp_p["accel_ft_s2"].to_numpy()
                j_p   = grp_p["jerk_ft_s3"].to_numpy()
                d_aln = d_p - ex_d   # 0 = stop point, negative = still moving

                # Show the braking zone leading up to the stop
                sl_mask = (d_aln >= -0.3) & (d_aln <= 0.02)
                if sl_mask.sum() < 3:
                    continue

                color = _colors[idx % len(_colors)]
                lbl   = _run_short_label(r_p)

                # Left — vs distance from exit
                fig_dec.add_trace(go.Scatter(
                    x=d_aln[sl_mask], y=v_p[sl_mask], mode="lines",
                    name=lbl, line=dict(color=color), showlegend=True, legendgroup=lbl,
                ), row=1, col=1)
                fig_dec.add_trace(go.Scatter(
                    x=d_aln[sl_mask], y=a_p[sl_mask], mode="lines",
                    name=lbl, line=dict(color=color), showlegend=False, legendgroup=lbl,
                ), row=2, col=1)
                fig_dec.add_trace(go.Scatter(
                    x=d_aln[sl_mask], y=j_p[sl_mask], mode="lines",
                    name=lbl, line=dict(color=color), showlegend=False, legendgroup=lbl,
                ), row=3, col=1)

                # Right — accel & jerk vs speed (rows 2 & 3, matching left column)
                dec_mask = (v_p[sl_mask] >= 1) & (a_p[sl_mask] <= 0)
                v_dp  = v_p[sl_mask][dec_mask]
                a_dp  = a_p[sl_mask][dec_mask]
                j_dp  = j_p[sl_mask][dec_mask]
                s_ord = np.argsort(-v_dp)
                fig_dec.add_trace(go.Scatter(
                    x=v_dp[s_ord], y=a_dp[s_ord], mode="lines",
                    name=lbl, line=dict(color=color), showlegend=False, legendgroup=lbl,
                ), row=2, col=2)
                fig_dec.add_trace(go.Scatter(
                    x=v_dp[s_ord], y=j_dp[s_ord], mode="lines",
                    name=lbl, line=dict(color=color), showlegend=False, legendgroup=lbl,
                ), row=3, col=2)

            for _r in (2, 3):
                fig_dec.add_hline(y=0, line_dash="dot", line_color="gray", row=_r, col=1)
            fig_dec.add_vline(x=0, line_dash="dash", line_color="red",
                              annotation_text="Stop", row=1, col=1)
            for _r in (2, 3):
                fig_dec.add_hline(y=0, line_dash="dot", line_color="gray", row=_r, col=2)

            fig_dec.update_xaxes(title_text="Distance from exit trap (mi)", row=3, col=1)
            fig_dec.update_xaxes(title_text="Speed (mph)", row=2, col=2)
            fig_dec.update_xaxes(title_text="Speed (mph)", row=3, col=2)
            fig_dec.update_layout(
                height=800,
                legend=dict(orientation="h", y=-0.08),
            )
            st.plotly_chart(fig_dec, use_container_width=True)
            st.caption(
                "Braking consistency: the deceleration vs speed curves (right) should overlap "
                "if the driver applies the brakes at the same rate each run. "
                "Braking application point shows as the distance at which speed begins to drop."
            )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab — GPS vs Accelerometer
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("GPS-Derived vs Accelerometer Acceleration")
    _has_gforce = runs and all(
        col in r["data"].columns
        for r in runs
        for col in ["GForceX", "GForceY", "GForceZ"]
    )
    if not runs:
        st.info("No exercise runs found.")
    elif not _has_gforce:
        st.info("No accelerometer (GForce) data in the selected log file(s). This tab requires a RaceBox file with GForceX/Y/Z columns.")
    else:
        # ── Controls ──────────────────────────────────────────────────────────
        _c1, _c2, _c3 = st.columns([2, 1, 2])
        with _c1:
            _cmp_axis = st.selectbox(
                "GForce axis",
                ["Auto (best r)", "X", "Y", "Z", "Horiz √(X²+Y²)"],
                key="cmp_axis",
                help="Which accelerometer axis to compare against GPS-derived acceleration. "
                     "Auto tests X/Y/Z ± sign and picks the highest Pearson r.",
            )
        with _c2:
            _cmp_flip = st.checkbox("Flip sign", value=False, key="cmp_flip",
                                    help="Flip the GForce sign (ignored in Auto mode).")
        with _c3:
            _cmp_run_idx = st.selectbox(
                "Run (overlay chart)",
                range(len(runs)),
                format_func=lambda i: f"R{i+1} {runs[i]['direction']} ~{runs[i]['target_speed']} mph",
                key="cmp_run",
            )

        # ── Helper: extract GForce acceleration for a grp DataFrame ───────────
        def _gf_accel(grp: pd.DataFrame, axis: str, flip: bool) -> np.ndarray:
            if axis == "X":
                raw = grp["GForceX"].to_numpy(dtype=float)
            elif axis == "Y":
                raw = grp["GForceY"].to_numpy(dtype=float)
            elif axis == "Z":
                raw = grp["GForceZ"].to_numpy(dtype=float)
            else:  # horizontal magnitude — always positive, sign from GPS
                raw = np.sqrt(grp["GForceX"].to_numpy(dtype=float) ** 2 +
                              grp["GForceY"].to_numpy(dtype=float) ** 2)
            if flip:
                raw = -raw
            t   = grp["Elapsed time (sec)"].to_numpy(dtype=float)
            s   = smooth_series(raw, smooth_window) * G_FT_S2   # g → ft/s²
            s[np.diff(t, prepend=t[0]) > 5] = np.nan            # mask big gaps
            return s

        # ── Auto-select axis ──────────────────────────────────────────────────
        _gps_all = np.concatenate([_grp_derivs(r, smooth_window)["accel_ft_s2"].to_numpy() for r in runs])
        if _cmp_axis == "Auto (best r)":
            _best_r, _use_axis, _use_flip = -np.inf, "X", False
            for _ax in ["X", "Y", "Z"]:
                for _fl in [False, True]:
                    _gf = np.concatenate([_gf_accel(_grp_derivs(r, smooth_window), _ax, _fl) for r in runs])
                    _m  = np.isfinite(_gps_all) & np.isfinite(_gf)
                    if _m.sum() < 10:
                        continue
                    _r = float(np.corrcoef(_gps_all[_m], _gf[_m])[0, 1])
                    if _r > _best_r:
                        _best_r, _use_axis, _use_flip = _r, _ax, _fl
            st.caption(f"Auto-selected: GForce **{_use_axis}**  "
                       f"flip={_use_flip}  —  Pearson r = **{_best_r:.4f}**")
        else:
            _use_axis = _cmp_axis
            _use_flip = _cmp_flip

        # ── Assemble per-run arrays ───────────────────────────────────────────
        _run_gps, _run_gf = [], []
        for _r in runs:
            _g = _grp_derivs(_r, smooth_window)
            _run_gps.append(_g["accel_ft_s2"].to_numpy())
            _run_gf.append(_gf_accel(_g, _use_axis, _use_flip))

        _gf_all = np.concatenate(_run_gf)
        _m_all  = np.isfinite(_gps_all) & np.isfinite(_gf_all)
        _gps_v  = _gps_all[_m_all]
        _gf_v   = _gf_all[_m_all]
        _resid  = _gps_v - _gf_v

        # ── Per-run stats ─────────────────────────────────────────────────────
        _stat_rows = []
        for _i, _r in enumerate(runs):
            _m = np.isfinite(_run_gps[_i]) & np.isfinite(_run_gf[_i])
            if _m.sum() < 5:
                continue
            _gv, _av = _run_gps[_i][_m], _run_gf[_i][_m]
            _stat_rows.append({
                "Run": f"R{_i+1} {_r['direction']}",
                "Target (mph)": _r["target_speed"],
                "Pearson r": round(float(np.corrcoef(_gv, _av)[0, 1]), 4),
                "Bias GPS−GF (ft/s²)": round(float(np.mean(_gv - _av)), 3),
                "RMSE (ft/s²)": round(float(np.sqrt(np.mean((_gv - _av) ** 2))), 3),
                "GPS max |a| (ft/s²)": round(float(np.nanmax(np.abs(_gv))), 2),
                "GF  max |a| (ft/s²)": round(float(np.nanmax(np.abs(_av))), 2),
            })

        # ── Row 1: Overlay for selected run ───────────────────────────────────
        # st.markdown("#### Overlay — Selected Run")
        _sel_r   = runs[_cmp_run_idx]
        _sel_grp = _grp_derivs(_sel_r, smooth_window)
        _sel_gps = _sel_grp["accel_ft_s2"].to_numpy()
        _sel_gf  = _gf_accel(_sel_grp, _use_axis, _use_flip)
        _sel_d   = _sel_grp["Distance (mi)"].to_numpy()
        _sel_t   = (_sel_grp["Elapsed time (sec)"] - _sel_grp["Elapsed time (sec)"].iloc[0]).to_numpy()

        _cmp_use_time = st.radio(
            "X axis", ["Distance (mi)", "Time (s)"], horizontal=True, key="cmp_xax"
        ) == "Time (s)"
        _sel_x = _sel_t if _cmp_use_time else _sel_d
        _sel_xl = "Time from run start (s)" if _cmp_use_time else "Distance (mi)"

        _fig_ov = go.Figure()
        _fig_ov.add_trace(go.Scatter(x=_sel_x, y=_sel_gps, mode="lines",
                                      name="GPS-derived", line=dict(color="royalblue", width=2)))
        _fig_ov.add_trace(go.Scatter(x=_sel_x, y=_sel_gf, mode="lines",
                                      name=f"GForce {_use_axis}", line=dict(color="tomato", width=2)))
        _fig_ov.add_hline(y=0, line_dash="dot", line_color="gray")
        _fig_ov.update_layout(
            xaxis_title=_sel_xl, yaxis_title="Acceleration (ft/s²)",
            height=320, legend=dict(orientation="h", y=1.08),
            yaxis=dict(range=_pct_range(
                np.concatenate([_sel_gps[np.isfinite(_sel_gps)], _sel_gf[np.isfinite(_sel_gf)]]),
                pct=1, min_half=2.0, zero=True)),
        )
        st.plotly_chart(_fig_ov, use_container_width=True)

        # ── Row 2: Unity scatter (2-D histogram) + residual histogram ─────────
        st.markdown("#### All Runs — Unity Plot & Residuals")
        _col_unity, _col_resid = st.columns(2)

        with _col_unity:
            _ax_lim = _pct_range(np.concatenate([_gps_v, _gf_v]), pct=1, min_half=2.0, zero=True)
            _fig_unity = go.Figure()
            _fig_unity.add_trace(go.Histogram2dContour(
                x=_gps_v, y=_gf_v,
                colorscale="Viridis", reversescale=False,
                contours=dict(showlabels=False),
                showscale=True, name="Density",
                hovertemplate="GPS: %{x:.2f}<br>GForce: %{y:.2f}<extra></extra>",
            ))
            # y = x unity line
            _fig_unity.add_trace(go.Scatter(
                x=_ax_lim, y=_ax_lim, mode="lines",
                line=dict(color="white", width=1.5, dash="dash"), name="y = x",
                showlegend=True,
            ))
            _fig_unity.update_layout(
                xaxis_title="GPS-derived accel (ft/s²)",
                yaxis_title=f"GForce {_use_axis} (ft/s²)",
                xaxis=dict(range=_ax_lim), yaxis=dict(range=_ax_lim, scaleanchor="x"),
                height=420, title="Unity Plot (density contour)",
            )
            st.plotly_chart(_fig_unity, use_container_width=True)

        with _col_resid:
            _fig_res = go.Figure()
            _fig_res.add_trace(go.Histogram(
                x=_resid, nbinsx=80,
                marker_color="steelblue", opacity=0.8, name="GPS − GForce",
            ))
            _fig_res.add_vline(x=0, line_dash="dash", line_color="white")
            _fig_res.add_vline(x=float(np.mean(_resid)), line_dash="dot", line_color="yellow",
                               annotation_text=f"mean={np.mean(_resid):.3f}", annotation_position="top right")
            _fig_res.update_layout(
                xaxis_title="Residual: GPS − GForce (ft/s²)",
                yaxis_title="Count",
                height=420, title="Residual Distribution",
            )
            st.plotly_chart(_fig_res, use_container_width=True)

        # ── Row 3: Stats table ────────────────────────────────────────────────
        if _stat_rows:
            st.markdown("#### Per-Run Statistics")
            st.dataframe(pd.DataFrame(_stat_rows), use_container_width=True, hide_index=True)

        if len(_gps_v) > 0:
            _overall_r = float(np.corrcoef(_gps_v, _gf_v)[0, 1])
            _overall_rmse = float(np.sqrt(np.mean(_resid ** 2)))
            _overall_bias = float(np.mean(_resid))
            st.caption(
                f"All runs combined — N = {len(_gps_v):,}  |  "
                f"Pearson r = **{_overall_r:.4f}**  |  "
                f"RMSE = **{_overall_rmse:.3f} ft/s²**  |  "
                f"Bias (GPS−GF) = **{_overall_bias:.3f} ft/s²**"
            )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 6 — Settings
# ═══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.subheader("Run Detection")
    st.slider(
        "Smoothing window (samples)", 3, 31, step=2,
        help="Larger = smoother acceleration/jerk curves but less temporal resolution.",
        key="cfg_smooth",
    )
    st.slider(
        "Standing detection window (ft)", 10, 500, value=100, step=10,
        help="Half-width of the speed-sampling window centered on each trap crossing.",
        key="cfg_flying_window_ft",
    )
    st.slider(
        "Min speed to count as 'moving' (mph)", 0.01, 2.0, step=0.05,
        key="cfg_min_speed",
    )
    st.slider(
        "Min GPS samples per run", 10, 100,
        key="cfg_min_rows",
    )

    st.divider()

    # ── Measured section traps ────────────────────────────────────────────────
    st.subheader("Measured Section Traps")
    st.caption(
        "Hover over the Track Map to read lat/lon from the tooltip, "
        "then paste the coordinates here.  Step = 0.00001° ≈ 3 ft."
    )

    _tc_e, _tc_w = st.columns(2)
    with _tc_e:
        st.markdown("**East trap**")
        st.number_input("East lat", format="%.5f", step=0.00001, key="cfg_east_ctr_lat")
        st.number_input("East lon", format="%.5f", step=0.00001, key="cfg_east_ctr_lon")
    with _tc_w:
        st.markdown("**West trap**")
        st.number_input("West lat", format="%.5f", step=0.00001, key="cfg_west_ctr_lat")
        st.number_input("West lon", format="%.5f", step=0.00001, key="cfg_west_ctr_lon")

    _dist_ft = trap.chord_mi * 5280
    st.metric(
        "Distance between traps",
        f"{_dist_ft:.1f} ft  ({trap.chord_mi:.4f} mi)",
        delta=f"{_dist_ft - 5280:.1f} ft vs 5280 ft target",
        delta_color="inverse",
    )

    st.divider()

    # ── At-rest point map ─────────────────────────────────────────────────────
    st.subheader("At-Rest Point Map")
    _rest_thresh = st.slider(
        "At-rest speed threshold (mph)", 0.0, 5.0, value=0.2, step=0.1,
        format="%.1f mph",
        help="Points below this speed are shown on the map as orange dots.",
    )

    _rest_pts = df_raw[df_raw["Speed (mph)"] < _rest_thresh]

    _set_map_style = st.radio(
        "Map style", ["Street (carto)", "Satellite (ESRI)"],
        horizontal=True, label_visibility="collapsed", key="cfg_rest_map_style",
    )
    _set_satellite = _set_map_style == "Satellite (ESRI)"

    _fig_rest = go.Figure()
    _fig_rest.add_trace(go.Scattermap(
        lat=df_plot.loc[df_plot["Zone"] == "Transit", "Latitude"],
        lon=df_plot.loc[df_plot["Zone"] == "Transit", "Longitude"],
        mode="lines", line=dict(color="#457b9d", width=2), name="Transit",
        hovertemplate="Transit<br>lat=%{lat:.5f}<br>lon=%{lon:.5f}<extra></extra>",
    ))
    _fig_rest.add_trace(go.Scattermap(
        lat=df_plot.loc[df_plot["Zone"] == "Exercise Area", "Latitude"],
        lon=df_plot.loc[df_plot["Zone"] == "Exercise Area", "Longitude"],
        mode="lines", line=dict(color="#e63946", width=2), name="Exercise",
        hovertemplate="Exercise<br>lat=%{lat:.5f}<br>lon=%{lon:.5f}<extra></extra>",
    ))
    _add_trap_lines(_fig_rest)
    _fig_rest.add_trace(go.Scattermap(
        lat=_rest_pts["Latitude"], lon=_rest_pts["Longitude"],
        mode="markers", marker=dict(size=6, color="#e76f51", opacity=0.8),
        name=f"At rest (<{_rest_thresh:.1f} mph)",
        hovertemplate="At rest<br>lat=%{lat:.5f}<br>lon=%{lon:.5f}<br>%{customdata:.2f} mph<extra></extra>",
        customdata=_rest_pts["Speed (mph)"].to_numpy(),
    ))
    _fig_rest.update_layout(**_map_layout(_map_center_lat, _map_center_lon, _set_satellite))
    st.plotly_chart(_fig_rest, use_container_width=True)
    st.caption(f"{len(_rest_pts):,} points at rest (speed < {_rest_thresh:.1f} mph).")

    st.info("Changes take effect immediately on the next interaction.")
