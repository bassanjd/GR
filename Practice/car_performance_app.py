"""
car_performance_app.py
Streamlit app to review speed_tracker parquet files.
- Any non-position channel can be selected for plotting.
- All selected channels on one figure with shared x-axis (one subplot row each).
- GPS track map colored by a chosen channel.
"""
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PRACTICE_DIR = Path(__file__).parent

POSITION_COLS = {"Date", "Time", "Latitude", "Longitude", "Altitude (ft)", "Accuracy (ft)"}
TIME_COLS = {"Elapsed time (sec)", "Elapsed (min)"}
# Native GPS-tracker outputs — not shifted by the datalogger offset.
GPS_TRACKER_COLS = {"Speed (mph)", "Distance (mi)"}

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

st.set_page_config(page_title="Car Performance Review", layout="wide")
st.title("Car Performance Review")

# ── File selector ─────────────────────────────────────────────────────────────
parquet_files = sorted(PRACTICE_DIR.glob("*.parquet"))
if not parquet_files:
    st.error(f"No *.parquet files found in {PRACTICE_DIR}")
    st.stop()

file_names = [f.name for f in parquet_files]
selected_name = st.sidebar.selectbox("Parquet file", file_names, index=len(file_names) - 1)
df = pd.read_parquet(PRACTICE_DIR / selected_name)

df = df.sort_values("Elapsed time (sec)").reset_index(drop=True)
df["Elapsed (min)"] = df["Elapsed time (sec)"] / 60.0

# ── Graphable channels ────────────────────────────────────────────────────────
excluded = POSITION_COLS | TIME_COLS
graphable = [c for c in df.columns if c not in excluded]
numeric_graphable = [c for c in graphable if pd.api.types.is_numeric_dtype(df[c])]

st.sidebar.markdown(f"**Rows:** {len(df):,}")
st.sidebar.markdown(f"**Duration:** {df['Elapsed time (sec)'].max() / 60:.1f} min")
st.sidebar.markdown(f"**Distance:** {df['Distance (mi)'].max():.2f} mi")

# ── Time range filter ─────────────────────────────────────────────────────────
t_min = float(df["Elapsed (min)"].min())
t_max = float(df["Elapsed (min)"].max())
t_range = st.sidebar.slider(
    "Elapsed time range (min)",
    min_value=t_min, max_value=t_max,
    value=(t_min, t_max), step=0.5,
)
dv = df[df["Elapsed (min)"].between(*t_range)].copy()

# ── Datalogger time offset ────────────────────────────────────────────────────
datalogger_offset_sec = st.sidebar.slider(
    "Datalogger time offset (sec)",
    min_value=-120, max_value=120, value=30, step=1,
    help="Shift datalogger channels (non-GPS) left/right to align with speed/position.",
)
datalogger_offset_min = datalogger_offset_sec / 60.0

# ── Map color selector ────────────────────────────────────────────────────────
map_color_col = st.sidebar.selectbox(
    "Map color channel",
    options=numeric_graphable,
    index=numeric_graphable.index("Speed (mph)") if "Speed (mph)" in numeric_graphable else 0,
)

# ── Channel selector ──────────────────────────────────────────────────────────
default_channels = [c for c in ["Speed (mph)"] if c in graphable]
selected_channels = st.sidebar.multiselect(
    "Channels to plot",
    options=graphable,
    default=default_channels,
)

# ── Per-channel value range sliders ──────────────────────────────────────────
# Build the set of channels that need a range slider (plot channels + map channel).
range_channels = list(dict.fromkeys(
    [c for c in selected_channels if c in numeric_graphable] +
    ([map_color_col] if map_color_col not in selected_channels else [])
))

ch_ranges: dict[str, tuple[float, float]] = {}
if range_channels:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Value ranges**")
    for ch in range_channels:
        lo = float(dv[ch].min())
        hi = float(dv[ch].max())
        if lo == hi:          # degenerate — no slider needed
            ch_ranges[ch] = (lo, hi)
        else:
            default_lo = max(lo, 80.0) if lo < 80.0 < hi else lo
            ch_ranges[ch] = st.sidebar.slider(
                ch, min_value=lo, max_value=hi,
                value=(default_lo, hi),
                key=f"range_{ch}",
            )

# For the chart: NaN each channel independently so a dropout in one
# channel creates a line gap without affecting any other channel.
dv_chart = dv.copy()
for ch, (lo, hi) in ch_ranges.items():
    out = ~dv_chart[ch].between(lo, hi)
    dv_chart.loc[out, ch] = float("nan")

# For the map: keep all GPS rows intact; range is applied via range_color.
# If the color channel is a datalogger channel, re-sample it at shifted times
# so the color at each GPS position reflects the offset-aligned reading.
import numpy as np
dv_map = dv.copy()
if map_color_col not in GPS_TRACKER_COLS and datalogger_offset_sec != 0:
    t = dv_map["Elapsed time (sec)"].to_numpy(float)
    c = dv_map[map_color_col].to_numpy(float)
    valid = ~np.isnan(c)
    if valid.sum() > 1:
        # For GPS time T, look up the datalogger value at (T - offset_sec).
        dv_map[map_color_col] = np.interp(
            t - datalogger_offset_sec,
            t[valid], c[valid],
            left=np.nan, right=np.nan,
        )

# ── Combined channel chart ────────────────────────────────────────────────────
# st.subheader("Channels")
if selected_channels:
    fig = go.Figure()
    for i, ch in enumerate(selected_channels):
        axis_id = "y" if i == 0 else f"y{i + 1}"
        color = COLORS[i % len(COLORS)]
        x_vals = dv_chart["Elapsed (min)"] + (
            0 if ch in GPS_TRACKER_COLS else datalogger_offset_min
        )
        fig.add_trace(go.Scatter(
            x=x_vals, y=dv_chart[ch],
            mode="lines",
            name=ch,
            line=dict(width=1, color=color),
            yaxis=axis_id,
        ))

    layout = dict(
        xaxis=dict(title="Elapsed time (min)", domain=[0, 1]),
        height=450,
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=60, b=40),
        hovermode="x unified",
        uirevision=selected_name,
    )
    right_offset = 0.0
    for i, ch in enumerate(selected_channels):
        color = COLORS[i % len(COLORS)]
        key = "yaxis" if i == 0 else f"yaxis{i + 1}"
        axis = dict(title=dict(text=ch, font=dict(color=color)), tickfont=dict(color=color))
        # Pin y-axis to the selected range when one exists.
        if ch in ch_ranges:
            axis["range"] = list(ch_ranges[ch])
        if i == 0:
            axis["side"] = "left"
        else:
            right_offset += 0.08
            axis.update(dict(
                overlaying="y",
                side="right",
                position=1 - right_offset + 0.08,
                anchor="free",
                showgrid=False,
            ))
        layout[key] = axis

    n_right = max(0, len(selected_channels) - 1)
    layout["xaxis"]["domain"] = [0, max(0.5, 1 - n_right * 0.08)]

    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one channel in the sidebar to plot.")

# ── GPS map ───────────────────────────────────────────────────────────────────
st.subheader("GPS Track")
if "Latitude" in dv_map.columns and dv_map["Latitude"].notna().any():
    gps = dv_map[dv_map["Latitude"].notna() & dv_map["Longitude"].notna()]
    lat_min, lat_max = gps["Latitude"].min(), gps["Latitude"].max()
    lon_min, lon_max = gps["Longitude"].min(), gps["Longitude"].max()
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    span = max(lat_max - lat_min, lon_max - lon_min)
    auto_zoom = max(1, min(18, math.floor(math.log2(360 / span)) - 1))

    color_range = list(ch_ranges[map_color_col]) if map_color_col in ch_ranges else None

    fig_map = px.scatter_mapbox(
        dv_map,
        lat="Latitude", lon="Longitude",
        color=map_color_col,
        color_continuous_scale="RdYlGn_r",
        range_color=color_range,
        hover_data={"Elapsed (min)": ":.1f", map_color_col: ":.2f"},
        center={"lat": center_lat, "lon": center_lon},
        zoom=auto_zoom,
        height=550,
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        uirevision=selected_name,
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})
else:
    st.warning("No GPS data in the selected time range.")

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("Raw data table"):
    st.dataframe(dv.reset_index(drop=True), use_container_width=True)
