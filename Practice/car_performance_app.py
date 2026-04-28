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
DATA_DIR = PRACTICE_DIR / "DataParquet"

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
parquet_files = sorted(DATA_DIR.glob("*speed*.parquet"))
if not parquet_files:
    st.error(f"No *.parquet files found in {DATA_DIR}")
    st.stop()

file_names = [f.name for f in parquet_files]
selected_name = st.sidebar.selectbox("Parquet file", file_names, index=len(file_names) - 1)
df = pd.read_parquet(DATA_DIR / selected_name)

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

# ── Correlation Analysis ──────────────────────────────────────────────────────
st.subheader("Correlation Analysis")

_corr_cols = [c for c in numeric_graphable if dv[c].notna().sum() > 10]
_corr_df   = dv[_corr_cols].copy()

if len(_corr_cols) < 2:
    st.info("Need at least two numeric channels for correlation analysis.")
else:
    _ctab_matrix, _ctab_scatter, _ctab_splom = st.tabs(
        ["Correlation Matrix", "Scatter Plot", "Scatter Matrix (SPLOM)"]
    )

    with _ctab_matrix:
        _corr = _corr_df.corr()
        _labels = _corr.columns.tolist()
        _fig_corr = go.Figure(go.Heatmap(
            z=_corr.values,
            x=_labels,
            y=_labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in _corr.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="X: %{x}<br>Y: %{y}<br>r = %{z:.3f}<extra></extra>",
        ))
        _fig_corr.update_layout(
            height=max(420, len(_labels) * 44 + 100),
            xaxis=dict(tickangle=-45),
            margin=dict(l=180, b=180, t=40, r=40),
        )
        st.plotly_chart(_fig_corr, use_container_width=True)

    with _ctab_scatter:
        _sc1, _sc2 = st.columns(2)
        _x_ch = _sc1.selectbox("X channel", _corr_cols, key="scat_x")
        _y_ch = _sc2.selectbox(
            "Y channel", _corr_cols,
            index=min(1, len(_corr_cols) - 1),
            key="scat_y",
        )
        _sd = _corr_df[[_x_ch, _y_ch]].dropna()
        if len(_sd) < 3:
            st.info("Not enough data points after dropping NaN.")
        else:
            _r_val = float(np.corrcoef(_sd[_x_ch], _sd[_y_ch])[0, 1])
            _m, _b = np.polyfit(_sd[_x_ch], _sd[_y_ch], 1)
            _x_line = np.array([float(_sd[_x_ch].min()), float(_sd[_x_ch].max())])
            _y_line = _m * _x_line + _b

            _fig_sc = go.Figure()
            _fig_sc.add_trace(go.Scattergl(
                x=_sd[_x_ch], y=_sd[_y_ch],
                mode="markers",
                marker=dict(size=3, opacity=0.4, color="steelblue"),
                name="Data",
                hovertemplate=f"{_x_ch}: %{{x:.3g}}<br>{_y_ch}: %{{y:.3g}}<extra></extra>",
            ))
            _fig_sc.add_trace(go.Scatter(
                x=_x_line, y=_y_line,
                mode="lines",
                line=dict(color="tomato", width=2),
                name="OLS fit",
            ))
            _fig_sc.update_layout(
                xaxis_title=_x_ch,
                yaxis_title=_y_ch,
                height=460,
                legend=dict(orientation="h", y=1.06),
                title=f"Pearson r = {_r_val:.4f}   |   slope = {_m:.4g}   |   N = {len(_sd):,}",
            )
            st.plotly_chart(_fig_sc, use_container_width=True)

    with _ctab_splom:
        _default_splom = _corr_cols[:min(4, len(_corr_cols))]
        _splom_channels = st.multiselect(
            "Channels for scatter matrix",
            options=_corr_cols,
            default=_default_splom,
            key="splom_channels",
        )
        if len(_splom_channels) < 2:
            st.info("Select at least 2 channels.")
        else:
            _splom_df = _corr_df[_splom_channels].dropna()
            _dims = [dict(label=c, values=_splom_df[c].tolist()) for c in _splom_channels]
            _fig_splom = go.Figure(go.Splom(
                dimensions=_dims,
                showupperhalf=False,
                marker=dict(size=2, opacity=0.3, color="steelblue"),
            ))
            _fig_splom.update_layout(
                height=max(500, len(_splom_channels) * 160),
                dragmode="select",
                margin=dict(l=120, b=120, t=40, r=40),
            )
            st.plotly_chart(_fig_splom, use_container_width=True)
            st.caption(
                f"N = {len(_splom_df):,} rows (after dropping any NaN across selected channels). "
                "Lower triangle only."
            )

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("Raw data table"):
    st.dataframe(dv.reset_index(drop=True), use_container_width=True)
