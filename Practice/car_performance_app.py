"""
car_performance_app.py
Streamlit app to review speed_tracker, RaceBox, and Generic datalogger parquet files.
- Any non-position channel can be selected for plotting.
- All selected channels on one figure with shared x-axis (one subplot row each).
- GPS track map colored by a chosen channel.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import loaders

PRACTICE_DIR = Path(__file__).parent
DATA_DIR = PRACTICE_DIR / "DataParquet"

MAP_MAX_ROWS = 4_000  # downsample GPS track for the map tab

POSITION_COLS = {
    "Date", "Time", "Datetime",
    "Latitude", "Longitude", "Altitude (ft)", "Accuracy (ft)",
}
TIME_COLS = {"Elapsed time (sec)", "Elapsed (min)"}
# GPS-native channels that do NOT need the datalogger time offset applied.
# Only relevant for speed_tracker files where GPS and datalogger are separate devices.
GPS_NATIVE_COLS = {"Speed (mph)", "Distance (mi)"}

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

@st.cache_data
def _load(path: str) -> pd.DataFrame:
    raw = loaders.load_any(path)
    raw = raw.sort_values("Elapsed time (sec)").reset_index(drop=True)
    raw["Elapsed (min)"] = raw["Elapsed time (sec)"] / 60.0
    # Downcast float64 → float32 to halve numeric memory usage.
    for col in raw.select_dtypes("float64").columns:
        raw[col] = raw[col].astype("float32")
    return raw


st.set_page_config(page_title="Car Performance Review", layout="wide")
st.title("Car Performance Review")

# ── File selector ─────────────────────────────────────────────────────────────
parquet_files = sorted(DATA_DIR.glob("*.parquet"))
if not parquet_files:
    st.error(f"No *.parquet files found in {DATA_DIR}")
    st.stop()

file_names = [f.name for f in parquet_files]
selected_name = st.sidebar.selectbox("Parquet file", file_names, index=len(file_names) - 1)
df = _load(str(DATA_DIR / selected_name))

# ── Graphable channels ────────────────────────────────────────────────────────
excluded = POSITION_COLS | TIME_COLS
graphable = [c for c in df.columns if c not in excluded]
numeric_graphable = [c for c in graphable if pd.api.types.is_numeric_dtype(df[c])]

st.sidebar.markdown(f"**Rows:** {len(df):,}")
st.sidebar.markdown(f"**Duration:** {df['Elapsed time (sec)'].max() / 60:.1f} min")
st.sidebar.markdown(f"**Distance:** {df['Distance (mi)'].max():.2f} mi")

if "Datetime" in df.columns and df["Datetime"].notna().any():
    _dt = pd.to_datetime(df["Datetime"], errors="coerce").dropna()
    if not _dt.empty:
        st.sidebar.markdown(f"**First:** {_dt.iloc[0].strftime('%Y-%m-%d %H:%M:%S')}")
        st.sidebar.markdown(f"**Last:** {_dt.iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}")
elif "Date" in df.columns and "Time" in df.columns:
    _dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce").dropna()
    if not _dt.empty:
        st.sidebar.markdown(f"**First:** {_dt.iloc[0].strftime('%Y-%m-%d %H:%M:%S')}")
        st.sidebar.markdown(f"**Last:** {_dt.iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}")

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
# Only speed_tracker files have a separate GPS device and datalogger that may
# have a clock offset.  RaceBox and Generic files log all channels in sync.
_is_speed_tracker = "speed_tracker" in selected_name.lower()
if _is_speed_tracker:
    datalogger_offset_sec = st.sidebar.slider(
        "Datalogger time offset (sec)",
        min_value=-120, max_value=120, value=30, step=1,
        help="Shift datalogger channels (non-GPS) left/right to align with speed/position.",
    )
else:
    datalogger_offset_sec = 0
    st.sidebar.caption("Datalogger offset: N/A — all channels are synchronized.")
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
# Only copy the columns actually used in the chart/range masking.
_chart_cols = list(dict.fromkeys(
    ["Elapsed (min)"] +
    [c for c in selected_channels if c in dv.columns] +
    [c for c in ch_ranges if c in dv.columns]
))
dv_chart = dv[_chart_cols].copy()
for ch, (lo, hi) in ch_ranges.items():
    if ch in dv_chart.columns:
        out = ~dv_chart[ch].between(lo, hi)
        dv_chart.loc[out, ch] = float("nan")

# For the map: keep all GPS rows intact; range is applied via range_color.
# For speed_tracker files, re-sample datalogger channels at shifted times so
# the color at each GPS position reflects the offset-aligned reading.
# Only copy columns needed for the map.
_map_cols = [c for c in ["Elapsed time (sec)", "Latitude", "Longitude", "Elapsed (min)", map_color_col] if c in dv.columns]
dv_map = dv[_map_cols].copy()
if _is_speed_tracker and map_color_col not in GPS_NATIVE_COLS and datalogger_offset_sec != 0:
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

# ── Correlation cols (computed here so session state can be initialised before tabs) ──
_corr_cols = [c for c in numeric_graphable if dv[c].notna().sum() > 10]

if "scat_x" not in st.session_state and len(_corr_cols) >= 1:
    st.session_state["scat_x"] = _corr_cols[0]
if "scat_y" not in st.session_state and len(_corr_cols) >= 2:
    st.session_state["scat_y"] = _corr_cols[1]

# ── Main tabs ─────────────────────────────────────────────────────────────────
_tab_channels, _tab_map, _tab_corr, _tab_scatter, _tab_data = st.tabs(
    ["Channels", "GPS Track", "Correlation", "Scatter", "Raw Data"]
)

with _tab_channels:
    if selected_channels:
        fig = go.Figure()
        for i, ch in enumerate(selected_channels):
            axis_id = "y" if i == 0 else f"y{i + 1}"
            color = COLORS[i % len(COLORS)]
            x_vals = dv_chart["Elapsed (min)"] + (
                0 if (not _is_speed_tracker or ch in GPS_NATIVE_COLS)
                else datalogger_offset_min
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

with _tab_map:
    if "Latitude" in dv_map.columns and dv_map["Latitude"].notna().any():
        gps = dv_map[dv_map["Latitude"].notna() & dv_map["Longitude"].notna()]
        if len(gps) > MAP_MAX_ROWS:
            gps = gps.iloc[::math.ceil(len(gps) / MAP_MAX_ROWS)]
        lat_min, lat_max = gps["Latitude"].min(), gps["Latitude"].max()
        lon_min, lon_max = gps["Longitude"].min(), gps["Longitude"].max()
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        span = max(lat_max - lat_min, lon_max - lon_min)
        auto_zoom = max(1, min(18, math.floor(math.log2(360 / span)) - 1))

        color_range = list(ch_ranges[map_color_col]) if map_color_col in ch_ranges else None

        fig_map = px.scatter_mapbox(
            gps,
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

with _tab_corr:
    if len(_corr_cols) < 2:
        st.info("Need at least two numeric channels for correlation analysis.")
    else:
        # ── Correlation Matrix ────────────────────────────────────────────────
        st.subheader("Correlation Matrix")
        _corr = dv[_corr_cols].corr()
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

with _tab_scatter:
    _corr_df = dv[_corr_cols].copy()
    if len(_corr_cols) < 2:
        st.info("Need at least two numeric channels.")
    else:
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

            _fig_sc = go.Figure()
            _fig_sc.add_trace(go.Scattergl(
                x=_sd[_x_ch], y=_sd[_y_ch],
                mode="markers",
                marker=dict(size=3, opacity=0.4, color="steelblue"),
                name="Data",
                hovertemplate=f"{_x_ch}: %{{x:.3g}}<br>{_y_ch}: %{{y:.3g}}<extra></extra>",
            ))
            _fig_sc.update_layout(
                xaxis_title=_x_ch,
                yaxis_title=_y_ch,
                height=460,
                showlegend=False,
                title=f"Pearson r = {_r_val:.4f}   |   N = {len(_sd):,}",
            )
            st.plotly_chart(_fig_sc, use_container_width=True)

with _tab_data:
    st.dataframe(dv.reset_index(drop=True), use_container_width=True)
