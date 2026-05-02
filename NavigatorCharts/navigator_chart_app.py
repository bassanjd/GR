"""
Interactive Navigator Charts analysis.

Sidebar — date filter, polynomial-smoothing threshold slider, and calibration
          runs table with per-row Exclude checkboxes auto-populated by a greedy
          degree-2 polynomial fit algorithm.

Tab "Data Summary" — strip charts, loss line charts (with polynomial fits + R²),
                     and losses tables for All Data (left) vs Filtered (right).

Tab "Charts" — the four reference matrices styled to match the Excel export
               (tricolor scale, dark headers, black zero-axis hiding)
               for All Data (left) vs Filtered (right).  Includes export buttons.

"All Data" always uses every visible row; exclusions only affect "Filtered".
"""
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from navigator_chart_helpers import (
    SPEEDS,
    STOP_SIGN_SECONDS,
    build_reference_workbook,
    compute_losses,
    load_calibration_runs,
    losses_to_dicts,
    matrix_stop_go,
    matrix_transition,
    matrix_turn_loss,
)

st.set_page_config(page_title="Navigator Charts", layout="wide", page_icon="🏎️")

DATE_2026 = "2026-04-29"


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_runs():
    return load_calibration_runs()


def build_export_bytes(accel, decel, label):
    wb = build_reference_workbook(accel, decel, label)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def compute_auto_excludes(df, n_excl):
    """
    Return a boolean mask of the n_excl 2026 rows to exclude for the Filtered column.

    Greedy degree-2 polynomial fit: at each step selects the single 2026 run (any
    test type — a straight-speed exclusion improves both curves simultaneously) whose
    removal most reduces the total sum of squared residuals from quadratic fits to
    both the accel and decel loss curves.  Ties broken by the run's z-score within
    its own group (prefer the most extreme run).  Stops at n_excl exclusions.
    """
    mask = pd.Series(False, index=df.index)
    if n_excl == 0:
        return mask
    df26 = df[df["date"] == DATE_2026]

    gstats = (df26.groupby(["test_type", "target_mph"])["time_s"]
              .agg(["mean", "std"]))
    gstats["std"] = gstats["std"].clip(lower=0.01)

    def _z(idx):
        row = df26.loc[idx]
        s = gstats.loc[(row["test_type"], row["target_mph"])]
        return abs(row["time_s"] - s["mean"]) / s["std"]

    def _ssr(excl_mask):
        """Sum of squared residuals from degree-2 polynomial fits on both curves."""
        kept = df26[~excl_mask.loc[df26.index]]
        grp = kept.groupby(["test_type", "target_mph"])["time_s"].mean().unstack("test_type")
        need = {"straight_speed", "start_speed", "speed_stop"}
        if not need.issubset(grp.columns):
            return float("inf")
        grp = grp.dropna()
        if len(grp) < 4:
            return float("inf")
        x = np.array(grp.index.tolist(), dtype=float)
        s = grp["straight_speed"].values
        a = grp["start_speed"].values - s
        d = grp["speed_stop"].values  - s
        def ss(y):
            return float(np.sum((y - np.polyval(np.polyfit(x, y, 2), x)) ** 2))
        return ss(a) + ss(d)

    for _ in range(n_excl):
        cur = _ssr(mask)
        if cur == 0.0:
            break
        best_red, best_z, best_idx = 0.0, -1.0, None
        for idx in df26.index[~mask.loc[df26.index]]:
            trial = mask.copy()
            trial.loc[idx] = True
            red = cur - _ssr(trial)
            z = _z(idx)
            if red > best_red + 1e-9 or (abs(red - best_red) < 1e-9 and z > best_z):
                best_red, best_z, best_idx = red, z, idx
        if best_idx is None or best_red <= 0:
            break
        mask.loc[best_idx] = True

    return mask


# ── Chart builders ────────────────────────────────────────────────────────────

TYPE_COLORS = {
    "straight_speed": "#1976D2",
    "start_speed":    "#388E3C",
    "speed_stop":     "#D32F2F",
}

_LABELS = {
    "target_mph": "Target MPH", "time_s": "Time (s)",
    "test_type":  "Test Type",  "date":   "Date",
    "run_number": "Run #",      "direction": "Direction",
}


def time_strip_chart(df, title):
    if df.empty:
        return go.Figure().update_layout(title=title, height=300)
    fig = px.strip(
        df, x="target_mph", y="time_s", color="test_type",
        title=title, stripmode="overlay",
        hover_data=["date", "run_number", "direction"],
        color_discrete_map=TYPE_COLORS,
        labels=_LABELS,
    )
    fig.update_traces(marker_size=7, opacity=0.8)
    fig.update_layout(
        height=300, margin=dict(t=35, b=5, l=5, r=5),
        legend=dict(orientation="h", y=-0.22, title_text=""),
    )
    return fig


def loss_line_chart(losses, title):
    if losses is None:
        return go.Figure().update_layout(title=title, height=280)
    x = losses["MPH"].values.astype(float)
    a = losses["Accel Loss (s)"].values
    d = losses["Decel Loss (s)"].values
    x_fine = np.linspace(x[0], x[-1], 200)
    ca = np.polyfit(x, a, 2)
    cd = np.polyfit(x, d, 2)

    def r2(y, coef):
        ss_res = float(np.sum((y - np.polyval(coef, x)) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    r2a = r2(a, ca)
    r2d = r2(d, cd)

    fig = go.Figure([
        go.Scatter(x=x, y=a, name=f"Accel (R²={r2a:.4f})", mode="lines+markers",
                   line=dict(color="#388E3C", width=2), marker_size=7),
        go.Scatter(x=x, y=d, name=f"Decel (R²={r2d:.4f})", mode="lines+markers",
                   line=dict(color="#D32F2F", width=2), marker_size=7),
        go.Scatter(x=x_fine, y=np.polyval(ca, x_fine), mode="lines",
                   showlegend=False, line=dict(color="#388E3C", width=1.5, dash="dash")),
        go.Scatter(x=x_fine, y=np.polyval(cd, x_fine), mode="lines",
                   showlegend=False, line=dict(color="#D32F2F", width=1.5, dash="dash")),
    ])
    fig.update_layout(
        title=title, height=280,
        xaxis_title="Speed (mph)", yaxis_title="Time Lost (s)",
        margin=dict(t=35, b=5, l=5, r=5),
        legend=dict(orientation="h", y=-0.22, title_text=""),
    )
    return fig


def losses_table(losses):
    if losses is None:
        st.caption("Insufficient 2026 data to compute losses.")
        return
    styled = losses.style.format({
        "Straight (s)":   "{:.2f}",
        "Accel Loss (s)": "{:.2f}",
        "Decel Loss (s)": "{:.2f}",
        "Actual MPH":     "{:.2f}",
        "Error (%)":      "{:.2f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ── HTML matrix rendering (matches Excel export styling) ──────────────────────

def _hex_to_rgb(h):
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp_color(c1, c2, t):
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return (f"{int(r1 + t*(r2-r1)):02X}"
            f"{int(g1 + t*(g2-g1)):02X}"
            f"{int(b1 + t*(b2-b1)):02X}")


def _scale_color(val, vmin, vmid, vmax, c_lo, c_mid, c_hi):
    """Tricolor interpolation: c_lo at vmin, c_mid at vmid, c_hi at vmax."""
    if val <= vmid:
        t = (val - vmin) / (vmid - vmin) if vmid > vmin else 0.0
        return _lerp_color(c_lo, c_mid, max(0.0, min(1.0, t)))
    t = (val - vmid) / (vmax - vmid) if vmax > vmid else 1.0
    return _lerp_color(c_mid, c_hi, max(0.0, min(1.0, t)))


def _luminance(hex_color):
    r, g, b = _hex_to_rgb(hex_color)
    return 0.299 * r + 0.587 * g + 0.114 * b


# Inline style strings matching Excel constants (C_HEADER_BG, C_AXIS_BG, etc.)
_S_TITLE  = ("background:#D6E4F0;color:#1F4E79;font-weight:bold;font-size:12px;"
              "padding:5px 7px;")
_S_SUB    = "color:#595959;font-style:italic;font-size:9px;padding:2px 7px 5px;"
_S_CORNER = ("background:#1F4E79;color:#FFF;font-weight:bold;font-size:9px;"
             "text-align:center;padding:3px 6px;border:1px solid #9DC3E6;"
             "white-space:nowrap;")
_S_HDR    = ("background:#1F4E79;color:#FFF;font-weight:bold;font-size:10px;"
             "text-align:center;padding:4px 10px;border:1px solid #9DC3E6;")
_S_AXIS   = ("background:#2E75B6;color:#FFF;font-weight:bold;font-size:10px;"
             "text-align:center;padding:4px 10px;border:1px solid #9DC3E6;")
_S_BLANK  = "background:#D9D9D9;border:1px solid #9DC3E6;padding:4px 10px;"
_S_BLACK  = ("background:#000;color:#000;font-size:10px;text-align:center;"
             "padding:4px 10px;border:1px solid #333;")


def matrix_html(matrix, title, subtitle, c_lo, c_mid, c_hi,
                mid_value, hide_zero_axis=False):
    """HTML string for one reference matrix, styled like the Excel export."""
    visible = [
        val
        for i, row in enumerate(matrix)
        for j, val in enumerate(row)
        if val is not None and not (hide_zero_axis and (i == 0 or j == 0))
    ]
    vmin = min(visible) if visible else 0.0
    vmax = max(visible) if visible else 1.0
    vmid = max(vmin, min(vmax, mid_value))

    n = len(SPEEDS)
    h = [f'<table style="border-collapse:collapse;'
         f'font-family:Calibri,Arial,sans-serif;margin-bottom:20px;">']
    h.append(f'<tr><td colspan="{n+1}" style="{_S_TITLE}">{title}</td></tr>')
    h.append(f'<tr><td colspan="{n+1}" style="{_S_SUB}">{subtitle}</td></tr>')

    h.append('<tr>')
    h.append(f'<td style="{_S_CORNER}">In ↓&nbsp;&nbsp;Out →</td>')
    for j, spd in enumerate(SPEEDS):
        s = _S_BLACK if (hide_zero_axis and j == 0) else _S_HDR
        h.append(f'<td style="{s}">{spd}</td>')
    h.append('</tr>')

    for i, in_spd in enumerate(SPEEDS):
        zero_row = hide_zero_axis and i == 0
        h.append('<tr>')
        h.append(f'<td style="{_S_BLACK if zero_row else _S_AXIS}">{in_spd}</td>')
        for j, val in enumerate(matrix[i]):
            zero_col = hide_zero_axis and j == 0
            if zero_row or zero_col:
                text = "" if val is None else f"{val:.1f}"
                h.append(f'<td style="{_S_BLACK}">{text}</td>')
            elif val is None:
                h.append(f'<td style="{_S_BLANK}"></td>')
            else:
                bg = _scale_color(val, vmin, vmid, vmax, c_lo, c_mid, c_hi)
                fg = "000000" if _luminance(bg) > 140 else "FFFFFF"
                s = (f"background:#{bg};color:#{fg};font-size:10px;"
                     f"text-align:center;padding:4px 10px;border:1px solid #9DC3E6;")
                h.append(f'<td style="{s}">{val:.1f}</td>')
        h.append('</tr>')

    h.append('</table>')
    return "".join(h)


def render_matrix_html(accel, decel):
    """Render all four reference matrices as styled HTML tables."""
    if accel is None:
        st.caption("Insufficient 2026 data to build matrices.")
        return

    m1 = matrix_transition(accel, decel)
    m2 = matrix_stop_go(accel, decel)
    m3 = matrix_turn_loss(accel, decel, ref_mph=15)
    m4 = matrix_turn_loss(accel, decel, ref_mph=20)

    html = '<div style="overflow-x:auto;">'
    html += matrix_html(
        m1,
        title="1.  Speed Transition Time (seconds)",
        subtitle="Extra seconds vs. flying straight through · blank diagonal = no change",
        c_lo="63BE7B", c_mid="FFEB84", c_hi="F8696B",
        mid_value=0.0,
    )
    html += matrix_html(
        m2,
        title="2.  Available Pause at Stop Sign (seconds)",
        subtitle=(f"Standstill seconds remaining inside a {int(STOP_SIGN_SECONDS)}-second "
                  "mandatory stop · Out=0 col = raw brake cost"),
        c_lo="F8696B", c_mid="FFEB84", c_hi="63BE7B",
        mid_value=STOP_SIGN_SECONDS / 2,
        hide_zero_axis=True,
    )
    html += matrix_html(
        m3,
        title="3.  Turn Time Lost vs. 15 mph reference (seconds)",
        subtitle="Extra seconds vs. In=15→15 · In=0 row / Out=0 col = raw accel/decel costs",
        c_lo="63BE7B", c_mid="FFEB84", c_hi="F8696B",
        mid_value=0.0,
        hide_zero_axis=True,
    )
    html += matrix_html(
        m4,
        title="4.  Turn Time Lost vs. 20 mph reference (seconds)",
        subtitle="Extra seconds vs. In=20→20 · In=0 row / Out=0 col = raw accel/decel costs",
        c_lo="63BE7B", c_mid="FFEB84", c_hi="F8696B",
        mid_value=0.0,
        hide_zero_axis=True,
    )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

df_all = load_runs()

# ── Sidebar: date filter + editable runs table ────────────────────────────────

with st.sidebar:
    st.title("Calibration Runs")

    n_excl_auto = st.slider(
        "Auto-exclude runs", min_value=0, max_value=15, value=6,
        help="Greedy degree-2 polynomial fit selects which runs to exclude. "
             "6 is a good default — gives smooth curves with minimal data loss.",
    )

    all_dates = sorted(df_all["date"].unique(), reverse=True)
    sel_dates = st.multiselect("Date", all_dates, default=[DATE_2026])

    df_visible = df_all[df_all["date"].isin(sel_dates)].reset_index(drop=True)

    auto_excl = compute_auto_excludes(df_visible, n_excl_auto)
    n_auto = int(auto_excl.sum())

    df_edit = df_visible.copy()
    df_edit.insert(0, "Excl.", auto_excl.values)

    st.caption(
        f"{n_auto} run(s) auto-excluded (degree-2 polynomial fit). "
        "Uncheck to override."
    )

    edited = st.data_editor(
        df_edit,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Excl.":      st.column_config.CheckboxColumn("Excl.",  width="small"),
            "date":       st.column_config.TextColumn("Date",       width="small"),
            "test_type":  st.column_config.TextColumn("Type",       width="medium"),
            "target_mph": st.column_config.NumberColumn("MPH",      width="small"),
            "run_number": st.column_config.NumberColumn("#",        width="small"),
            "time_s":     st.column_config.NumberColumn("Sec",      format="%.2f", width="small"),
            "notes":      st.column_config.TextColumn("Notes",      width="small"),
            "direction":  None,
            "time_raw":   None,
        },
        height=700,
        disabled=[c for c in df_edit.columns if c != "Excl."],
    )

n_excl   = int(edited["Excl."].sum())
df_kept  = edited[~edited["Excl."]].drop(columns=["Excl."])

# ── Precompute losses for both views ──────────────────────────────────────────

losses_all  = compute_losses(df_visible)
accel_all,  decel_all  = losses_to_dicts(losses_all)

losses_filt = compute_losses(df_kept)
accel_filt, decel_filt = losses_to_dicts(losses_filt)

filt_label = f"Filtered ({n_excl} excluded)" if n_excl else "Filtered (none excluded)"

# ── Main tabs ─────────────────────────────────────────────────────────────────

st.title("Navigator Charts")

tab_summary, tab_charts = st.tabs(["Data Summary", "Charts"])

# ── Tab: Data Summary ─────────────────────────────────────────────────────────

with tab_summary:
    col_all, col_filt = st.columns(2)

    with col_all:
        st.subheader("All Data")
        st.plotly_chart(time_strip_chart(df_visible, "Run Times"),
                        use_container_width=True)
        st.plotly_chart(loss_line_chart(losses_all, "Accel / Decel Losses"),
                        use_container_width=True)
        losses_table(losses_all)

    with col_filt:
        st.subheader(filt_label)
        st.plotly_chart(time_strip_chart(df_kept, "Run Times (filtered)"),
                        use_container_width=True)
        st.plotly_chart(loss_line_chart(losses_filt, "Accel / Decel Losses (filtered)"),
                        use_container_width=True)
        losses_table(losses_filt)

# ── Tab: Charts ───────────────────────────────────────────────────────────────

with tab_charts:
    col_charts_all, col_charts_filt = st.columns(2)

    with col_charts_all:
        st.subheader("All Data")
        if accel_all is not None:
            st.download_button(
                "⬇ Export reference charts (all data)",
                data=build_export_bytes(accel_all, decel_all, "all data"),
                file_name="reference_charts_all.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button("⬇ Export reference charts (all data)", disabled=True,
                      use_container_width=True)
        render_matrix_html(accel_all, decel_all)

    with col_charts_filt:
        st.subheader(filt_label)
        if accel_filt is not None:
            st.download_button(
                "⬇ Export reference charts (filtered)",
                data=build_export_bytes(accel_filt, decel_filt, "filtered"),
                file_name="reference_charts_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button("⬇ Export reference charts (filtered)", disabled=True,
                      use_container_width=True)
        render_matrix_html(accel_filt, decel_filt)
1