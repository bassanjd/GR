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
from datetime import datetime

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



# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_runs():
    return load_calibration_runs()


def build_export_bytes(accel, decel, label, color_scale=True):
    wb = build_reference_workbook(accel, decel, label, color_scale=color_scale)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def compute_auto_excludes(df, n_excl):
    """
    Return (mask, excl_order) where mask is a boolean Series and excl_order is an
    integer Series (1 = first excluded, 2 = second, …; 0 = not excluded).

    Greedy degree-2 polynomial fit: at each step selects the single run from the latest
    date (any test type — a straight-speed exclusion improves both curves simultaneously) whose
    removal most reduces the total sum of squared residuals from quadratic fits to
    both the accel and decel loss curves.  Ties broken by the run's z-score within
    its own group (prefer the most extreme run).  Stops at n_excl exclusions.
    """
    mask = pd.Series(False, index=df.index)
    excl_order = pd.Series(0, index=df.index)
    if n_excl == 0:
        return mask, excl_order
    df26 = df[df["date"] == df["date"].max()]

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

    for step in range(1, n_excl + 1):
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
        excl_order.loc[best_idx] = step

    return mask, excl_order


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


def fit_losses(losses):
    """Return (ca, cd) degree-2 polynomial coefficients, or (None, None) if insufficient data."""
    if losses is None or len(losses) < 3:
        return None, None
    x = losses["MPH"].values.astype(float)
    return np.polyfit(x, losses["Accel Loss (s)"].values, 2), np.polyfit(x, losses["Decel Loss (s)"].values, 2)


def _poly_eq(coef, var="x"):
    a2, a1, a0 = coef
    parts = [f"{a2:.6f}{var}²", f"{'+' if a1 >= 0 else '-'} {abs(a1):.6f}{var}", f"{'+' if a0 >= 0 else '-'} {abs(a0):.6f}"]
    return " ".join(parts)


def loss_line_chart(losses, title, show_fit=True):
    if losses is None:
        return go.Figure().update_layout(title=title, height=280)
    x = losses["MPH"].values.astype(float)
    a = losses["Accel Loss (s)"].values
    d = losses["Decel Loss (s)"].values

    traces = []
    if show_fit:
        ca, cd = fit_losses(losses)
        if ca is not None:
            x_fine = np.linspace(x[0], x[-1], 200)

            r2a = _r2(a, ca, x)
            r2d = _r2(d, cd, x)
            accel_name = f"Accel (R²={r2a:.4f})"
            decel_name = f"Decel (R²={r2d:.4f})"
            traces += [
                go.Scatter(x=x_fine, y=np.polyval(ca, x_fine), mode="lines",
                           showlegend=False, line=dict(color="#388E3C", width=1.5, dash="dash")),
                go.Scatter(x=x_fine, y=np.polyval(cd, x_fine), mode="lines",
                           showlegend=False, line=dict(color="#D32F2F", width=1.5, dash="dash")),
            ]
        else:
            accel_name, decel_name = "Accel", "Decel"
    else:
        accel_name = "Accel"
        decel_name = "Decel"

    fig = go.Figure([
        go.Scatter(x=x, y=a, name=accel_name, mode="lines+markers",
                   line=dict(color="#388E3C", width=2), marker_size=7),
        go.Scatter(x=x, y=d, name=decel_name, mode="lines+markers",
                   line=dict(color="#D32F2F", width=2), marker_size=7),
        *traces,
    ])
    fig.update_layout(
        title=title, height=280,
        xaxis_title="Speed (mph)", yaxis_title="Time Lost (s)",
        margin=dict(t=35, b=5, l=5, r=5),
        legend=dict(orientation="h", y=-0.22, title_text=""),
    )
    return fig


def _r2(y, coef, x):
    ss_res = float(np.sum((y - np.polyval(coef, x)) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def losses_table(losses, ca=None, cd=None):
    if losses is None:
        st.caption("Insufficient data to compute losses.")
        return
    display_cols = ["MPH", "Straight (s)", "Accel Loss (s)", "Decel Loss (s)"]
    df = losses[display_cols].copy()
    fmt = {
        "Straight (s)":   "{:.2f}",
        "Accel Loss (s)": "{:.2f}",
        "Decel Loss (s)": "{:.2f}",
    }
    if ca is not None and cd is not None:
        x = losses["MPH"].values.astype(float)
        df["Fit Accel Loss (s)"] = np.polyval(ca, x)
        df["Fit Decel Loss (s)"] = np.polyval(cd, x)
        fmt["Fit Accel Loss (s)"] = "{:.2f}"
        fmt["Fit Decel Loss (s)"] = "{:.2f}"
    st.dataframe(df.style.format(fmt), use_container_width=True, hide_index=True)
    if ca is not None and cd is not None:
        x = losses["MPH"].values.astype(float)
        r2a = _r2(losses["Accel Loss (s)"].values, ca, x)
        r2d = _r2(losses["Decel Loss (s)"].values, cd, x)
        st.caption(f"Accel fit (R²={r2a:.4f}): {_poly_eq(ca)}")
        st.caption(f"Decel fit (R²={r2d:.4f}): {_poly_eq(cd)}")


_TYPE_LABELS = {
    "straight_speed": "Straight",
    "start_speed":    "Start",
    "speed_stop":     "Stop",
}


def run_pivot_table(df, df_ref=None):
    """Pivot of count/mean/std time_s by speed (rows) × run type (columns).

    When df_ref is provided, N cells that differ from the reference pivot are highlighted.
    """
    if df is None or df.empty:
        st.caption("No data.")
        return
    _STAT_ORDER = ["Avg", "Std", "N"]
    piv = (
        df.groupby(["target_mph", "test_type"])["time_s"]
        .agg(N="count", Avg="mean", Std="std")
        .unstack("test_type")
        .swaplevel(axis=1)
    )
    piv.columns = pd.MultiIndex.from_tuples(
        [(_TYPE_LABELS.get(t, t), stat) for t, stat in piv.columns]
    )
    types_sorted = sorted(piv.columns.get_level_values(0).unique())
    piv = piv[[(t, s) for t in types_sorted for s in _STAT_ORDER if (t, s) in piv.columns]]
    piv.index.name = "MPH"
    fmt = {c: ("{:.0f}" if c[1] == "N" else "{:.2f}") for c in piv.columns}

    if df_ref is not None and not df_ref.empty:
        piv_ref = (
            df_ref.groupby(["target_mph", "test_type"])["time_s"]
            .agg(N="count")
            .unstack("test_type")
            .swaplevel(axis=1)
        )
        piv_ref.columns = pd.MultiIndex.from_tuples(
            [(_TYPE_LABELS.get(t, t), stat) for t, stat in piv_ref.columns]
        )

        def _highlight_n(val, ref_val):
            if pd.isna(val) or pd.isna(ref_val) or val != ref_val:
                return "background-color: #1C3352; color: #E8EFF7"
            return ""

        def _apply_n_highlight(df_style):
            styles = pd.DataFrame("", index=piv.index, columns=piv.columns)
            for col in piv.columns:
                if col[1] == "N":
                    ref_col = piv_ref[col] if col in piv_ref.columns else None
                    for idx in piv.index:
                        val = piv.loc[idx, col]
                        ref_val = piv_ref.loc[idx, col] if (ref_col is not None and idx in piv_ref.index) else None
                        if ref_val is None or pd.isna(val) or val != ref_val:
                            styles.loc[idx, col] = "background-color: #1C3352; color: #E8EFF7"
            return styles

        styled = piv.style.format(fmt, na_rep="—").apply(_apply_n_highlight, axis=None)
    else:
        styled = piv.style.format(fmt, na_rep="—")

    st.dataframe(styled, use_container_width=True)


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
                mid_value, hide_zero_axis=False, color_scale=True):
    """HTML string for one reference matrix, styled like the Excel export."""
    visible = [
        round(val, 1)
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
                if color_scale:
                    bg = _scale_color(round(val, 1), vmin, vmid, vmax, c_lo, c_mid, c_hi)
                    fg = "000000" if _luminance(bg) > 140 else "FFFFFF"
                else:
                    bg, fg = "F2F2F2", "000000"
                s = (f"background:#{bg};color:#{fg};font-size:10px;"
                     f"text-align:center;padding:4px 10px;border:1px solid #9DC3E6;")
                h.append(f'<td style="{s}">{val:.1f}</td>')
        h.append('</tr>')

    h.append('</table>')
    return "".join(h)


def render_matrix_html(accel, decel, color_scale=True):
    """Render all four reference matrices as styled HTML tables."""
    if accel is None:
        st.caption("Insufficient data to build matrices.")
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
        color_scale=color_scale,
    )
    html += matrix_html(
        m2,
        title="2.  Available Pause at Stop Sign (seconds)",
        subtitle=(f"Standstill seconds remaining inside a {int(STOP_SIGN_SECONDS)}-second "
                  "mandatory stop · Out=0 col = raw brake cost"),
        c_lo="F8696B", c_mid="FFEB84", c_hi="63BE7B",
        mid_value=STOP_SIGN_SECONDS / 2,
        hide_zero_axis=True,
        color_scale=color_scale,
    )
    html += matrix_html(
        m3,
        title="3.  Turn Time Lost vs. 15 mph reference (seconds)",
        subtitle="Extra seconds vs. In=15→15 · In=0 row / Out=0 col = raw accel/decel costs",
        c_lo="63BE7B", c_mid="FFEB84", c_hi="F8696B",
        mid_value=0.0,
        hide_zero_axis=True,
        color_scale=color_scale,
    )
    html += matrix_html(
        m4,
        title="4.  Turn Time Lost vs. 20 mph reference (seconds)",
        subtitle="Extra seconds vs. In=20→20 · In=0 row / Out=0 col = raw accel/decel costs",
        c_lo="63BE7B", c_mid="FFEB84", c_hi="F8696B",
        mid_value=0.0,
        hide_zero_axis=True,
        color_scale=color_scale,
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
    sel_dates = st.multiselect("Date", all_dates, default=[all_dates[0]] if all_dates else [])

    df_visible = df_all[df_all["date"].isin(sel_dates)].reset_index(drop=True)

    auto_excl, excl_order = compute_auto_excludes(df_visible, n_excl_auto)
    n_auto = int(auto_excl.sum())

    df_edit = df_visible.copy()
    df_edit.insert(0, "Excl.", auto_excl.values)
    _ordered = ["Excl.", "target_mph", "test_type", "time_s", "run_number", "notes", "date"]
    _hidden = [c for c in df_edit.columns if c not in _ordered]
    df_edit = df_edit[_ordered + _hidden]

    # Sort: excluded rows first (in exclusion order), then included rows in original order
    df_edit["_excl_order"] = excl_order.values
    df_edit["_orig_order"] = range(len(df_edit))
    df_edit = df_edit.sort_values(
        ["Excl.", "_excl_order", "_orig_order"],
        ascending=[False, True, True],
    ).drop(columns=["_excl_order", "_orig_order"]).reset_index(drop=True)

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

tab_summary, tab_charts, tab_method = st.tabs(["Summary", "Charts", "Help"])

# ── Tab: Data Summary ─────────────────────────────────────────────────────────

with tab_summary:
    col_all, col_filt = st.columns(2)

    with col_all:
        st.subheader("All Data")
        st.plotly_chart(loss_line_chart(losses_all, "Accel / Decel Losses", show_fit=False),
                        use_container_width=True)
        ca_all, cd_all = fit_losses(losses_all)
        losses_table(losses_all, ca=ca_all, cd=cd_all)
        st.subheader("Run Times by Speed & Type")
        run_pivot_table(df_visible, df_ref=df_kept)

    with col_filt:
        st.subheader(filt_label)
        st.plotly_chart(loss_line_chart(losses_filt, "Accel / Decel Losses (filtered)"),
                        use_container_width=True)
        ca_filt, cd_filt = fit_losses(losses_filt)
        losses_table(losses_filt, ca=ca_filt, cd=cd_filt)
        st.subheader("Run Times by Speed & Type")
        run_pivot_table(df_kept, df_ref=df_visible)

# ── Tab: Charts ───────────────────────────────────────────────────────────────

with tab_charts:
    use_color_scale = st.toggle("Conditional formatting", value=True)

    col_charts_all, col_charts_filt = st.columns(2)

    _ts = datetime.now().strftime("%Y-%m-%d-%H%M")

    with col_charts_all:
        st.subheader("All Data")
        if accel_all is not None:
            st.download_button(
                "⬇ Export navigator charts (all data)",
                data=build_export_bytes(accel_all, decel_all, "all data",
                                        color_scale=use_color_scale),
                file_name=f"{_ts}_navigator_charts_all.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button("⬇ Export navigator charts (all data)", disabled=True,
                      use_container_width=True)
        render_matrix_html(accel_all, decel_all, color_scale=use_color_scale)

    with col_charts_filt:
        st.subheader(filt_label)
        if accel_filt is not None:
            st.download_button(
                "⬇ Export navigator charts (filtered)",
                data=build_export_bytes(accel_filt, decel_filt, "filtered",
                                        color_scale=use_color_scale),
                file_name=f"{_ts}_navigator_charts_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button("⬇ Export navigator charts (filtered)", disabled=True,
                      use_container_width=True)
        render_matrix_html(accel_filt, decel_filt, color_scale=use_color_scale)

# ── Tab: Methodology ──────────────────────────────────────────────────────────

with tab_method:
    st.markdown("""
## About This App

This app processes timed calibration runs to produce the **navigator charts**
used in-car during the Great Race. The charts give the navigator a lookup table showing
how many seconds are gained or lost whenever the car changes speed — for example,
accelerating from a stop sign to cruise speed, or slowing from 45 mph to make a turn.

---

## Intent

The goal is to build reference matrices that are as accurate as possible for *this car*
with *this driver and navigator*. Because every car accelerates and decelerates differently,
the charts must be derived from actual timed runs rather than theoretical calculations.
Three run types are collected:

- **Straight** — timed over a measured course at a constant target speed; establishes
  the baseline time for that speed.
- **Start** — standing start to the end of the same measured course at target speed;
  the excess over the straight time is the acceleration loss.
- **Stop** — run over the same measured course ending in a full stop; the excess over
  the straight time is the deceleration loss.

All three run types must be performed over the same course distance. The loss values
are pure time differences and are valid regardless of what that distance is.

Polynomial curve fits (degree 2) are applied across speeds so the matrices can be
populated at all speed combinations, including those not directly measured.

---

## How to Use

1. **Select calibration date(s)** in the sidebar. Typically use the most recent session.
2. **Set the auto-exclude count** with the slider. The algorithm automatically identifies
   and removes the runs that most degrade the curve fits. Six is a reasonable starting
   point — reduce it if data is sparse, increase it if outliers are visible in the charts.
3. **Review the sidebar table.** Excluded runs appear at the top in the order they were
   removed. Uncheck any row to manually override the exclusion.
4. **Check the Summary tab** to compare the all-data and filtered results side by side.
   The loss charts, fit equations, R² values, and run-time pivot table all update live.
5. **Open the Charts tab** to inspect the four reference matrices and export them to
   Excel for printing and in-car use.

---

## Interpreting the Results

- **R² values** near 1.0 indicate a smooth, reliable fit. Values below ~0.95 suggest
  the data is noisy or that more exclusions may be warranted.
- **Fit Accel Loss / Fit Decel Loss columns** in the Summary table show what the curve
  fit predicts at each speed. Large differences between the raw average and the fit
  value at the same speed indicate a noisy data point.
- **The filtered column is the one that feeds the navigator charts.** The all-data
  column is provided for comparison — to help judge whether exclusions are improving
  or distorting the result.
- **Highlighted N cells** in the pivot tables flag where the filtered run count differs
  from the full dataset, making it easy to see which speed/type combinations lost runs
  to exclusion.
- If the filtered and all-data fits are very similar, the exclusions had little effect
  and the data is consistent. If they diverge significantly, review the excluded runs
  carefully before trusting the charts.

---

## Auto-Exclude Methodology

Runs are excluded one at a time using a greedy search. At each step, every remaining
run from the latest date is trialled for removal and the one whose exclusion most reduces the total
sum of squared residuals from degree-2 polynomial fits to both the accel-loss and
decel-loss curves is permanently excluded. Ties are broken by choosing the run with
the highest z-score within its speed/type group (i.e. the most statistically extreme
run). The process repeats until the requested number of exclusions is reached.
Excluded runs are listed first in the sidebar calibration runs table, in the order
they were removed.
""")

