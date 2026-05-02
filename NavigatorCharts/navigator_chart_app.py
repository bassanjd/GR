"""
Interactive Navigator Charts analysis.

Three-column layout — no tabs:
  Left   — raw calibration runs table with per-row Exclude checkboxes
  Middle — charts from all visible data (no row exclusions)
  Right  — charts recomputed after excluded rows are removed

Export buttons produce in-memory Excel files containing the four
reference matrices (same format as make_navigator_charts.py).
"""
import io

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from navigator_chart_helpers import (
    build_reference_workbook,
    compute_losses,
    load_calibration_runs,
    losses_to_dicts,
)

st.set_page_config(page_title="Navigator Charts", layout="wide", page_icon="🏎️")

DATE_2026 = "2026-04-29"


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_runs():
    return load_calibration_runs()


def build_export_bytes(accel, decel, label):
    """Return bytes of an in-memory Excel workbook with 4 reference matrices."""
    wb = build_reference_workbook(accel, decel, label)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ── Chart builders ────────────────────────────────────────────────────────────

TYPE_COLORS = {
    "straight_speed": "#1976D2",
    "start_speed":    "#388E3C",
    "speed_stop":     "#D32F2F",
}


def time_strip_chart(df, title):
    if df.empty:
        return go.Figure().update_layout(title=title, height=320)
    fig = px.strip(
        df, x="Target MPH", y="Time (s)", color="Test Type",
        title=title, stripmode="overlay",
        hover_data=["Date", "Run #", "Direction"],
        color_discrete_map=TYPE_COLORS,
    )
    fig.update_traces(marker_size=7, opacity=0.8)
    fig.update_layout(
        height=320, margin=dict(t=35, b=5, l=5, r=5),
        legend=dict(orientation="h", y=-0.18, title_text=""),
    )
    return fig


def loss_line_chart(losses, title):
    if losses is None:
        return go.Figure().update_layout(title=title, height=260)
    fig = go.Figure([
        go.Scatter(x=losses["MPH"], y=losses["Accel Loss (s)"],
                   name="Accel Loss", mode="lines+markers",
                   line=dict(color="#388E3C", width=2), marker_size=7),
        go.Scatter(x=losses["MPH"], y=losses["Decel Loss (s)"],
                   name="Decel Loss", mode="lines+markers",
                   line=dict(color="#D32F2F", width=2), marker_size=7),
    ])
    fig.update_layout(
        title=title, height=260,
        xaxis_title="Speed (mph)", yaxis_title="Time Lost (s)",
        margin=dict(t=35, b=5, l=5, r=5),
        legend=dict(orientation="h", y=-0.2, title_text=""),
    )
    return fig



def losses_table(losses):
    if losses is None:
        st.caption("Insufficient 2026 data to compute losses.")
        return
    styled = losses.style.format({
        "Straight (s)":  "{:.2f}",
        "Accel Loss (s)":"{:.2f}",
        "Decel Loss (s)":"{:.2f}",
        "Actual MPH":    "{:.2f}",
        "Error (%)":     "{:.2f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ── State init ────────────────────────────────────────────────────────────────

df_all = load_runs()


# ── Main layout ───────────────────────────────────────────────────────────────

st.title("Navigator Charts")

col_data, col_all, col_filt = st.columns([3, 2.5, 2.5])

# ── Left: editable data table + filters ──────────────────────────────────────
with col_data:
    st.subheader("Input Data")
    st.caption("Check Excl. to remove a run from the Filtered column.")

    all_dates = sorted(df_all["Date"].unique(), reverse=True)
    sel_dates = st.multiselect("Date", all_dates, default=[DATE_2026])
    all_types = sorted(df_all["Test Type"].unique())
    sel_types = st.multiselect("Test Type", all_types, default=all_types)

    df_visible = df_all[
        df_all["Date"].isin(sel_dates) & df_all["Test Type"].isin(sel_types)
    ].reset_index(drop=True)

    df_edit = df_visible.copy()
    df_edit.insert(0, "Excl.", False)

    edited = st.data_editor(
        df_edit,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Excl.":      st.column_config.CheckboxColumn("Excl.", width="small"),
            "Date":       st.column_config.TextColumn("Date",     width="small"),
            "Test Type":  st.column_config.TextColumn("Type",     width="medium"),
            "Target MPH": st.column_config.NumberColumn("MPH",    width="small"),
            "Run #":      st.column_config.NumberColumn("#",      width="small"),
            "Direction":  st.column_config.TextColumn("Dir",      width="small"),
            "Time (raw)": st.column_config.TextColumn("Raw",      width="small"),
            "Time (s)":   st.column_config.NumberColumn("Sec",    format="%.2f", width="small"),
            "Notes":      st.column_config.TextColumn("Notes",    width="small"),
        },
        height=600,
        disabled=[c for c in df_edit.columns if c != "Excl."],
    )

n_excl = int(edited["Excl."].sum())
df_kept = edited[~edited["Excl."]].drop(columns=["Excl."])

# ── Middle: all data ──────────────────────────────────────────────────────────
with col_all:
    st.subheader("All Data")

    losses_all = compute_losses(df_visible)
    accel_all, decel_all = losses_to_dicts(losses_all)

    if accel_all is not None:
        xlsx_all = build_export_bytes(accel_all, decel_all, "all data")
        st.download_button(
            "⬇ Export reference charts (all data)",
            data=xlsx_all,
            file_name="reference_charts_all.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.button("⬇ Export reference charts (all data)", disabled=True,
                  use_container_width=True)

    st.plotly_chart(time_strip_chart(df_visible, "Run Times"), use_container_width=True)
    st.plotly_chart(loss_line_chart(losses_all, "Accel / Decel Losses"), use_container_width=True)
    losses_table(losses_all)

# ── Right: filtered data ──────────────────────────────────────────────────────
with col_filt:
    st.subheader(f"Filtered ({n_excl} excluded)" if n_excl else "Filtered (none excluded)")

    losses_filt = compute_losses(df_kept)
    accel_filt, decel_filt = losses_to_dicts(losses_filt)

    if accel_filt is not None:
        xlsx_filt = build_export_bytes(accel_filt, decel_filt, "filtered")
        st.download_button(
            "⬇ Export reference charts (filtered)",
            data=xlsx_filt,
            file_name="reference_charts_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.button("⬇ Export reference charts (filtered)", disabled=True,
                  use_container_width=True)

    st.plotly_chart(time_strip_chart(df_kept, "Run Times (filtered)"),
                    use_container_width=True)
    st.plotly_chart(loss_line_chart(losses_filt, "Accel / Decel Losses (filtered)"),
                    use_container_width=True)
    losses_table(losses_filt)
