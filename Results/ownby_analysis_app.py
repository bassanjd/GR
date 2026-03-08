"""
Team Ownby – Great Race 2025 Performance Analysis
Car #123 · 1961 Mercedes-Benz 190 SL · Crew: Ownby/Wallace · Division: R
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Team Ownby Analysis", layout="wide", page_icon="🏁")

PARQUET = Path(__file__).parent / "long_format_times.parquet"
OWNBY_CAR = 123

# Stage route descriptions (from stage notes PDFs)
STAGE_LABELS = {
    0: "Trophy Run – St. Paul, MN",
    1: "St. Paul to Rochester, MN",
    2: "Stage 2",
    3: "Stage 3",
    4: "Stage 4",
    5: "Stage 5",
    6: "Stage 6",
    7: "Stage 7",
    8: "Stage 8",
    9: "Stage 9 – Final to Irmo, SC",
}

# ─── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_parquet(PARQUET)
    # Exclude discarded legs from scoring analysis
    return df[~df["Discarded"]].copy()


df = load_data()
ownby = df[df["CAR"] == OWNBY_CAR].copy()
r_div = df[df["DIV"] == "R"].copy()

# ─── Pre-compute stage summaries ───────────────────────────────────────────────

def stage_totals(data):
    return (
        data.groupby(["Stage", "CAR", "CREW"])
        .agg(
            total_penalty=("Time", "sum"),
            early_count=("Early", "sum"),
            legs=("Leg", "count"),
        )
        .reset_index()
    )


field_stage = stage_totals(df)
ownby_stage_totals = stage_totals(ownby)

# Add field rank for Ownby per stage
def add_field_rank(ownby_totals, field_totals):
    rows = []
    for _, row in ownby_totals.iterrows():
        stage = row["Stage"]
        others = field_totals[field_totals["Stage"] == stage]["total_penalty"]
        rank = int((others < row["total_penalty"]).sum()) + 1
        size = len(others)
        rows.append({**row.to_dict(), "rank": rank, "field_size": size,
                     "percentile": round((1 - rank / size) * 100, 1)})
    return pd.DataFrame(rows)


ownby_ranked = add_field_rank(ownby_stage_totals, field_stage)
ownby_ranked["early_pct"] = (ownby_ranked["early_count"] / ownby_ranked["legs"] * 100).round(0)
ownby_ranked["stage_label"] = ownby_ranked["Stage"].map(STAGE_LABELS)

# Ownby vs field median by (Stage, Leg)
field_leg_med = (
    df.groupby(["Stage", "Leg"])["Time"]
    .median()
    .reset_index()
    .rename(columns={"Time": "field_median"})
)
leg_compare = ownby[["Stage", "Leg", "Time", "Early"]].merge(field_leg_med, on=["Stage", "Leg"])
leg_compare["vs_median"] = leg_compare["Time"] - leg_compare["field_median"]
leg_compare["stage_leg"] = "S" + leg_compare["Stage"].astype(str) + "-L" + leg_compare["Leg"].astype(str)

# R division season standings
r_season = (
    r_div.groupby(["CAR", "CREW"])
    .agg(total_penalty=("Time", "sum"), early_count=("Early", "sum"), legs=("Leg", "count"))
    .reset_index()
    .sort_values("total_penalty")
    .reset_index(drop=True)
)
r_season.index += 1
r_season["early_pct"] = (r_season["early_count"] / r_season["legs"] * 100).round(1)

# ─── Page header ───────────────────────────────────────────────────────────────

st.title("Team Ownby — Great Race 2025 Performance Analysis")
st.caption(
    "Car #123 · 1961 Mercedes-Benz 190 SL · Crew: Ownby/Wallace · Division: R (Retrochallenge) · "
    "St. Paul, MN → Irmo, SC"
)

tabs = st.tabs([
    "Overview",
    "Stage Breakdown",
    "Leg Analysis",
    "Early/Late Bias",
    "Division Comparison",
    "Recommendations",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.header("Season Overview")

    total_penalty = ownby["Time"].sum()
    overall_early_pct = ownby["Early"].mean() * 100
    best = ownby_ranked.loc[ownby_ranked["rank"].idxmin()]
    worst = ownby_ranked.loc[ownby_ranked["rank"].idxmax()]
    n_stages = len(ownby_ranked)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Stages Competed", n_stages)
    k2.metric("Total Penalty", f"{total_penalty:.0f}s")
    k3.metric("Best Stage Rank", f"#{int(best['rank'])} / {int(best['field_size'])}",
              f"Stage {int(best['Stage'])}")
    k4.metric("Worst Stage Rank", f"#{int(worst['rank'])} / {int(worst['field_size'])}",
              f"Stage {int(worst['Stage'])}")
    k5.metric("Early Arrival Rate", f"{overall_early_pct:.0f}%", "of all legs")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Field Rank by Stage")
        fig_rank = px.line(
            ownby_ranked.sort_values("Stage"),
            x="Stage", y="rank", markers=True,
            title="Field Rank by Stage (lower = better)",
            labels={"rank": "Rank", "Stage": "Stage"},
        )
        fig_rank.update_yaxes(autorange="reversed")
        fig_rank.update_traces(line_color="#d62728", marker_size=10)
        fig_rank.update_layout(template="plotly_white")
        st.plotly_chart(fig_rank, use_container_width=True)

    with col_right:
        st.subheader("Total Penalty by Stage (seconds)")
        fig_pen = px.bar(
            ownby_ranked.sort_values("Stage"),
            x="Stage", y="total_penalty",
            title="Total Stage Penalty (s)",
            labels={"total_penalty": "Penalty (s)", "Stage": "Stage"},
            color="total_penalty",
            color_continuous_scale="Reds",
        )
        fig_pen.update_layout(template="plotly_white", coloraxis_showscale=False)
        st.plotly_chart(fig_pen, use_container_width=True)

    st.subheader("Stage-by-Stage Summary")
    disp = ownby_ranked[[
        "Stage", "stage_label", "total_penalty", "legs",
        "rank", "field_size", "percentile", "early_pct"
    ]].rename(columns={
        "stage_label": "Route",
        "total_penalty": "Penalty (s)",
        "legs": "Legs",
        "rank": "Field Rank",
        "field_size": "Field Size",
        "percentile": "Top %",
        "early_pct": "% Early",
    }).sort_values("Stage")
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – STAGE BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("Stage Breakdown")

    stage_options = sorted(ownby["Stage"].unique())
    stage_sel = st.selectbox(
        "Select Stage",
        stage_options,
        format_func=lambda s: f"Stage {s} — {STAGE_LABELS.get(s, '')}",
    )

    ownby_s = ownby[ownby["Stage"] == stage_sel]
    field_s = df[df["Stage"] == stage_sel]

    # Stage rank context
    stage_row = ownby_ranked[ownby_ranked["Stage"] == stage_sel].iloc[0]
    st.info(
        f"Stage {stage_sel}: Ownby ranked **#{int(stage_row['rank'])} of "
        f"{int(stage_row['field_size'])}** with **{stage_row['total_penalty']:.0f}s** total penalty. "
        f"Early on **{stage_row['early_pct']:.0f}%** of legs."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ownby Leg Penalties")
        fig_legs = px.bar(
            ownby_s.sort_values("Leg"),
            x="Leg", y="Time",
            color="Early",
            color_discrete_map={True: "#d62728", False: "#1f77b4"},
            title=f"Stage {stage_sel} — Penalty per Leg  [Red = Early]",
            labels={"Time": "Penalty (s)", "Leg": "Leg #", "Early": "Arrived Early"},
        )
        fig_legs.update_layout(template="plotly_white")
        st.plotly_chart(fig_legs, use_container_width=True)

    with col2:
        # Field distribution for Ownby's worst leg this stage
        if len(ownby_s) > 0:
            worst_leg_num = int(ownby_s.loc[ownby_s["Time"].idxmax(), "Leg"])
            field_leg_data = field_s[field_s["Leg"] == worst_leg_num]
            ownby_pen = float(ownby_s[ownby_s["Leg"] == worst_leg_num]["Time"].iloc[0])

            st.subheader(f"Field Distribution — Worst Leg ({worst_leg_num})")
            fig_dist = px.histogram(
                field_leg_data, x="Time", nbins=30,
                title=f"All competitors on Leg {worst_leg_num}",
                labels={"Time": "Penalty (s)"},
            )
            fig_dist.add_vline(
                x=ownby_pen, line_dash="dash", line_color="red",
                annotation_text=f"Ownby: {ownby_pen:.0f}s",
                annotation_position="top right",
            )
            fig_dist.update_layout(template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)

    # Full competitor table
    st.subheader(f"All Competitors — Stage {stage_sel}")
    stage_comp = (
        field_s.groupby(["CAR", "YEAR", "DIV", "CREW"])
        .agg(total_penalty=("Time", "sum"), early_count=("Early", "sum"), legs=("Leg", "count"))
        .reset_index()
        .sort_values("total_penalty")
        .reset_index(drop=True)
    )
    stage_comp.index += 1
    stage_comp["early_pct"] = (stage_comp["early_count"] / stage_comp["legs"] * 100).round(0)

    def highlight_ownby(row):
        color = "background-color: #ffe0e0" if row["CAR"] == OWNBY_CAR else ""
        return [color] * len(row)

    disp_comp = stage_comp[["CAR", "YEAR", "DIV", "CREW", "total_penalty", "early_pct"]].rename(
        columns={"total_penalty": "Penalty (s)", "early_pct": "% Early"}
    )
    st.dataframe(
        disp_comp.style.apply(highlight_ownby, axis=1),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – LEG ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Leg-Level Analysis")

    # Heatmap: Stage × Leg
    st.subheader("Penalty Heatmap — Stage × Leg (seconds)")
    heat_data = ownby.pivot_table(index="Stage", columns="Leg", values="Time", aggfunc="mean")
    fig_heat = px.imshow(
        heat_data,
        title="Ownby Penalty (s) — Stage × Leg",
        labels={"color": "Penalty (s)"},
        color_continuous_scale="Reds",
        text_auto=".0f",
    )
    fig_heat.update_layout(template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Ownby vs field median
    st.subheader("Ownby vs. Field Median per Leg")
    fig_vs = px.bar(
        leg_compare.sort_values(["Stage", "Leg"]),
        x="stage_leg",
        y="vs_median",
        color="Early",
        color_discrete_map={True: "#d62728", False: "#1f77b4"},
        title="Ownby Penalty vs. Field Median  (positive = worse than median, red = early)",
        labels={"vs_median": "Delta vs Median (s)", "stage_leg": "Stage-Leg", "Early": "Arrived Early"},
    )
    fig_vs.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_vs.update_layout(template="plotly_white")
    st.plotly_chart(fig_vs, use_container_width=True)

    # Worst legs table
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("10 Worst Legs vs. Field Median")
        worst10 = (
            leg_compare.sort_values("vs_median", ascending=False)
            .head(10)[["stage_leg", "Stage", "Leg", "Time", "field_median", "vs_median", "Early"]]
            .rename(columns={
                "stage_leg": "Stage-Leg",
                "Time": "Ownby (s)",
                "field_median": "Median (s)",
                "vs_median": "Delta (s)",
                "Early": "Early",
            })
        )
        st.dataframe(worst10, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("10 Best Legs vs. Field Median")
        best10 = (
            leg_compare.sort_values("vs_median", ascending=True)
            .head(10)[["stage_leg", "Stage", "Leg", "Time", "field_median", "vs_median", "Early"]]
            .rename(columns={
                "stage_leg": "Stage-Leg",
                "Time": "Ownby (s)",
                "field_median": "Median (s)",
                "vs_median": "Delta (s)",
                "Early": "Early",
            })
        )
        st.dataframe(best10, use_container_width=True, hide_index=True)

    # Leg number pattern
    st.subheader("Penalty by Leg Number (across all stages)")
    leg_avg = (
        ownby.groupby("Leg")
        .agg(avg_penalty=("Time", "mean"), early_rate=("Early", "mean"), count=("Leg", "count"))
        .reset_index()
    )
    field_leg_avg = (
        df.groupby("Leg")["Time"].mean().reset_index().rename(columns={"Time": "field_avg"})
    )
    leg_avg = leg_avg.merge(field_leg_avg, on="Leg")
    leg_avg_melt = leg_avg.melt(id_vars="Leg", value_vars=["avg_penalty", "field_avg"],
                                var_name="Who", value_name="Avg Penalty (s)")
    leg_avg_melt["Who"] = leg_avg_melt["Who"].map({"avg_penalty": "Ownby", "field_avg": "Field Avg"})
    fig_leg_num = px.bar(
        leg_avg_melt, x="Leg", y="Avg Penalty (s)", color="Who",
        barmode="group",
        color_discrete_map={"Ownby": "#d62728", "Field Avg": "#aec7e8"},
        title="Average Penalty by Leg Number — Ownby vs. Field",
    )
    fig_leg_num.update_layout(template="plotly_white")
    st.plotly_chart(fig_leg_num, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – EARLY/LATE BIAS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("Early vs. Late Arrival Analysis")

    st.info(
        "In time-distance rallying, arriving **early** means the car traveled a leg faster than "
        "the target speed. Arriving late means it was slower. Both are penalized. An 'Early' "
        "pattern across most legs signals a systematic fast-running bias — the car's indicated "
        "speed is reading *lower* than the actual ground speed."
    )

    early_count = int(ownby["Early"].sum())
    late_count = int((~ownby["Early"]).sum())
    total_legs = len(ownby)
    avg_pen_early = ownby.loc[ownby["Early"], "Time"].mean()
    avg_pen_late = ownby.loc[~ownby["Early"], "Time"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Early Legs", f"{early_count} / {total_legs}", f"{early_count/total_legs*100:.0f}%")
    k2.metric("Late/On-Time Legs", f"{late_count} / {total_legs}", f"{late_count/total_legs*100:.0f}%")
    k3.metric("Avg Penalty (Early legs)", f"{avg_pen_early:.1f}s")
    k4.metric("Avg Penalty (Late legs)", f"{avg_pen_late:.1f}s" if not np.isnan(avg_pen_late) else "N/A")

    col1, col2 = st.columns(2)

    with col1:
        labels = ["Early", "Late / On Time"]
        values = [early_count, late_count]
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            marker_colors=["#d62728", "#1f77b4"],
            hole=0.35,
        )])
        fig_pie.update_layout(title="All Legs: Early vs. Late/On Time", template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        stage_early = (
            ownby.groupby("Stage")
            .agg(early_legs=("Early", "sum"), total_legs_=("Early", "count"))
            .reset_index()
        )
        stage_early["early_pct"] = stage_early["early_legs"] / stage_early["total_legs_"] * 100
        fig_eb = px.bar(
            stage_early, x="Stage", y="early_pct",
            title="% of Legs Arriving Early — by Stage",
            labels={"early_pct": "% Early", "Stage": "Stage"},
            color="early_pct",
            color_continuous_scale="Reds",
        )
        fig_eb.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
        fig_eb.update_layout(template="plotly_white", coloraxis_showscale=False)
        st.plotly_chart(fig_eb, use_container_width=True)

    # Early-leg penalty distribution vs late
    st.subheader("Penalty Distribution: Early vs. Late Legs")
    fig_box = px.box(
        ownby, x="Early", y="Time",
        color="Early",
        color_discrete_map={True: "#d62728", False: "#1f77b4"},
        title="Penalty (s) Distribution by Arrival Type",
        labels={"Early": "Arrived Early", "Time": "Penalty (s)"},
        points="all",
    )
    fig_box.update_xaxes(ticktext=["Late / On Time", "Early"], tickvals=[False, True])
    fig_box.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # Where does Ownby sit in the field for early rate?
    st.subheader("Early Arrival Rate: Ownby vs. All Competitors")
    field_early_rates = (
        df.groupby(["CAR", "CREW"])
        .agg(early_rate=("Early", "mean"))
        .reset_index()
    )
    field_early_rates["early_pct"] = field_early_rates["early_rate"] * 100
    ownby_er = float(field_early_rates[field_early_rates["CAR"] == OWNBY_CAR]["early_pct"].iloc[0])

    fig_er = px.histogram(
        field_early_rates, x="early_pct", nbins=25,
        title="Distribution of Early Arrival Rate — All Competitors",
        labels={"early_pct": "% Legs Arriving Early"},
    )
    fig_er.add_vline(
        x=ownby_er, line_dash="dash", line_color="red",
        annotation_text=f"Ownby: {ownby_er:.0f}%",
        annotation_position="top right",
    )
    fig_er.update_layout(template="plotly_white")
    st.plotly_chart(fig_er, use_container_width=True)
    st.caption(
        f"Ownby's early rate of {ownby_er:.0f}% places them in the "
        f"{'upper' if ownby_er > field_early_rates['early_pct'].median() else 'lower'} "
        f"half of the field for early arrivals. Field median: "
        f"{field_early_rates['early_pct'].median():.0f}%."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – DIVISION COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.header("R Division Comparison")

    # R division stage totals
    r_stage_totals = stage_totals(r_div)
    r_stage_totals["r_rank"] = (
        r_stage_totals.groupby("Stage")["total_penalty"]
        .rank(method="min")
        .astype(int)
    )
    ownby_r_rank = r_stage_totals[r_stage_totals["CAR"] == OWNBY_CAR].copy()

    n_r = len(r_season)
    ownby_r_pos = int(r_season[r_season["CAR"] == OWNBY_CAR].index[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("R Division Competitors", n_r)
    col2.metric("Ownby Season Rank (R Div)", f"#{ownby_r_pos} of {n_r}")
    col3.metric("Gap to R Division Leader",
                f"{r_season.iloc[0]['total_penalty'] - r_season[r_season['CAR']==OWNBY_CAR].iloc[0]['total_penalty']:.0f}s",
                delta_color="inverse")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Ownby R Division Rank by Stage")
        fig_rr = px.line(
            ownby_r_rank.sort_values("Stage"),
            x="Stage", y="r_rank", markers=True,
            title="R Division Rank by Stage (lower = better)",
            labels={"r_rank": "R Division Rank", "Stage": "Stage"},
        )
        fig_rr.update_yaxes(autorange="reversed")
        fig_rr.update_traces(line_color="#d62728", marker_size=10)
        fig_rr.update_layout(template="plotly_white")
        st.plotly_chart(fig_rr, use_container_width=True)

    with col_r:
        st.subheader("R Division: Stage Penalty Comparison")
        r_stage_avg = (
            r_stage_totals.groupby("Stage")
            .agg(div_avg=("total_penalty", "mean"), div_min=("total_penalty", "min"))
            .reset_index()
        )
        ownby_stage_r = r_stage_totals[r_stage_totals["CAR"] == OWNBY_CAR][["Stage", "total_penalty"]].rename(
            columns={"total_penalty": "Ownby"}
        )
        r_comp = r_stage_avg.merge(ownby_stage_r, on="Stage")
        r_melt = r_comp.melt(id_vars="Stage", value_vars=["Ownby", "div_avg", "div_min"],
                             var_name="Series", value_name="Penalty (s)")
        r_melt["Series"] = r_melt["Series"].map({
            "Ownby": "Ownby", "div_avg": "R Div Avg", "div_min": "R Div Best"
        })
        fig_rc = px.line(
            r_melt, x="Stage", y="Penalty (s)", color="Series",
            markers=True,
            color_discrete_map={"Ownby": "#d62728", "R Div Avg": "#ff7f0e", "R Div Best": "#2ca02c"},
            title="Ownby vs. R Division (avg & best) per Stage",
        )
        fig_rc.update_layout(template="plotly_white")
        st.plotly_chart(fig_rc, use_container_width=True)

    st.subheader("R Division Season Standings")

    def highlight_ownby(row):
        color = "background-color: #ffe0e0" if row["CAR"] == OWNBY_CAR else ""
        return [color] * len(row)

    r_disp = r_season[["CAR", "CREW", "total_penalty", "early_pct"]].rename(
        columns={"total_penalty": "Total Penalty (s)", "CREW": "Crew", "early_pct": "% Early"}
    )
    st.dataframe(r_disp.style.apply(highlight_ownby, axis=1), use_container_width=True)

    st.subheader("Early Arrival Rate — R Division")
    fig_re = px.bar(
        r_season.sort_values("early_pct", ascending=False),
        x="CREW", y="early_pct",
        title="% Legs Arriving Early — R Division",
        labels={"early_pct": "% Early", "CREW": "Crew"},
        color="CAR",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_re.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_re, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 – RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.header("Areas to Focus On — Actionable Recommendations")

    # Compute dynamic stats for recommendations
    early_pct_all = ownby["Early"].mean() * 100
    avg_pen_all = ownby["Time"].mean()
    avg_pen_early_legs = ownby.loc[ownby["Early"], "Time"].mean()
    worst3 = leg_compare.sort_values("vs_median", ascending=False).head(3)
    best_stage_row = ownby_ranked.loc[ownby_ranked["rank"].idxmin()]
    consistent_early = list(ownby_ranked[ownby_ranked["early_pct"] >= 80]["Stage"].astype(int))
    penalty_std = ownby.groupby("Stage")["Time"].sum().std()

    # R div stats
    best_r_crew = r_season.iloc[0]["CREW"]
    best_r_penalty = r_season.iloc[0]["total_penalty"]
    ownby_r_penalty = r_season[r_season["CAR"] == OWNBY_CAR].iloc[0]["total_penalty"]
    r_gap = ownby_r_penalty - best_r_penalty
    ownby_r_pos_num = int(r_season[r_season["CAR"] == OWNBY_CAR].index[0])

    leg1_early = ownby[ownby["Leg"] == 1]["Early"].mean() * 100

    # ─── Priority 1 ──────────────────────────────────────────────────────────
    st.subheader("Priority 1 — Speedometer Calibration (Running Too Fast)")
    st.error(
        f"**{early_pct_all:.0f}%** of all legs, Ownby arrived at the checkpoint EARLY. "
        "This is not a random result — it's a systematic bias. The car is consistently traveling "
        "faster than the target speed, meaning the speedometer reads **lower than actual ground speed**.\n\n"
        "**What the navigator's own notes confirm:** The stage notes show accumulated corrections "
        "of -2.9s, -3.7s, -4.3s, -7s at interim waypoints throughout Stage 1. The navigator was "
        "constantly fighting an early-running car by calculating makeup adjustments.\n\n"
        "**The root cause:** When the car holds 50 MPH on the speedometer, it is actually doing "
        "~51–52 MPH over ground. This compounds over a 38-minute leg into 16+ seconds early.\n\n"
        "**Actions:**\n"
        "- During the daily calibration run, dial the speedometer *up* — if 1 mile takes 1m12s at "
        "indicated 50 MPH, the corrected factor should be adjusted to reflect actual travel time.\n"
        "- As an immediate on-course fix: target **1–2 MPH below** every posted speed throughout "
        "each leg to absorb the systematic overspeed.\n"
        "- The navigator should track cumulative error starting from zero at each leg start, "
        "and give small speed corrections (e.g. 'slow down 5 seconds over next 2 miles') "
        "rather than waiting until a waypoint is already past-early."
    )

    if consistent_early:
        st.warning(
            f"Stages where Ownby was early on 80%+ of legs: **{consistent_early}** — "
            "the calibration problem is most acute on these days."
        )

    # ─── Priority 2 ──────────────────────────────────────────────────────────
    st.subheader("Priority 2 — Worst Specific Legs")
    for _, r in worst3.iterrows():
        direction = "early" if r["Early"] else "late"
        st.write(
            f"- **Stage {int(r['Stage'])}, Leg {int(r['Leg'])}** ({r['stage_leg']}): "
            f"Ownby {r['Time']:.0f}s · Field median {r['field_median']:.0f}s · "
            f"**+{r['vs_median']:.0f}s above median** — arrived {direction}"
        )
    st.info(
        "Review the stage notes PDF pages for these specific legs. Look for:\n"
        "- Speed limit drops that were missed or delayed\n"
        "- Stop signs / traffic light stops that were shorter than expected\n"
        "- Highway sections where the car accelerated above target without noticing\n\n"
        "The navigator's handwritten corrections on those pages will pinpoint exactly "
        "where in the leg the time was gained."
    )

    # ─── Priority 3 ──────────────────────────────────────────────────────────
    st.subheader("Priority 3 — Leg 1 Start Discipline")
    st.write(
        f"On first legs (Leg 1) across all stages, Ownby arrived early **{leg1_early:.0f}%** "
        "of the time. The CDT countdown start is a high-stress moment where it's easy to "
        "over-accelerate immediately after the clock starts.\n\n"
        "**Action:** After pressing the CDT start button, hold the car at target speed for a "
        "full 30 seconds before making any adjustments. Resist the urge to accelerate to "
        "'make up' any perceived gap from the start."
    )

    # ─── Priority 4 ──────────────────────────────────────────────────────────
    st.subheader("Priority 4 — Replicate the Best Stage")
    st.write(
        f"Best stage was **Stage {int(best_stage_row['Stage'])}** with rank "
        f"**#{int(best_stage_row['rank'])} of {int(best_stage_row['field_size'])}** "
        f"and only {best_stage_row['total_penalty']:.0f}s total penalty. "
        f"The navigator's notes from that stage are the most valuable reference material "
        f"for what the crew did correctly — particularly around speed maintenance and "
        f"correction technique."
    )

    if not np.isnan(penalty_std):
        st.write(
            f"Stage-to-stage penalty standard deviation: **{penalty_std:.1f}s** — "
            "this is the consistency gap between best and worst days."
        )

    # ─── Priority 5 ──────────────────────────────────────────────────────────
    st.subheader("Priority 5 — R Division Competitive Gap")
    st.write(
        f"Ownby is currently **#{ownby_r_pos_num} of {n_r}** in R division. "
        f"Division leader ({best_r_crew}) has accumulated **{best_r_penalty:.0f}s** total penalty "
        f"vs. Ownby's **{ownby_r_penalty:.0f}s** — a gap of **{r_gap:.0f}s** over the full race.\n\n"
        "Most of that gap is recoverable from the calibration fix alone. "
        f"If Ownby reduced their average leg penalty from "
        f"**{avg_pen_all:.1f}s to ~{max(avg_pen_all*0.5, 2):.0f}s** per leg, "
        f"the total savings across {total_legs} legs would be approximately "
        f"**{(avg_pen_all - max(avg_pen_all*0.5, 2)) * total_legs:.0f}s**."
    )

    # ─── Summary table ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Priority Summary")
    st.markdown("""
| Priority | Focus Area | Root Cause | Expected Impact |
|---|---|---|---|
| 1 | Speedometer calibration — running consistently too fast | Speedometer reads low vs. ground speed | High — affects every single leg |
| 2 | Worst specific legs (see Leg Analysis tab) | Missed speed changes / stop timing | Medium — targeted 10–20s savings |
| 3 | Leg 1 start discipline — over-acceleration at CDT start | Pressure at stage start | Medium — first leg sets tone |
| 4 | Replicate best-stage technique | Inconsistent correction method | Low-medium — reduce variance |
| 5 | Close R division gap | Compound of above | High if calibration fixed |
""")
