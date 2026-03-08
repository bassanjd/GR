"""
Great Race 2025 — Team Performance Analysis
Select any team from the sidebar to explore their stage results, leg-level penalties,
early/late bias, and division comparison.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Great Race 2025 — Team Analysis", layout="wide", page_icon="🏁")

PARQUET = Path(__file__).parent / "long_format_times.parquet"

# Stage route descriptions
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
    return df[~df["Discarded"]].copy()


df = load_data()

# ─── Team selector ─────────────────────────────────────────────────────────────

# Build a sorted list of (CAR, label) tuples
car_info = (
    df.drop_duplicates("CAR")[["CAR", "YEAR", "DIV", "CREW"]]
    .sort_values("CAR")
)
car_options = car_info["CAR"].tolist()
car_labels = {
    row["CAR"]: f"#{row['CAR']} — {row['CREW']} ({row['YEAR']}, Div {row['DIV']})"
    for _, row in car_info.iterrows()
}

with st.sidebar:
    st.header("Team Selection")
    selected_car = st.selectbox(
        "Select team",
        car_options,
        format_func=lambda c: car_labels[c],
    )

# ─── Derive team metadata ───────────────────────────────────────────────────────

team_row = car_info[car_info["CAR"] == selected_car].iloc[0]
team_crew = team_row["CREW"]
team_div = team_row["DIV"]
team_year = team_row["YEAR"]

team_df = df[df["CAR"] == selected_car].copy()
div_df = df[df["DIV"] == team_div].copy()

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
team_stage_totals = stage_totals(team_df)


def add_field_rank(team_totals, field_totals):
    rows = []
    for _, row in team_totals.iterrows():
        stage = row["Stage"]
        others = field_totals[field_totals["Stage"] == stage]["total_penalty"]
        rank = int((others < row["total_penalty"]).sum()) + 1
        size = len(others)
        rows.append({**row.to_dict(), "rank": rank, "field_size": size,
                     "percentile": round((1 - rank / size) * 100, 1)})
    return pd.DataFrame(rows)


team_ranked = add_field_rank(team_stage_totals, field_stage)
team_ranked["early_pct"] = (team_ranked["early_count"] / team_ranked["legs"] * 100).round(0)
team_ranked["stage_label"] = team_ranked["Stage"].map(STAGE_LABELS)

# Team vs field median by (Stage, Leg)
field_leg_med = (
    df.groupby(["Stage", "Leg"])["Time"]
    .median()
    .reset_index()
    .rename(columns={"Time": "field_median"})
)
leg_compare = team_df[["Stage", "Leg", "Time", "Early"]].merge(field_leg_med, on=["Stage", "Leg"])
leg_compare["vs_median"] = leg_compare["Time"] - leg_compare["field_median"]
leg_compare["stage_leg"] = "S" + leg_compare["Stage"].astype(str) + "-L" + leg_compare["Leg"].astype(str)

# Division season standings
div_season = (
    div_df.groupby(["CAR", "CREW"])
    .agg(total_penalty=("Time", "sum"), early_count=("Early", "sum"), legs=("Leg", "count"))
    .reset_index()
    .sort_values("total_penalty")
    .reset_index(drop=True)
)
div_season.index += 1
div_season["early_pct"] = (div_season["early_count"] / div_season["legs"] * 100).round(1)

# ─── Page header ───────────────────────────────────────────────────────────────

st.title(f"Great Race 2025 — Team Performance Analysis")
st.caption(
    f"Car #{selected_car} · {team_year} · Crew: {team_crew} · Division: {team_div} · "
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

    total_penalty = team_df["Time"].sum()
    overall_early_pct = team_df["Early"].mean() * 100
    best = team_ranked.loc[team_ranked["rank"].idxmin()]
    worst = team_ranked.loc[team_ranked["rank"].idxmax()]
    n_stages = len(team_ranked)

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
            team_ranked.sort_values("Stage"),
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
            team_ranked.sort_values("Stage"),
            x="Stage", y="total_penalty",
            title="Total Stage Penalty (s)",
            labels={"total_penalty": "Penalty (s)", "Stage": "Stage"},
            color="total_penalty",
            color_continuous_scale="Reds",
        )
        fig_pen.update_layout(template="plotly_white", coloraxis_showscale=False)
        st.plotly_chart(fig_pen, use_container_width=True)

    st.subheader("Stage-by-Stage Summary")
    disp = team_ranked[[
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

    stage_options = sorted(team_df["Stage"].unique())
    stage_sel = st.selectbox(
        "Select Stage",
        stage_options,
        format_func=lambda s: f"Stage {s} — {STAGE_LABELS.get(s, '')}",
    )

    team_s = team_df[team_df["Stage"] == stage_sel]
    field_s = df[df["Stage"] == stage_sel]

    stage_row = team_ranked[team_ranked["Stage"] == stage_sel].iloc[0]
    st.info(
        f"Stage {stage_sel}: {team_crew} ranked **#{int(stage_row['rank'])} of "
        f"{int(stage_row['field_size'])}** with **{stage_row['total_penalty']:.0f}s** total penalty. "
        f"Early on **{stage_row['early_pct']:.0f}%** of legs."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{team_crew} Leg Penalties")
        fig_legs = px.bar(
            team_s.sort_values("Leg"),
            x="Leg", y="Time",
            color="Early",
            color_discrete_map={True: "#d62728", False: "#1f77b4"},
            title=f"Stage {stage_sel} — Penalty per Leg  [Red = Early]",
            labels={"Time": "Penalty (s)", "Leg": "Leg #", "Early": "Arrived Early"},
        )
        fig_legs.update_layout(template="plotly_white")
        st.plotly_chart(fig_legs, use_container_width=True)

    with col2:
        if len(team_s) > 0:
            worst_leg_num = int(team_s.loc[team_s["Time"].idxmax(), "Leg"])
            field_leg_data = field_s[field_s["Leg"] == worst_leg_num]
            team_pen = float(team_s[team_s["Leg"] == worst_leg_num]["Time"].iloc[0])

            st.subheader(f"Field Distribution — Worst Leg ({worst_leg_num})")
            fig_dist = px.histogram(
                field_leg_data, x="Time", nbins=30,
                title=f"All competitors on Leg {worst_leg_num}",
                labels={"Time": "Penalty (s)"},
            )
            fig_dist.add_vline(
                x=team_pen, line_dash="dash", line_color="red",
                annotation_text=f"{team_crew}: {team_pen:.0f}s",
                annotation_position="top right",
            )
            fig_dist.update_layout(template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)

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

    def highlight_team(row):
        color = "background-color: #ffe0e0" if row["CAR"] == selected_car else ""
        return [color] * len(row)

    disp_comp = stage_comp[["CAR", "YEAR", "DIV", "CREW", "total_penalty", "early_pct"]].rename(
        columns={"total_penalty": "Penalty (s)", "early_pct": "% Early"}
    )
    st.dataframe(
        disp_comp.style.apply(highlight_team, axis=1),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – LEG ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Leg-Level Analysis")

    st.subheader(f"Penalty Heatmap — Stage × Leg (seconds)")
    heat_data = team_df.pivot_table(index="Stage", columns="Leg", values="Time", aggfunc="mean")
    fig_heat = px.imshow(
        heat_data,
        title=f"{team_crew} Penalty (s) — Stage × Leg",
        labels={"color": "Penalty (s)"},
        color_continuous_scale="Reds",
        text_auto=".0f",
    )
    fig_heat.update_layout(template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader(f"{team_crew} vs. Field Median per Leg")
    fig_vs = px.bar(
        leg_compare.sort_values(["Stage", "Leg"]),
        x="stage_leg",
        y="vs_median",
        color="Early",
        color_discrete_map={True: "#d62728", False: "#1f77b4"},
        title=f"{team_crew} Penalty vs. Field Median  (positive = worse than median, red = early)",
        labels={"vs_median": "Delta vs Median (s)", "stage_leg": "Stage-Leg", "Early": "Arrived Early"},
    )
    fig_vs.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_vs.update_layout(template="plotly_white")
    st.plotly_chart(fig_vs, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("10 Worst Legs vs. Field Median")
        worst10 = (
            leg_compare.sort_values("vs_median", ascending=False)
            .head(10)[["stage_leg", "Stage", "Leg", "Time", "field_median", "vs_median", "Early"]]
            .rename(columns={
                "stage_leg": "Stage-Leg",
                "Time": "Team (s)",
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
                "Time": "Team (s)",
                "field_median": "Median (s)",
                "vs_median": "Delta (s)",
                "Early": "Early",
            })
        )
        st.dataframe(best10, use_container_width=True, hide_index=True)

    st.subheader("Penalty by Leg Number (across all stages)")
    leg_avg = (
        team_df.groupby("Leg")
        .agg(avg_penalty=("Time", "mean"), early_rate=("Early", "mean"), count=("Leg", "count"))
        .reset_index()
    )
    field_leg_avg = (
        df.groupby("Leg")["Time"].mean().reset_index().rename(columns={"Time": "field_avg"})
    )
    leg_avg = leg_avg.merge(field_leg_avg, on="Leg")
    leg_avg_melt = leg_avg.melt(id_vars="Leg", value_vars=["avg_penalty", "field_avg"],
                                var_name="Who", value_name="Avg Penalty (s)")
    leg_avg_melt["Who"] = leg_avg_melt["Who"].map({"avg_penalty": team_crew, "field_avg": "Field Avg"})
    fig_leg_num = px.bar(
        leg_avg_melt, x="Leg", y="Avg Penalty (s)", color="Who",
        barmode="group",
        color_discrete_map={team_crew: "#d62728", "Field Avg": "#aec7e8"},
        title=f"Average Penalty by Leg Number — {team_crew} vs. Field",
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

    early_count = int(team_df["Early"].sum())
    late_count = int((~team_df["Early"]).sum())
    total_legs = len(team_df)
    avg_pen_early = team_df.loc[team_df["Early"], "Time"].mean()
    avg_pen_late = team_df.loc[~team_df["Early"], "Time"].mean()

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
            team_df.groupby("Stage")
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

    st.subheader("Penalty Distribution: Early vs. Late Legs")
    fig_box = px.box(
        team_df, x="Early", y="Time",
        color="Early",
        color_discrete_map={True: "#d62728", False: "#1f77b4"},
        title="Penalty (s) Distribution by Arrival Type",
        labels={"Early": "Arrived Early", "Time": "Penalty (s)"},
        points="all",
    )
    fig_box.update_xaxes(ticktext=["Late / On Time", "Early"], tickvals=[False, True])
    fig_box.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader(f"Early Arrival Rate: {team_crew} vs. All Competitors")
    field_early_rates = (
        df.groupby(["CAR", "CREW"])
        .agg(early_rate=("Early", "mean"))
        .reset_index()
    )
    field_early_rates["early_pct"] = field_early_rates["early_rate"] * 100
    team_er = float(field_early_rates[field_early_rates["CAR"] == selected_car]["early_pct"].iloc[0])

    fig_er = px.histogram(
        field_early_rates, x="early_pct", nbins=25,
        title="Distribution of Early Arrival Rate — All Competitors",
        labels={"early_pct": "% Legs Arriving Early"},
    )
    fig_er.add_vline(
        x=team_er, line_dash="dash", line_color="red",
        annotation_text=f"{team_crew}: {team_er:.0f}%",
        annotation_position="top right",
    )
    fig_er.update_layout(template="plotly_white")
    st.plotly_chart(fig_er, use_container_width=True)
    st.caption(
        f"{team_crew}'s early rate of {team_er:.0f}% places them in the "
        f"{'upper' if team_er > field_early_rates['early_pct'].median() else 'lower'} "
        f"half of the field for early arrivals. Field median: "
        f"{field_early_rates['early_pct'].median():.0f}%."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – DIVISION COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.header(f"Division {team_div} Comparison")

    div_stage_totals = stage_totals(div_df)
    div_stage_totals["div_rank"] = (
        div_stage_totals.groupby("Stage")["total_penalty"]
        .rank(method="min")
        .astype(int)
    )
    team_div_rank = div_stage_totals[div_stage_totals["CAR"] == selected_car].copy()

    n_div = len(div_season)
    team_div_pos = int(div_season[div_season["CAR"] == selected_car].index[0])

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Div {team_div} Competitors", n_div)
    col2.metric(f"{team_crew} Season Rank (Div {team_div})", f"#{team_div_pos} of {n_div}")

    leader_row = div_season.iloc[0]
    team_div_penalty = div_season[div_season["CAR"] == selected_car].iloc[0]["total_penalty"]
    gap = team_div_penalty - leader_row["total_penalty"]
    col3.metric(
        f"Gap to Div {team_div} Leader",
        f"{gap:.0f}s",
        delta_color="inverse",
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(f"{team_crew} Div {team_div} Rank by Stage")
        fig_rr = px.line(
            team_div_rank.sort_values("Stage"),
            x="Stage", y="div_rank", markers=True,
            title=f"Div {team_div} Rank by Stage (lower = better)",
            labels={"div_rank": f"Div {team_div} Rank", "Stage": "Stage"},
        )
        fig_rr.update_yaxes(autorange="reversed")
        fig_rr.update_traces(line_color="#d62728", marker_size=10)
        fig_rr.update_layout(template="plotly_white")
        st.plotly_chart(fig_rr, use_container_width=True)

    with col_r:
        st.subheader(f"Div {team_div}: Stage Penalty Comparison")
        div_stage_avg = (
            div_stage_totals.groupby("Stage")
            .agg(div_avg=("total_penalty", "mean"), div_min=("total_penalty", "min"))
            .reset_index()
        )
        team_stage_r = div_stage_totals[div_stage_totals["CAR"] == selected_car][["Stage", "total_penalty"]].rename(
            columns={"total_penalty": team_crew}
        )
        r_comp = div_stage_avg.merge(team_stage_r, on="Stage")
        r_melt = r_comp.melt(id_vars="Stage", value_vars=[team_crew, "div_avg", "div_min"],
                             var_name="Series", value_name="Penalty (s)")
        r_melt["Series"] = r_melt["Series"].map({
            team_crew: team_crew,
            "div_avg": f"Div {team_div} Avg",
            "div_min": f"Div {team_div} Best",
        })
        fig_rc = px.line(
            r_melt, x="Stage", y="Penalty (s)", color="Series",
            markers=True,
            color_discrete_map={
                team_crew: "#d62728",
                f"Div {team_div} Avg": "#ff7f0e",
                f"Div {team_div} Best": "#2ca02c",
            },
            title=f"{team_crew} vs. Div {team_div} (avg & best) per Stage",
        )
        fig_rc.update_layout(template="plotly_white")
        st.plotly_chart(fig_rc, use_container_width=True)

    st.subheader(f"Division {team_div} Season Standings")

    def highlight_team_div(row):
        color = "background-color: #ffe0e0" if row["CAR"] == selected_car else ""
        return [color] * len(row)

    div_disp = div_season[["CAR", "CREW", "total_penalty", "early_pct"]].rename(
        columns={"total_penalty": "Total Penalty (s)", "CREW": "Crew", "early_pct": "% Early"}
    )
    st.dataframe(div_disp.style.apply(highlight_team_div, axis=1), use_container_width=True)

    st.subheader(f"Early Arrival Rate — Division {team_div}")
    fig_re = px.bar(
        div_season.sort_values("early_pct", ascending=False),
        x="CREW", y="early_pct",
        title=f"% Legs Arriving Early — Division {team_div}",
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
    st.header(f"Areas to Focus On — {team_crew}")

    early_pct_all = team_df["Early"].mean() * 100
    avg_pen_all = team_df["Time"].mean()
    avg_pen_early_legs = team_df.loc[team_df["Early"], "Time"].mean()
    worst3 = leg_compare.sort_values("vs_median", ascending=False).head(3)
    best_stage_row = team_ranked.loc[team_ranked["rank"].idxmin()]
    consistent_early_stages = list(team_ranked[team_ranked["early_pct"] >= 80]["Stage"].astype(int))
    penalty_std = team_df.groupby("Stage")["Time"].sum().std()

    best_div_crew = leader_row["CREW"]
    best_div_penalty = leader_row["total_penalty"]
    r_gap = team_div_penalty - best_div_penalty
    leg1_early = team_df[team_df["Leg"] == 1]["Early"].mean() * 100

    # ─── Priority 1 — Speed/Timing Bias ──────────────────────────────────────
    st.subheader("Priority 1 — Speed/Timing Bias")
    if early_pct_all >= 60:
        st.error(
            f"**{early_pct_all:.0f}%** of all legs, {team_crew} arrived at the checkpoint EARLY. "
            "This is a systematic fast-running bias — the car is consistently traveling faster than "
            "the target speed, suggesting the speedometer reads **lower than actual ground speed**.\n\n"
            "**Actions:**\n"
            "- During daily calibration, dial the speedometer correction factor upward.\n"
            "- As an on-course fix: target **1–2 MPH below** every posted speed to absorb the overspeed.\n"
            "- Navigator should track cumulative error from leg start and issue small speed corrections "
            "proactively rather than after reaching a waypoint early."
        )
        if consistent_early_stages:
            st.warning(
                f"Stages where {team_crew} was early on 80%+ of legs: **{consistent_early_stages}** — "
                "calibration correction is most critical on these days."
            )
    elif early_pct_all <= 40:
        st.warning(
            f"**{100 - early_pct_all:.0f}%** of all legs, {team_crew} arrived LATE. "
            "This suggests the car may be running slower than target speed — the speedometer "
            "could be reading higher than actual ground speed, or the crew is intentionally "
            "conservative.\n\n"
            "**Actions:**\n"
            "- Check speedometer calibration: if indicated speed is high vs. actual, "
            "increase target MPH slightly.\n"
            "- Review late legs to identify whether the cause is traffic, stops, or pacing."
        )
    else:
        st.success(
            f"{team_crew} has a reasonably balanced arrival pattern ({early_pct_all:.0f}% early). "
            "Focus on reducing peak penalties rather than correcting a directional bias."
        )

    # ─── Priority 2 — Worst Specific Legs ────────────────────────────────────
    st.subheader("Priority 2 — Worst Specific Legs")
    for _, r in worst3.iterrows():
        direction = "early" if r["Early"] else "late"
        st.write(
            f"- **Stage {int(r['Stage'])}, Leg {int(r['Leg'])}** ({r['stage_leg']}): "
            f"{team_crew} {r['Time']:.0f}s · Field median {r['field_median']:.0f}s · "
            f"**+{r['vs_median']:.0f}s above median** — arrived {direction}"
        )
    st.info(
        "Review the stage notes PDF pages for these specific legs. Look for:\n"
        "- Speed limit drops that were missed or delayed\n"
        "- Stop signs / traffic light stops that were shorter or longer than expected\n"
        "- Highway sections where speed deviated from target without correction\n\n"
        "Handwritten navigator corrections on those pages will pinpoint where in the leg "
        "the time was gained or lost."
    )

    # ─── Priority 3 — Leg 1 Start Discipline ─────────────────────────────────
    st.subheader("Priority 3 — Leg 1 Start Discipline")
    st.write(
        f"On first legs (Leg 1) across all stages, {team_crew} arrived early **{leg1_early:.0f}%** "
        "of the time. The CDT countdown start is a high-stress moment where it's easy to "
        "over- or under-accelerate immediately after the clock starts.\n\n"
        "**Action:** After pressing the CDT start button, hold the car at target speed for a "
        "full 30 seconds before making any adjustments."
    )

    # ─── Priority 4 — Replicate Best Stage ───────────────────────────────────
    st.subheader("Priority 4 — Replicate the Best Stage")
    st.write(
        f"Best stage was **Stage {int(best_stage_row['Stage'])}** — ranked "
        f"**#{int(best_stage_row['rank'])} of {int(best_stage_row['field_size'])}** "
        f"with only {best_stage_row['total_penalty']:.0f}s total penalty. "
        f"The navigator's notes from that stage are the most valuable reference material "
        f"for what the crew did correctly."
    )
    if not np.isnan(penalty_std):
        st.write(
            f"Stage-to-stage penalty standard deviation: **{penalty_std:.1f}s** — "
            "this is the consistency gap between best and worst days."
        )

    # ─── Priority 5 — Division Competitive Gap ───────────────────────────────
    st.subheader(f"Priority 5 — Division {team_div} Competitive Gap")
    st.write(
        f"{team_crew} is currently **#{team_div_pos} of {n_div}** in Division {team_div}. "
        f"Division leader ({best_div_crew}) has accumulated **{best_div_penalty:.0f}s** total penalty "
        f"vs. {team_crew}'s **{team_div_penalty:.0f}s** — a gap of **{r_gap:.0f}s** over the full race.\n\n"
        f"If {team_crew} reduced their average leg penalty from "
        f"**{avg_pen_all:.1f}s to ~{max(avg_pen_all * 0.5, 2):.0f}s** per leg, "
        f"the total savings across {total_legs} legs would be approximately "
        f"**{(avg_pen_all - max(avg_pen_all * 0.5, 2)) * total_legs:.0f}s**."
    )

    # ─── Summary table ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Priority Summary")
    bias_type = "Running consistently too fast" if early_pct_all >= 60 else (
        "Running consistently too slow" if early_pct_all <= 40 else "Balanced — reduce peak penalties"
    )
    st.markdown(f"""
| Priority | Focus Area | Root Cause | Expected Impact |
|---|---|---|---|
| 1 | Speed/timing bias | {bias_type} | High — affects every single leg |
| 2 | Worst specific legs (see Leg Analysis tab) | Missed speed changes / stop timing | Medium — targeted savings |
| 3 | Leg 1 start discipline | Pressure at stage start | Medium — first leg sets tone |
| 4 | Replicate best-stage technique | Inconsistent correction method | Low-medium — reduce variance |
| 5 | Close Div {team_div} gap | Compound of above | High if bias corrected |
""")
