"""
Great Race 2025 — Field vs. Team Comparison
Compare any team's leg-level performance against the full field.
Includes variable-importance analysis using leg characteristics
extracted from the printed navigation booklets (no handwritten notes).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Great Race 2025 — Field Comparison",
    layout="wide",
    page_icon="🏁",
)

BASE = Path(__file__).resolve().parent
TIMES_PARQUET = BASE.parent / "Results" / "long_format_times.parquet"
LEG_CHAR_PARQUET = BASE / "leg_characteristics.parquet"
INSTR_PARQUET = BASE / "stage_instructions.parquet"

STAGE_LABELS = {
    0: "S0 – Trophy Run",
    1: "S1 – St. Paul→Rochester",
    2: "S2", 3: "S3", 4: "S4", 5: "S5",
    6: "S6", 7: "S7", 8: "S8",
    9: "S9 – Final (Irmo SC)",
}

# ─── Printed-only feature columns (exclude handwritten cols) ──────────────────
PRINTED_FEATURE_COLS = [
    "instruction_count",
    "stop_count", "yield_count",
    "traffic_light_count", "railroad_count", "roundabout_count",
    "turn_left_count", "turn_right_count", "straight_count",
    "quick_count", "very_quick_count",
    "caution_count", "highlighted_count",
    "highway_enter_count", "highway_exit_count",
    "speed_mean", "speed_std", "speed_max", "speed_min",
    "leg_duration_s",
    "stop_density", "turn_density", "quick_density",
    "highway_density", "speed_change_density",
]


# ─── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_times():
    df = pd.read_parquet(TIMES_PARQUET)
    return df[~df["Discarded"]].copy()


@st.cache_data
def load_leg_char():
    lc = pd.read_parquet(LEG_CHAR_PARQUET)
    # Convert nullable integer/float types to plain float for sklearn compatibility
    for col in lc.select_dtypes(include="Float64").columns:
        lc[col] = lc[col].astype(float)
    for col in lc.select_dtypes(include="Int64").columns:
        lc[col] = lc[col].astype(float)
    return lc


@st.cache_data
def load_instructions():
    return pd.read_parquet(INSTR_PARQUET)


df = load_times()
leg_char = load_leg_char()
instructions = load_instructions()

# Available printed features (only those present in leg_char)
feature_cols = [c for c in PRINTED_FEATURE_COLS if c in leg_char.columns]

# ─── Field-level precompute ────────────────────────────────────────────────────

field_leg_stats = (
    df.groupby(["Stage", "Leg"])["Time"]
    .agg(field_median="median", field_mean="mean", field_std="std", field_n="count")
    .reset_index()
)

# ─── Team selector ─────────────────────────────────────────────────────────────

car_info = (
    df.drop_duplicates("CAR")[["CAR", "YEAR", "DIV", "CREW", "FACTOR"]]
    .sort_values("CAR")
)
car_options = car_info["CAR"].tolist()
car_labels = {
    row["CAR"]: f"#{row['CAR']} — {row['CREW']} ({row['YEAR']}, Div {row['DIV']})"
    for _, row in car_info.iterrows()
}

with st.sidebar:
    st.header("Team Selection")
    default_idx = car_options.index(123) if 123 in car_options else 0
    selected_car = st.selectbox(
        "Select team",
        car_options,
        index=default_idx,
        format_func=lambda c: car_labels[c],
    )
    st.markdown("---")
    st.caption(
        "Comparison uses leg-level median across all non-discarded legs. "
        "Leg characteristics come from printed navigation booklets only."
    )

# ─── Team / comparison data ────────────────────────────────────────────────────

team_meta = car_info[car_info["CAR"] == selected_car].iloc[0]
team_crew = team_meta["CREW"]
team_div = team_meta["DIV"]
team_year = team_meta["YEAR"]

team_df = df[df["CAR"] == selected_car].copy()
team_leg = team_df[["Stage", "Leg", "Time", "Early"]].merge(
    field_leg_stats, on=["Stage", "Leg"]
)
team_leg["vs_median"] = team_leg["Time"] - team_leg["field_median"]
team_leg["stage_leg"] = (
    "S" + team_leg["Stage"].astype(str) + "-L" + team_leg["Leg"].astype(str)
)

# ─── Page header ───────────────────────────────────────────────────────────────

st.title("Great Race 2025 — Field vs. Team Analysis")
st.caption(
    f"Car #{selected_car} · {team_year} · Crew: {team_crew} · Division: {team_div}"
)

tabs = st.tabs([
    "Team vs Field",
    "Leg Performance Map",
    "Variable Importance",
    "Data Verification",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – TEAM VS FIELD
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.header("Team vs. Field — Overview")

    total_team = team_leg["Time"].sum()
    total_field_med_sum = field_leg_stats["field_median"].sum()
    early_pct = team_leg["Early"].mean() * 100
    field_early_pct = df.groupby("CAR")["Early"].mean().mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Penalty (team)", f"{total_team:.1f}s")
    c2.metric(
        "Sum of Leg Medians (field)",
        f"{total_field_med_sum:.1f}s",
        delta=f"{total_team - total_field_med_sum:+.1f}s vs median baseline",
        delta_color="inverse",
    )
    c3.metric(
        "Early Arrival %",
        f"{early_pct:.0f}%",
        delta=f"{early_pct - field_early_pct:+.0f}% vs field avg",
    )
    c4.metric("Legs Competed", len(team_leg))

    st.markdown("---")

    # Stage-level ranking
    stage_totals_field = df.groupby(["Stage", "CAR"])["Time"].sum().reset_index()
    stage_totals_team = team_leg.groupby("Stage")["Time"].sum().reset_index()

    rank_rows = []
    for _, row in stage_totals_team.iterrows():
        s = int(row["Stage"])
        others = stage_totals_field[stage_totals_field["Stage"] == s]["Time"]
        rank = int((others < row["Time"]).sum()) + 1
        n = len(others)
        rank_rows.append({
            "Stage": s,
            "Label": STAGE_LABELS.get(s, f"Stage {s}"),
            "Team Penalty": row["Time"],
            "Field Median": float(others.median()),
            "Rank": rank,
            "N": n,
            "Percentile": round((1 - rank / n) * 100, 1),
        })
    rank_df = pd.DataFrame(rank_rows)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Stage Penalty — Team vs. Field Median")
        fig_bar = go.Figure()
        fig_bar.add_bar(
            x=rank_df["Label"],
            y=rank_df["Field Median"],
            name="Field Median",
            marker_color="lightgray",
        )
        fig_bar.add_bar(
            x=rank_df["Label"],
            y=rank_df["Team Penalty"],
            name=f"#{selected_car}",
            marker_color="steelblue",
        )
        fig_bar.update_layout(
            barmode="group",
            xaxis_tickangle=-30,
            yaxis_title="Penalty (s)",
            legend_orientation="h",
            height=350,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.subheader("Percentile Rank by Stage (higher = better)")
        fig_pct = px.bar(
            rank_df,
            x="Label",
            y="Percentile",
            color="Percentile",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            labels={"Percentile": "Percentile", "Label": "Stage"},
            height=350,
        )
        fig_pct.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
        fig_pct.add_hline(y=50, line_dash="dash", line_color="gray",
                          annotation_text="50th pct")
        st.plotly_chart(fig_pct, use_container_width=True)

    st.subheader("Stage Summary Table")
    display_rank = rank_df.copy()
    display_rank["Team Penalty"] = display_rank["Team Penalty"].map("{:.1f}s".format)
    display_rank["Field Median"] = display_rank["Field Median"].map("{:.1f}s".format)
    display_rank["Percentile"] = display_rank["Percentile"].map("{:.1f}%".format)
    display_rank["Rank"] = (
        display_rank["Rank"].astype(str) + " / " + display_rank["N"].astype(str)
    )
    st.dataframe(
        display_rank[["Label", "Team Penalty", "Field Median", "Rank", "Percentile"]]
        .rename(columns={"Label": "Stage"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Penalty Distribution — All Legs (field vs team)")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df[df["CAR"] != selected_car]["Time"],
        name="Field",
        opacity=0.55,
        xbins=dict(start=0, end=305, size=5),
        marker_color="lightgray",
        histnorm="probability density",
    ))
    fig_dist.add_trace(go.Histogram(
        x=team_leg["Time"],
        name=f"#{selected_car}",
        opacity=0.85,
        xbins=dict(start=0, end=305, size=10),
        marker_color="steelblue",
        histnorm="probability density",
    ))
    fig_dist.update_layout(
        barmode="overlay",
        xaxis_title="Penalty (s)",
        yaxis_title="Density",
        height=300,
        legend_orientation="h",
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – LEG PERFORMANCE MAP
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("Leg-Level Performance Map")
    st.caption(
        "Each cell = team penalty minus field median. "
        "Blue = better than median (lower penalty). Red = worse."
    )

    pivot_vs = team_leg.pivot(index="Leg", columns="Stage", values="vs_median")
    z_vals = pivot_vs.values.astype(float)
    text_vals = np.where(
        np.isnan(z_vals), "", np.round(z_vals, 1).astype(str)
    )

    fig_heat = go.Figure(go.Heatmap(
        z=z_vals,
        x=[STAGE_LABELS.get(int(s), f"S{s}") for s in pivot_vs.columns],
        y=[f"Leg {l}" for l in pivot_vs.index],
        colorscale="RdBu_r",
        zmid=0,
        text=text_vals,
        texttemplate="%{text}",
        colorbar_title="vs Median (s)",
    ))
    fig_heat.update_layout(height=330, xaxis_tickangle=-30)
    st.plotly_chart(fig_heat, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("vs. Field Median per Leg")
        sorted_legs = team_leg.sort_values(["Stage", "Leg"])
        fig_vs = px.bar(
            sorted_legs,
            x="stage_leg",
            y="vs_median",
            color="vs_median",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            labels={"vs_median": "vs Median (s)", "stage_leg": "Stage-Leg"},
            height=350,
        )
        fig_vs.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_vs.update_layout(xaxis_tickangle=-50, coloraxis_showscale=False)
        st.plotly_chart(fig_vs, use_container_width=True)

    with col_r:
        st.subheader("Penalty by Leg — Early vs. Late")
        team_leg_plot = team_leg.copy()
        team_leg_plot["Arrival"] = team_leg_plot["Early"].map(
            {True: "Early", False: "Late/On-time"}
        )
        fig_el = px.scatter(
            team_leg_plot.sort_values(["Stage", "Leg"]),
            x="stage_leg",
            y="Time",
            color="Arrival",
            color_discrete_map={"Early": "cornflowerblue", "Late/On-time": "tomato"},
            labels={"Time": "Penalty (s)", "stage_leg": "Stage-Leg"},
            height=350,
        )
        fig_el.update_traces(marker_size=11)
        fig_el.update_layout(xaxis_tickangle=-50)
        st.plotly_chart(fig_el, use_container_width=True)

    # Detailed table with leg_char features
    st.subheader("Leg Detail Table")
    lc_cols = [c for c in ["instruction_count", "stop_count", "quick_count",
                            "speed_mean", "leg_duration_s"] if c in leg_char.columns]
    leg_table = team_leg.merge(
        leg_char[["Stage", "Leg"] + lc_cols],
        on=["Stage", "Leg"], how="left",
    )
    leg_table["Arrival"] = leg_table["Early"].map({True: "Early", False: "Late"})
    display_cols = (
        ["stage_leg", "Time", "vs_median", "Arrival",
         "field_median", "field_n"]
        + lc_cols
    )
    rename_map = {
        "stage_leg": "Leg", "Time": "Penalty (s)", "vs_median": "vs Median (s)",
        "field_median": "Field Median (s)", "field_n": "Field N",
        "instruction_count": "Instructions", "stop_count": "Stops",
        "quick_count": "Quicks", "speed_mean": "Avg Speed (mph)",
        "leg_duration_s": "Duration (s)",
    }
    st.dataframe(
        leg_table[display_cols].rename(columns=rename_map),
        use_container_width=True,
        hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – VARIABLE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Variable Importance — What Leg Features Drive Performance?")
    st.markdown(
        "A Random Forest model is trained on **all teams** to predict each leg's "
        "relative penalty (vs field median). Feature importances reveal which route "
        "characteristics most affect scoring. The team-specific panel shows Pearson "
        "correlations between those same features and the selected team's relative penalty."
    )

    # ── Join all teams with leg characteristics ──────────────────────────────
    all_leg_perf = df.merge(
        field_leg_stats[["Stage", "Leg", "field_median"]], on=["Stage", "Leg"]
    )
    all_leg_perf["rel_penalty"] = all_leg_perf["Time"] - all_leg_perf["field_median"]

    merged = all_leg_perf.merge(
        leg_char[["Stage", "Leg"] + feature_cols],
        on=["Stage", "Leg"],
        how="inner",
    ).dropna(subset=feature_cols)

    team_merged = merged[merged["CAR"] == selected_car].copy()
    team_merged["stage_leg"] = (
        "S" + team_merged["Stage"].astype(str)
        + "-L" + team_merged["Leg"].astype(str)
    )

    # ── Train RF on field ────────────────────────────────────────────────────
    X_all = merged[feature_cols].values
    y_all = merged["rel_penalty"].values

    rf_all = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_all.fit(X_all, y_all)

    fi_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_all.feature_importances_,
    }).sort_values("Importance", ascending=False)

    # ── Correlations for selected team ───────────────────────────────────────
    corr_series = (
        team_merged[feature_cols + ["rel_penalty"]]
        .corr()["rel_penalty"]
        .drop("rel_penalty")
    )
    corr_df = (
        corr_series.reset_index()
        .rename(columns={"index": "Feature", "rel_penalty": "Correlation"})
        .sort_values("Correlation", key=abs, ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Field-wide: RF Feature Importance")
        st.caption(f"Trained on {len(merged):,} leg×team observations")
        fig_fi = px.bar(
            fi_df.head(15),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            height=480,
        )
        fig_fi.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col2:
        st.subheader(f"#{selected_car}: Correlation with Relative Penalty")
        n_pts = len(team_merged)
        st.caption(
            f"{n_pts} legs — Pearson r (positive = feature associated with higher penalty)"
        )
        if n_pts < 3:
            st.warning("Too few data points for meaningful correlation.")
        else:
            fig_corr = px.bar(
                corr_df.head(15),
                x="Correlation",
                y="Feature",
                orientation="h",
                color="Correlation",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                height=480,
            )
            fig_corr.add_vline(x=0, line_color="gray")
            fig_corr.update_layout(
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.subheader("Scatter Explorer")

    feat_choice = st.selectbox("Feature to plot against team's relative penalty", feature_cols)

    if len(team_merged) >= 2:
        fig_scat = px.scatter(
            team_merged,
            x=feat_choice,
            y="rel_penalty",
            color=team_merged["Early"].map({True: "Early", False: "Late"}),
            text="stage_leg",
            color_discrete_map={"Early": "cornflowerblue", "Late": "tomato"},
            labels={
                "rel_penalty": "Team Penalty vs Field Median (s)",
                feat_choice: feat_choice,
            },
    
            height=380,
        )
        fig_scat.update_traces(textposition="top center")
        fig_scat.add_hline(y=0, line_dash="dash", line_color="gray",
                           annotation_text="field median")
        st.plotly_chart(fig_scat, use_container_width=True)
    else:
        st.info("Not enough data points for scatter plot.")

    # ── Division comparison ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Team vs. Division: Avg Relative Penalty per Leg Feature Quartile")
    st.caption(
        "Split legs into quartiles by the selected feature. "
        "Does the team struggle more on legs where the feature is high?"
    )

    if len(team_merged) >= 4 and feat_choice in merged.columns:
        quartile_df = merged.copy()
        quartile_df["quartile"] = pd.qcut(
            quartile_df[feat_choice], q=4,
            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
            duplicates="drop",
        )
        div_cars = df[df["DIV"] == team_div]["CAR"].unique()
        q_team = team_merged.copy()
        q_team["quartile"] = pd.cut(
            q_team[feat_choice],
            bins=pd.qcut(quartile_df[feat_choice], q=4, retbins=True, duplicates="drop")[1],
            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
            include_lowest=True,
        )

        q_field = quartile_df.groupby("quartile")["rel_penalty"].median().reset_index()
        q_field["group"] = "Field"
        q_div = quartile_df[quartile_df["CAR"].isin(div_cars)].groupby("quartile")["rel_penalty"].median().reset_index()
        q_div["group"] = f"Div {team_div}"
        q_t = q_team.groupby("quartile")["rel_penalty"].median().reset_index()
        q_t["group"] = f"#{selected_car}"

        q_all = pd.concat([q_field, q_div, q_t], ignore_index=True)
        q_all = q_all.rename(columns={"rel_penalty": "Median Relative Penalty (s)"})

        fig_q = px.line(
            q_all,
            x="quartile",
            y="Median Relative Penalty (s)",
            color="group",
            markers=True,
            color_discrete_map={
                "Field": "lightgray",
                f"Div {team_div}": "orange",
                f"#{selected_car}": "steelblue",
            },
            height=320,
        )
        fig_q.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_q, use_container_width=True)
    else:
        st.info("Need ≥4 data points to compute quartiles.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – DATA VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("Data Verification")
    st.markdown(
        "Verify the data extracted from the printed navigation booklets. "
        "Checks leg_characteristics.parquet and stage_instructions.parquet for "
        "completeness, boundary correctness, and plausible values."
    )

    # ── Leg Characteristics coverage ─────────────────────────────────────────
    st.subheader("leg_characteristics.parquet — Coverage")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        lc_cov = (
            leg_char.groupby("Stage")
            .agg(Legs=("Leg", "count"), Leg_List=("Leg", lambda x: str(sorted(x.astype(int).tolist()))))
            .reset_index()
        )
        st.markdown("**Legs per Stage**")
        st.dataframe(lc_cov, use_container_width=True, hide_index=True)

        # Expected vs actual
        expected_legs = df.groupby("Stage")["Leg"].nunique().reset_index()
        expected_legs.columns = ["Stage", "Expected Legs (from results)"]
        lc_cov2 = lc_cov[["Stage", "Legs"]].merge(expected_legs, on="Stage", how="outer")
        lc_cov2["Match"] = lc_cov2["Legs"] == lc_cov2["Expected Legs (from results)"]
        lc_cov2 = lc_cov2.rename(columns={"Legs": "Legs in leg_char"})
        issues = lc_cov2[~lc_cov2["Match"]]
        if len(issues):
            st.warning(f"{len(issues)} stage(s) have mismatched leg counts:")
            st.dataframe(issues, use_container_width=True, hide_index=True)
        else:
            st.success("Leg counts match results data for all stages.")

    with col_b:
        st.markdown("**Instruction Count Heatmap (Stage × Leg)**")
        lc_pivot = leg_char.pivot(index="Leg", columns="Stage", values="instruction_count")
        z = lc_pivot.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        txt = np.full(z.shape, "", dtype=object)
        txt[~np.isnan(z)] = np.round(z[~np.isnan(z)], 0).astype(int).astype(str)
        fig_lc_h = go.Figure(go.Heatmap(
            z=z,
            x=[f"S{int(s)}" for s in lc_pivot.columns],
            y=[f"L{int(l)}" for l in lc_pivot.index],
            colorscale="YlOrRd",
            text=txt,
            texttemplate="%{text}",
            colorbar_title="# Instrs",
        ))
        fig_lc_h.update_layout(height=300)
        st.plotly_chart(fig_lc_h, use_container_width=True)

    # Speed reasonableness
    st.subheader("Speed Statistics (printed, mph)")
    speed_df = leg_char[["Stage", "Leg", "speed_mean", "speed_min", "speed_max", "speed_std"]].dropna()
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        fig_speed = px.scatter(
            speed_df,
            x="Stage",
            y="speed_mean",
            error_y="speed_std",
            color="Leg",
            labels={"speed_mean": "Mean Target Speed (mph)"},
            title="Mean Speed ± SD per (Stage, Leg)",
            height=320,
        )
        fig_speed.update_yaxes(range=[0, max(75, speed_df["speed_max"].max() + 5)])
        st.plotly_chart(fig_speed, use_container_width=True)

    with col_s2:
        st.markdown("**Speed range per leg**")
        st.dataframe(
            speed_df.rename(columns={
                "speed_mean": "Mean", "speed_min": "Min",
                "speed_max": "Max", "speed_std": "SD",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Null audit
    st.subheader("Null Audit — leg_characteristics.parquet")
    null_counts = leg_char.isnull().sum()
    null_pct = (null_counts / len(leg_char) * 100).round(1)
    null_df = pd.DataFrame({
        "Column": null_counts.index,
        "Null Count": null_counts.values,
        "Null %": null_pct.values,
    })
    null_df = null_df[null_df["Null Count"] > 0].sort_values("Null %", ascending=False)
    if len(null_df):
        col_n1, col_n2 = st.columns([1, 1])
        with col_n1:
            st.dataframe(null_df, use_container_width=True, hide_index=True)
        with col_n2:
            fig_null = px.bar(
                null_df.head(15),
                x="Null %",
                y="Column",
                orientation="h",
                color="Null %",
                color_continuous_scale="Reds",
                height=380,
            )
            fig_null.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_null, use_container_width=True)
    else:
        st.success("No nulls in leg_characteristics.parquet!")

    st.markdown("---")

    # ── Stage Instructions coverage ───────────────────────────────────────────
    st.subheader("stage_instructions.parquet — Coverage & Boundary Checks")

    instr_coverage = (
        instructions.groupby(["stage", "leg_num"])
        .agg(
            N_Instructions=("instruction_num", "count"),
            Leg_Starts=("is_leg_start", "sum"),
            Checkpoints=("is_checkpoint", "sum"),
            Quick_Count=("is_quick", "sum"),
            Very_Quick_Count=("is_very_quick", "sum"),
            Caution_Count=("caution_noted", "sum"),
        )
        .reset_index()
        .rename(columns={"stage": "Stage", "leg_num": "Leg"})
    )
    instr_coverage["Start_OK"] = instr_coverage["Leg_Starts"] == 1
    instr_coverage["Checkpoint_OK"] = instr_coverage["Checkpoints"] == 1
    instr_coverage["Issues"] = (
        ~instr_coverage["Start_OK"] | ~instr_coverage["Checkpoint_OK"]
    )

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**Instruction count heatmap**")
        ip = instr_coverage.pivot(index="Leg", columns="Stage", values="N_Instructions")
        z2 = ip.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        txt2 = np.full(z2.shape, "", dtype=object)
        txt2[~np.isnan(z2)] = np.round(z2[~np.isnan(z2)], 0).astype(int).astype(str)
        fig_ip = go.Figure(go.Heatmap(
            z=z2,
            x=[f"S{int(s)}" for s in ip.columns],
            y=[f"L{int(l)}" for l in ip.index],
            colorscale="Blues",
            text=txt2,
            texttemplate="%{text}",
            colorbar_title="Count",
        ))
        fig_ip.update_layout(height=300)
        st.plotly_chart(fig_ip, use_container_width=True)

    with col_d:
        st.markdown("**Leg boundary check** (each leg needs exactly 1 start + 1 checkpoint)")
        bc_display = instr_coverage[
            ["Stage", "Leg", "N_Instructions", "Leg_Starts", "Checkpoints", "Issues"]
        ].copy()

        def highlight_issues(row):
            style = [""] * len(row)
            if row["Issues"]:
                style = ["background-color: #ffcccc"] * len(row)
            return style

        styled = bc_display.style.apply(highlight_issues, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        n_issues = instr_coverage["Issues"].sum()
        if n_issues:
            st.warning(f"{n_issues} leg(s) have boundary issues (missing or duplicate markers).")
        else:
            st.success("All legs have exactly one start and one checkpoint.")

    # Instruction type distribution
    st.subheader("Instruction Type Distribution (stage_instructions)")
    col_e, col_f = st.columns(2)

    with col_e:
        type_counts = (
            instructions["diagram_type"]
            .value_counts()
            .reset_index()
            .rename(columns={"diagram_type": "Type", "count": "Count"})
        )
        fig_types = px.bar(
            type_counts.head(20),
            x="Count",
            y="Type",
            orientation="h",
            height=400,
            title="Turn/Instruction Type Frequencies",
        )
        fig_types.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_types, use_container_width=True)

    with col_f:
        quick_by_stage = (
            instructions.groupby("stage")[["is_quick", "is_very_quick", "caution_noted"]]
            .sum()
            .reset_index()
            .rename(columns={
                "stage": "Stage",
                "is_quick": "Quick",
                "is_very_quick": "Very Quick",
                "caution_noted": "Caution",
            })
        )
        fig_quick = px.bar(
            quick_by_stage.melt(id_vars="Stage", value_vars=["Quick", "Very Quick", "Caution"]),
            x="Stage",
            y="value",
            color="variable",
            barmode="group",
            labels={"value": "Count", "variable": "Warning Type"},
            height=350,
            title="Warning/Quick Instruction Counts by Stage",
        )
        st.plotly_chart(fig_quick, use_container_width=True)

    # Speed coverage check: leg_char vs stage_instructions
    st.subheader("Cross-Check: Speed Data leg_char vs stage_instructions")
    instr_speeds = (
        instructions[instructions["target_speed_mph"].notna()]
        .groupby(["stage", "leg_num"])["target_speed_mph"]
        .agg(["mean", "min", "max", "std", "count"])
        .reset_index()
        .rename(columns={"stage": "Stage", "leg_num": "Leg",
                         "mean": "SI_mean", "min": "SI_min",
                         "max": "SI_max", "std": "SI_std", "count": "SI_n"})
    )
    lc_speeds = leg_char[["Stage", "Leg", "speed_mean", "speed_min", "speed_max"]].dropna()
    speed_compare = lc_speeds.merge(instr_speeds, on=["Stage", "Leg"], how="outer")
    speed_compare["mean_delta"] = (speed_compare["speed_mean"] - speed_compare["SI_mean"]).round(2)
    st.dataframe(
        speed_compare[["Stage", "Leg", "speed_mean", "SI_mean", "mean_delta",
                       "speed_min", "SI_min", "speed_max", "SI_max", "SI_n"]],
        use_container_width=True,
        hide_index=True,
    )
    max_delta = speed_compare["mean_delta"].abs().max()
    if pd.notna(max_delta) and max_delta > 2:
        st.warning(
            f"Max mean-speed discrepancy between leg_char and stage_instructions: "
            f"{max_delta:.1f} mph — review those legs."
        )
    elif pd.notna(max_delta):
        st.success(f"Speed values are consistent (max delta {max_delta:.2f} mph).")

    # Raw data expanders
    with st.expander("Raw leg_characteristics.parquet"):
        st.dataframe(leg_char, use_container_width=True)

    with st.expander("Raw stage_instructions.parquet (printed fields only)"):
        printed_cols = [c for c in instructions.columns if "handwritten" not in c.lower()
                        and "hw_" not in c.lower()]
        st.dataframe(instructions[printed_cols].head(300), use_container_width=True)
