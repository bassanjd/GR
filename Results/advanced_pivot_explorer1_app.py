import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor
import json
import zipfile
import tempfile

st.set_page_config(page_title="BI Monte Carlo Explorer", layout="wide")
st.title("📊 BI + Monte Carlo Exploration Platform")

# ---------------------------------------------------
# Session State Initialization
# ---------------------------------------------------
if "df_original" not in st.session_state:
    st.session_state.df_original = None

if "filters" not in st.session_state:
    st.session_state.filters = {}

if "chart_gallery" not in st.session_state:
    st.session_state.chart_gallery = []

if "calculated_fields" not in st.session_state:
    st.session_state.calculated_fields = {}

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def apply_filters(df):
    filtered_df = df.copy()
    for col, values in st.session_state.filters.items():
        if values:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
    return filtered_df

def run_monte_carlo(df, target, sim_type, iterations):
    data = df[target].dropna().values

    if len(data) < 5:
        return None

    if sim_type == "Bootstrap (Empirical)":
        sims = np.random.choice(data, size=iterations, replace=True)

    elif sim_type == "Normal":
        mu = np.mean(data)
        sigma = np.std(data)
        sims = np.random.normal(mu, sigma, iterations)

    elif sim_type == "Lognormal":
        data = data[data > 0]
        mu = np.mean(np.log(data))
        sigma = np.std(np.log(data))
        sims = np.random.lognormal(mu, sigma, iterations)

    return sims

# ---------------------------------------------------
# File Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader("Upload Parquet File", type=["parquet"])

if uploaded_file:
    df = pd.read_parquet(uploaded_file)
    st.session_state.df_original = df
else:
    if st.session_state.df_original is None:
        st.info("Upload a parquet file to begin.")
        st.stop()
    df = st.session_state.df_original

# ---------------------------------------------------
# Calculated Fields
# ---------------------------------------------------
st.sidebar.header("🧮 Calculated Fields")

new_col = st.sidebar.text_input("New Column Name")
formula = st.sidebar.text_input("Formula (pandas eval syntax)")

if st.sidebar.button("Add Calculated Field"):
    try:
        df[new_col] = df.eval(formula)
        st.session_state.calculated_fields[new_col] = formula
        st.success("Calculated field added")
    except Exception as e:
        st.error(f"Invalid formula: {e}")

# ---------------------------------------------------
# Time-Series Controls
# ---------------------------------------------------
datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

if datetime_cols:
    st.sidebar.header("⏳ Time Controls")
    time_col = st.sidebar.selectbox("Time Column", datetime_cols)
    freq = st.sidebar.selectbox("Resample Frequency", ["None", "D", "W", "M", "Q", "Y"])

    if freq != "None":
        df = (
            df.set_index(time_col)
              .resample(freq)
              .mean(numeric_only=True)
              .reset_index()
        )

# ---------------------------------------------------
# Apply Filters
# ---------------------------------------------------
df_filtered = apply_filters(df)

st.success(f"Filtered Rows: {len(df_filtered):,}")

numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
all_cols = df_filtered.columns.tolist()

# ---------------------------------------------------
# Pivot Controls
# ---------------------------------------------------
st.sidebar.header("📊 Pivot Controls")

rows = st.sidebar.multiselect("Rows", all_cols)
values = st.sidebar.selectbox("Values", numeric_cols)
agg_func = st.sidebar.selectbox("Aggregation", ["sum", "mean", "median", "count", "min", "max"])
chart_type = st.sidebar.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Area", "Box"])

# ---------------------------------------------------
# Pivot Table & Chart
# ---------------------------------------------------
if rows:
    pivot_df = pd.pivot_table(
        df_filtered,
        index=rows,
        values=values,
        aggfunc=agg_func
    ).reset_index()

    st.subheader("Pivot Table")
    st.dataframe(pivot_df, use_container_width=True)

    if chart_type == "Bar":
        fig = px.bar(pivot_df, x=rows[0], y=values)
    elif chart_type == "Line":
        fig = px.line(pivot_df, x=rows[0], y=values)
    elif chart_type == "Scatter":
        fig = px.scatter(pivot_df, x=rows[0], y=values)
    elif chart_type == "Area":
        fig = px.area(pivot_df, x=rows[0], y=values)
    elif chart_type == "Box":
        fig = px.box(df_filtered, x=rows[0], y=values)

    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

    # Crossfiltering
    if event and "selection" in event:
        selected = event["selection"]["points"]
        if selected:
            selected_vals = [p["x"] for p in selected]
            st.session_state.filters[rows[0]] = selected_vals
            st.rerun()

    if st.button("💾 Save Chart to Gallery"):
        st.session_state.chart_gallery.append({
            "figure": fig,
            "description": f"{chart_type} | {values}"
        })

# ---------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------
if len(numeric_cols) > 1:
    if st.sidebar.checkbox("Show Correlation Heatmap"):
        corr = df_filtered[numeric_cols].corr()
        heatmap = go.Figure(data=go.Heatmap(
            z=corr,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0
        ))
        st.plotly_chart(heatmap, use_container_width=True)

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------
if st.sidebar.checkbox("Run Feature Importance"):
    target = st.sidebar.selectbox("Target Variable", numeric_cols)
    X = df_filtered[numeric_cols].drop(columns=[target])
    y = df_filtered[target]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance")

    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------------------------------------
# Monte Carlo Simulator
# ---------------------------------------------------
st.sidebar.header("🎲 Monte Carlo")

if st.sidebar.checkbox("Enable Monte Carlo"):
    target_metric = st.sidebar.selectbox("Target Metric", numeric_cols)
    sim_type = st.sidebar.selectbox("Simulation Type", ["Bootstrap (Empirical)", "Normal", "Lognormal"])
    iterations = st.sidebar.slider("Iterations", 1000, 50000, 10000, 1000)
    confidence = st.sidebar.slider("Confidence Level", 0.80, 0.99, 0.95)

    sims = run_monte_carlo(df_filtered, target_metric, sim_type, iterations)

    if sims is not None:
        lower = np.percentile(sims, (1-confidence)/2*100)
        upper = np.percentile(sims, (1+confidence)/2*100)
        mean_val = np.mean(sims)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{mean_val:,.4f}")
        col2.metric("Lower CI", f"{lower:,.4f}")
        col3.metric("Upper CI", f"{upper:,.4f}")

        fig_mc = px.histogram(sims, nbins=50, title="Monte Carlo Distribution")
        fig_mc.add_vline(x=mean_val)
        fig_mc.add_vline(x=lower, line_dash="dash")
        fig_mc.add_vline(x=upper, line_dash="dash")

        st.plotly_chart(fig_mc, use_container_width=True)

        if st.button("💾 Save Monte Carlo to Gallery"):
            st.session_state.chart_gallery.append({
                "figure": fig_mc,
                "description": f"Monte Carlo | {target_metric}"
            })

# ---------------------------------------------------
# Chart Gallery
# ---------------------------------------------------
if st.session_state.chart_gallery:
    st.divider()
    st.header("🗂 Chart Gallery")

    for i, item in enumerate(st.session_state.chart_gallery):
        st.subheader(item["description"])
        st.plotly_chart(item["figure"], use_container_width=True)

        html_str = pio.to_html(item["figure"], full_html=True)
        st.download_button(
            "Download HTML",
            data=html_str.encode("utf-8"),
            file_name=f"chart_{i}.html",
            key=f"dl_{i}"
        )

# ---------------------------------------------------
# Export Dashboard ZIP
# ---------------------------------------------------
if st.button("📦 Export Full Dashboard (ZIP)"):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        with zipfile.ZipFile(tmp.name, "w") as zf:
            for i, item in enumerate(st.session_state.chart_gallery):
                html = pio.to_html(item["figure"], full_html=True)
                zf.writestr(f"chart_{i}.html", html)

        with open(tmp.name, "rb") as f:
            st.download_button(
                "Download Dashboard ZIP",
                f,
                file_name="dashboard.zip"
            )

# ---------------------------------------------------
# Save / Load Session
# ---------------------------------------------------
st.sidebar.header("💾 Session")

session_data = {
    "filters": st.session_state.filters,
    "calculated_fields": st.session_state.calculated_fields
}

st.sidebar.download_button(
    "Download Session",
    data=json.dumps(session_data),
    file_name="session.json"
)

uploaded_session = st.sidebar.file_uploader("Load Session", type=["json"])

if uploaded_session:
    session_loaded = json.load(uploaded_session)
    st.session_state.filters = session_loaded.get("filters", {})
    st.session_state.calculated_fields = session_loaded.get("calculated_fields", {})
    st.success("Session Loaded")