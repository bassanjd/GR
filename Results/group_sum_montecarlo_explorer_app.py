import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
import zipfile
import io
from plotly.colors import DEFAULT_PLOTLY_COLORS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

st.set_page_config(page_title="Advanced Rally Data Explorer", layout="wide")

# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------

if "df" not in st.session_state:
    st.session_state.df = None

if "gallery" not in st.session_state:
    st.session_state.gallery = []

# ------------------------------------------------------------
# File Upload
# ------------------------------------------------------------

st.sidebar.title("Data")

uploaded = st.sidebar.file_uploader("Upload Parquet File", type=["parquet"])

if uploaded:
    st.session_state.df = pd.read_parquet(uploaded)

df = st.session_state.df

if df is None:
    st.info("Upload a parquet file to begin.")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
all_cols = df.columns.tolist()

# ------------------------------------------------------------
# Crossfilter Controls
# ------------------------------------------------------------

st.sidebar.markdown("## Global Filters")

filter_col1 = st.sidebar.selectbox("Filter Column 1", ["None"] + all_cols, index=12)

if filter_col1 != "None":
    if df[filter_col1].dtype == "object":
        values = st.sidebar.multiselect("Select Values", df[filter_col1].unique())
        if values:
            df = df[df[filter_col1].isin(values)]
    else:
        min_val = float(df[filter_col1].min())
        max_val = float(df[filter_col1].max())
        rng = st.sidebar.slider("Range", min_val, max_val, (min_val, max_val))
        df = df[(df[filter_col1] >= rng[0]) & (df[filter_col1] <= rng[1])]

filter_col2 = st.sidebar.selectbox("Filter Column 2", ["None"] + all_cols)

if filter_col2 != "None":
    if df[filter_col2].dtype == "object":
        values = st.sidebar.multiselect("Select Values", df[filter_col2].unique())
        if values:
            df = df[df[filter_col2].isin(values)]
    else:
        min_val = float(df[filter_col2].min())
        max_val = float(df[filter_col2].max())
        rng = st.sidebar.slider("Range", min_val, max_val, (min_val, max_val))
        df = df[(df[filter_col2] >= rng[0]) & (df[filter_col2] <= rng[1])]

filter_col3 = st.sidebar.selectbox("Filter Column 3", ["None"] + all_cols)

if filter_col3 != "None":
    if df[filter_col3].dtype == "object":
        values = st.sidebar.multiselect("Select Values", df[filter_col3].unique())
        if values:
            df = df[df[filter_col3].isin(values)]
    else:
        min_val = float(df[filter_col3].min())
        max_val = float(df[filter_col3].max())
        rng = st.sidebar.slider("Range", min_val, max_val, (min_val, max_val))
        df = df[(df[filter_col3] >= rng[0]) & (df[filter_col3] <= rng[1])]

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

tabs = st.tabs([
    "Pivot Table Explorer",
    "Correlation",
    "Monte Carlo Analysis",
    "Gallery",
    "Session"
])

# ============================================================
# PIVOT EXPLORER
# ============================================================

with tabs[0]:
    st.header("Pivot Table Explorer")

    row = st.selectbox("Rows", all_cols, index=6)
    col = st.selectbox("Columns", ["None"] + all_cols, index = 8)
    val = st.selectbox("Values", numeric_cols, index = 8)
    agg = st.selectbox("Aggregation", ["sum", "mean", "median", "count"])

    pivot_df = pd.pivot_table(
        df,
        index=row,
        columns=None if col == "None" else col,
        values=val,
        aggfunc=agg
    )

    st.dataframe(pivot_df)

    # fig = px.bar(pivot_df.reset_index(), x=row, y=val)

    pivot_reset = pivot_df.reset_index()

    # If no column dimension → single value column
    if col == "None":
        value_col = pivot_reset.columns[-1]  # last column is aggregated value
        fig = px.bar(pivot_reset, x=row, y=value_col)

    # If column dimension exists → wide format
    else:
        fig = px.bar(
            pivot_reset,
            x=row,
            y=pivot_reset.columns[1:],  # all value columns
        )
        fig.update_layout(
            legend_title_text=col
            )

    st.plotly_chart(fig, use_container_width=True)

    if st.button("Save Chart to Gallery"):
        st.session_state.gallery.append({
            "title": f"{val} by {row}",
            "figure": fig
        })

    html_str = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
    st.download_button("Download HTML", html_str.encode(), "pivot_chart.html")

# ============================================================
# CORRELATION
# ============================================================
with tabs[1]:
    st.header("Field Clustering & Team Comparison")

    team_col = st.selectbox("Team Column", all_cols, key="cluster_team_col")
    time_col = st.selectbox("Performance Metric (e.g., Time)", numeric_cols, key="cluster_time_col")

    index_cols = st.multiselect(
        "Columns Defining Shared Event (Stage / Leg etc.)",
        all_cols,
        default=[c for c in ["Stage", "Leg"] if c in all_cols],
        key="cluster_index_cols"
    )

    if not index_cols:
        st.info("Select at least one index column.")
        st.stop()

    # ---- Pivot to Team x Event matrix ----
    pivot_df = df.pivot_table(
        index=index_cols,
        columns=team_col,
        values=time_col,
        aggfunc="sum"
    )

    pivot_df = pivot_df.dropna()

    if pivot_df.shape[1] < 3:
        st.warning("Need at least 3 teams for clustering.")
        st.stop()

    # Teams as rows
    team_matrix = pivot_df.T

    # ---- Scaling Option ----
    scale = st.checkbox("Standardize Teams (Recommended)", value=True)

    X = team_matrix.values

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # ---- PCA Projection ----
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Team": team_matrix.index
    })

    # ---- Clustering ----
    k = st.slider("Number of Clusters", 2, min(10, len(team_matrix)), 3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    pca_df["Cluster"] = clusters.astype(str)

    selected_teams = st.multiselect(
    "Highlight Teams",
    team_matrix.index,
    default=[team_matrix.index[0]],
    key="highlight_teams"
    )

    # ---- PCA Scatter Plot ----
    fig_scatter = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_name="Team",
        title="Team Clusters (PCA Projection)"
    )

    # Highlight selected team
    highlight = pca_df[pca_df["Team"].isin(selected_teams)]
    
    fig_scatter.add_trace(
        go.Scatter(
            x=highlight["PC1"],
            y=highlight["PC2"],
            mode="markers+text",
            text=highlight["Team"],
            textposition="top center",
            marker=dict(size=18, symbol="star"),
            name="Highlighted Teams"
        )
    )

    st.plotly_chart(fig_scatter, use_container_width=True, key="cluster_scatter")

    # ---- Distance Heatmap ----
    dist_matrix = squareform(pdist(X))
    dist_df = pd.DataFrame(
        dist_matrix,
        index=team_matrix.index,
        columns=team_matrix.index
    )

    fig_dist = px.imshow(
        dist_df,
        title="Team Distance Matrix (Lower = More Similar)",
        text_auto=False
    )

    st.plotly_chart(fig_dist, use_container_width=True, key="cluster_distance")

    # ---- Field Comparison Metrics ----
    field_mean = X.mean(axis=0)
    if selected_teams:
        deviations = []
        for team in selected_teams:
            idx = list(team_matrix.index).index(team)
            team_vector = X[idx]
            deviations.append(np.linalg.norm(team_vector - field_mean))
    
        st.write("### Distance From Field Centroid")
        for team, dev in zip(selected_teams, deviations):
            st.write(f"{team}: {dev:.3f}")

    if st.button("Save Cluster View to Gallery", key="save_cluster"):
        st.session_state.gallery.append({
            "title": "Field Clustering",
            "figure": fig_scatter
        })

# ============================================================
# MONTE CARLO RALLY
# ============================================================

with tabs[2]:
    st.header("Monte Carlo Simulation – Cumulative Time")

    group_col = st.selectbox("Group Column", all_cols, index=6)
    time_col = st.selectbox("Stage Time Column", numeric_cols, index=8)
    group_by_leg = st.checkbox("Group by Leg (if applicable)", value=False)

    n_sim = st.slider("Simulations", 1000, 50000, 10000, step=1000)

    groups = df[group_col].unique()

    results = {}
    actual = {}

    for g in groups:
        group_data = df[df[group_col] == g][time_col].values

        if group_by_leg:
            # If grouping by leg, we need to sum times for each leg
            leg_groups = df[df[group_col] == g].groupby('Leg')[time_col].sum().values
            group_data = leg_groups
            sims = np.random.choice(group_data, size=(n_sim, len(group_data)), replace=True)
            totals = sims.sum(axis=1)
            results[g] = totals
            actual[g] = group_data.sum()

        else:
            sims = np.random.choice(group_data, size=(n_sim, len(group_data)), replace=True)
            totals = sims.sum(axis=1)
            results[g] = totals
            actual[g] = group_data.sum()

    fig = go.Figure()
    color_cycle = DEFAULT_PLOTLY_COLORS  # Plotly default colors

    for i, g in enumerate(results):
        color = color_cycle[i % len(color_cycle)]  # cycle through default colors

        # Add histogram with assigned color
        fig.add_trace(go.Histogram(
            x=results[g],
            name=str(g),
            histnorm="probability density",
            marker_color=color,
            opacity=0.5,          
            marker_line=dict(color=color, width=2)  # show only outline
        ))

        # Add vertical line with same color
        fig.add_vline(
            x=actual[g],
            line_color=color,
            line_dash="dash",
            name=str(g)
        )
            # Add invisible scatter for hovertext at the vline
        # y_min, y_max = 0, max([max(results[g]) for g in results]) * 1.05  # adjust to your y-axis range
        fig.add_trace(go.Scatter(
            x=[actual[g], actual[g]],
            y=[0,0],
            mode='lines',
            line=dict(color=color, width=0),  # invisible line for hover
            hoverinfo='text',
            hovertext=[f"Group: {g}<br>Time: {actual[g]}"]*2,
            showlegend=False
        ))

    # for g in results:
    #     fig.add_trace(go.Histogram(
    #         x=results[g],
    #         name=str(g),
    #         opacity=0.5,
    #         histnorm="probability density"))
    #     fig.add_vline(
    #         x=actual[g],  
    #         # line_color='red', 
    #         name=str(g))
    
    fig.update_layout(barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Save Monte Carlo to Gallery"):
        st.session_state.gallery.append({
            "title": "Monte Carlo Cumulative",
            "figure": fig
        })



# ============================================================
# GALLERY
# ============================================================

with tabs[3]:
    st.header("Chart Gallery")

    if st.button("Clear All Charts", key="clear_gallery"):
        st.session_state.gallery = []
        st.rerun()

    for i, item in enumerate(st.session_state.gallery):
        st.subheader(item["title"])
        st.plotly_chart(
            item["figure"],
            use_container_width=True,
            key=f"gallery_chart_{i}"   # 👈 ADD THIS
        )

        html_str = pio.to_html(item["figure"], full_html=True, include_plotlyjs="cdn")
        st.download_button(
            f"Download {item['title']}",
            html_str.encode(),
            file_name=f"{item['title'].replace(' ', '_')}.html",
            key=f"gallery_download_{i}"   # 👈 ALSO ADD THIS
        )


    # Export Full Dashboard
    if st.button("Export Full Dashboard (ZIP)"):
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as z:
            for i, item in enumerate(st.session_state.gallery):
                html_str = pio.to_html(item["figure"], full_html=True, include_plotlyjs="cdn")
                z.writestr(f"chart_{i}.html", html_str)

        st.download_button("Download ZIP", zip_buffer.getvalue(), "dashboard.zip")

# ============================================================
# SESSION SAVE / LOAD
# ============================================================

with tabs[4]:
    st.header("Save / Load Session")

    if st.button("Save Session"):
        session_data = {
            "gallery_titles": [g["title"] for g in st.session_state.gallery]
        }

        st.download_button(
            "Download Session JSON",
            json.dumps(session_data).encode(),
            "session.json"
        )

    uploaded_session = st.file_uploader("Load Session", type=["json"])

    if uploaded_session:
        data = json.load(uploaded_session)
        st.success("Session Loaded (titles only)")