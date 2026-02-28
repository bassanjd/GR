import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO

st.set_page_config(page_title="Advanced Parquet Explorer", layout="wide")

st.title("📊 Advanced Parquet Pivot & Chart Explorer")

# -----------------------------------------
# Initialize Persistent Chart Gallery
# -----------------------------------------
if "chart_gallery" not in st.session_state:
    st.session_state.chart_gallery = []

# -----------------------------------------
# File Upload
# -----------------------------------------
uploaded_file = st.file_uploader("Upload a Parquet file", type=["parquet"])

if uploaded_file:

    df = pd.read_parquet(uploaded_file)

    st.success(f"Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    # -----------------------------------------
    # Sidebar Controls
    # -----------------------------------------
    st.sidebar.header("Pivot Controls")

    rows = st.sidebar.multiselect("Rows (Group By)", all_cols)
    cols = st.sidebar.multiselect("Columns (Optional)", all_cols)
    values = st.sidebar.selectbox("Values (Numeric)", numeric_cols)

    agg_func = st.sidebar.selectbox(
        "Aggregation",
        ["sum", "mean", "median", "count", "min", "max"]
    )

    chart_type = st.sidebar.selectbox(
        "Chart Type",
        ["Bar", "Line", "Scatter", "Area", "Box"]
    )

    st.sidebar.divider()

    show_corr = st.sidebar.checkbox("Show Correlation Heatmap")
    show_raw = st.sidebar.checkbox("Show Raw Data")

    # -----------------------------------------
    # Pivot Table & Chart
    # -----------------------------------------
    if rows and values:

        pivot_df = pd.pivot_table(
            df,
            index=rows,
            columns=cols if cols else None,
            values=values,
            aggfunc=agg_func
        )

        pivot_df = pivot_df.reset_index()

        st.subheader("📋 Pivot Table")
        st.dataframe(pivot_df, use_container_width=True)

        st.subheader("📈 Pivot Chart")

        plot_df = pivot_df.copy()

        if isinstance(plot_df.columns, pd.MultiIndex):
            plot_df.columns = [
                "_".join([str(i) for i in col if i]) for col in plot_df.columns
            ]

        x_axis = rows[0]
        y_columns = plot_df.columns[1:]

        if chart_type == "Bar":
            fig = px.bar(plot_df, x=x_axis, y=y_columns)

        elif chart_type == "Line":
            fig = px.line(plot_df, x=x_axis, y=y_columns)

        elif chart_type == "Scatter":
            fig = px.scatter(plot_df, x=x_axis, y=y_columns)

        elif chart_type == "Area":
            fig = px.area(plot_df, x=x_axis, y=y_columns)

        elif chart_type == "Box":
            fig = px.box(df, x=rows[0], y=values)

        fig.update_layout(
            height=600,
            template="plotly_white",
            legend_title_text=""
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        # Save chart
        if col1.button("💾 Save to Gallery"):
            st.session_state.chart_gallery.append({
                "figure": fig,
                "description": f"{chart_type} | {values} by {rows}"
            })
            st.success("Chart saved to gallery")

        # Download immediate HTML
        html_buffer = BytesIO()
        pio.write_html(fig, file=html_buffer, full_html=True, include_plotlyjs="cdn")

        col2.download_button(
            label="⬇️ Download Chart as HTML",
            data=html_buffer.getvalue(),
            file_name="pivot_chart.html",
            mime="text/html"
        )

    # -----------------------------------------
    # Correlation Heatmap
    # -----------------------------------------
    if show_corr and len(numeric_cols) > 1:

        st.subheader("🔥 Correlation Heatmap")

        corr_matrix = df[numeric_cols].corr()

        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0
            )
        )

        heatmap.update_layout(height=700)

        st.plotly_chart(heatmap, use_container_width=True)

        if st.button("💾 Save Heatmap to Gallery"):
            st.session_state.chart_gallery.append({
                "figure": heatmap,
                "description": "Correlation Heatmap"
            })
            st.success("Heatmap saved")

    # -----------------------------------------
    # Persistent Chart Gallery
    # -----------------------------------------
    if st.session_state.chart_gallery:

        st.divider()
        st.header("🗂 Chart Gallery")

        for i, item in enumerate(st.session_state.chart_gallery):

            st.subheader(f"{i+1}. {item['description']}")
            st.plotly_chart(item["figure"], use_container_width=True)

            colA, colB = st.columns(2)

            # Download
            html_buffer = BytesIO()
            pio.write_html(
                item["figure"],
                file=html_buffer,
                full_html=True,
                include_plotlyjs="cdn"
            )

            colA.download_button(
                label="Download HTML",
                data=html_buffer.getvalue(),
                file_name=f"chart_{i+1}.html",
                mime="text/html",
                key=f"download_{i}"
            )

            # Remove
            if colB.button("Remove from Gallery", key=f"remove_{i}"):
                st.session_state.chart_gallery.pop(i)
                st.rerun()

    # -----------------------------------------
    # Raw Data
    # -----------------------------------------
    if show_raw:
        st.subheader("🔍 Raw Data Preview")
        st.dataframe(df.head(1000), use_container_width=True)

else:
    st.info("Upload a Parquet file to begin.")