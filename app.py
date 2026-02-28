import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Great Race", layout="wide")

st.title("🏁 Great Race Dashboard")

st.sidebar.header("Settings")
name = st.sidebar.text_input("Enter your name:", "Racer")

st.write(f"Welcome, {name}!")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Lap Time", "1:23.45", "-0:05.2")

with col2:
    st.metric("Position", "1st", "+1")

with col3:
    st.metric("Top Speed", "245 mph", "+12 mph")

st.subheader("Race Progress")

# Sample data
race_data = pd.DataFrame({
    "Lap": range(1, 11),
    "Time": np.random.uniform(83, 88, 10),
    "Speed": np.random.uniform(240, 250, 10),
})

st.line_chart(race_data.set_index("Lap"))

st.subheader("Details")
st.dataframe(race_data, use_container_width=True)
