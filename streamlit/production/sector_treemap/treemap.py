import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="FUREcast Treemap", layout="wide")

st.title("Sector → Holdings Drill-Down (SPLG)")

csv_path = "data_out/level4_treemap_nodes.csv"
df = pd.read_csv(csv_path)

st.caption("Size = SPLG Weight (%). Color = Per-stock Daily % Change. Hover shows KPIs.")

# Controls
color_metric = st.selectbox("Color by", ["DailyChangePct", "PE", "Beta", "DividendYield"], index=0)
range_center = 0 if color_metric == "DailyChangePct" else None

# Plotly treemap
hover_cols = ["Weight (%)", "DailyChangePct", "PE", "Beta", "DividendYield"]
fig = px.treemap(
    df,
    path=["Sector", "Company"],
    values="Weight (%)",
    color=color_metric,
    color_continuous_midpoint=range_center,
    color_continuous_scale="RdYlGn" if color_metric == "DailyChangePct" else "Viridis",
    hover_data=hover_cols,
    title="SPLG Sector → Company Treemap"
)

st.plotly_chart(fig, use_container_width=True)

# Optional: sector summary table
st.subheader("Sector Summary (Weighted by SPLG)")
summ = df.groupby("Sector").agg({
    "Weight (%)": "sum",
    "DailyChangePct": "mean",
    "PE": "mean",
    "Beta": "mean",
    "DividendYield": "mean"
}).reset_index().sort_values("Weight (%)", ascending=False)
st.dataframe(summ, use_container_width=True)
