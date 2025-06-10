import streamlit as st
import pandas as pd
from Scripts.clustering import run_clustering

st.title("E-Commerce Customer Segmentation App")

uploaded_file = st.file_uploader("Upload your file in .csv format", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    features_df, radar_figs = run_clustering(df)

    st.write("Clustering Dataframe", features_df)

    st.write("Radar Charts for each cluster:")
    for fig in radar_figs:
        st.pyplot(fig)
