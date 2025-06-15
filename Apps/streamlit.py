import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from Scripts.clustering import run_clustering

st.title("E-Commerce Customer Segmentation App")

uploaded_file = st.file_uploader("Upload your file in .csv format", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    features_df, radar_figs = run_clustering(df)

    st.write("Clustering Dataframe", features_df)

    # Add a search box for Customer ID filtering
    customer_id = st.text_input("Search for Customer ID")
    if customer_id:
        filtered_df = features_df[features_df["Customer ID"].astype(str) == customer_id]
        if filtered_df.empty:
            st.warning("This customer ID does not exist")
        else:
            st.write(
                f"Filtered Results for Customer ID '{customer_id}':",
                filtered_df,
            )
    else:
        st.write("Showing all customers.")

    st.write("Radar Charts for each cluster:")
    # Custom descriptions for each segment
    segment_descriptions = {
        "Occasional Low Spender": "Customers that only shop occasionally, spending conservatively when they do. They are not loyal but still make purchases from time to time.",
        "VIP Loyalist": "High-value, loyal customers who purchase frequently and spend the most. They are likely to respond well to loyalty programs and exclusive offers.",
        "Luxury Deal Seeker": "Customers who prefer high-value, branded items but are motivated by discounts. They often look for the best deals on premium products.",
        "Budget Bulk Buyer": "New customers who buy low-value items in large quantities. They may be stocking up on essentials or frequently used items.",
    }
    cluster_names = (
        features_df["Cluster Name"].unique()
        if "Cluster Name" in features_df.columns
        else [f"Cluster {i}" for i in range(len(radar_figs))]
    )
    for idx, fig in enumerate(radar_figs):
        with st.container():
            cluster_name = (
                cluster_names[idx] if idx < len(cluster_names) else f"Cluster {idx}"
            )
            st.markdown(f"### {cluster_name}")
            description = segment_descriptions.get(
                cluster_name,
                "This radar chart visualizes the average profile of customers in this segment across all features.",
            )
            st.markdown(f"{description}")
            st.pyplot(fig)
