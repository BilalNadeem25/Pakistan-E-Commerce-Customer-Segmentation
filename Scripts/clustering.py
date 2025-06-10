# Import necessary libraries
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


# Define the clustering function
def run_clustering(df, optimal_k=4):
    # Feature engineering...
    # Converting 'Order Date' to datetime format
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # Calculating recency for each customer
    # Set the reference date to one day after the last transaction
    latest_date = df["Order Date"].max()
    ref_date = latest_date + pd.Timedelta(days=1)

    # Find the number of days since last purchase for each customer
    recency_df = (
        df.groupby("Customer ID")
        .agg({"Order Date": lambda x: (ref_date - x.max()).days})
        .reset_index()
    )
    recency_df.columns = ["Customer ID", "Recency"]

    # Calculating frequency for each customer
    frequency_df = df.groupby("Customer ID").size().reset_index()
    frequency_df.columns = ["Customer ID", "Frequency"]

    # Calculating monetary for each customer
    monetary_df = df.groupby("Customer ID").agg({"Grand Total": "sum"}).reset_index()
    monetary_df.columns = ["Customer ID", "Monetary"]
    monetary_df["Monetary"] = monetary_df["Monetary"].round(2)

    # Calculating average quantity per order for each customer
    avg_quantity_df = (
        df.groupby("Customer ID")
        .agg({"Quantity": "sum", "Order Date": "count"})
        .reset_index()
    )
    avg_quantity_df["Average Quantity Per Order"] = (
        avg_quantity_df["Quantity"] / avg_quantity_df["Order Date"]
    ).round(2)
    avg_quantity_df = avg_quantity_df[["Customer ID", "Average Quantity Per Order"]]

    # Calculating average unit price for each customer
    avg_unitprice_df = (
        df.groupby("Customer ID").agg({"Unit Price": "mean"}).reset_index()
    )
    avg_unitprice_df = avg_unitprice_df.round(2)
    avg_unitprice_df.columns = ["Customer ID", "Average Unit Price"]

    # Calculating average discount % for each customer
    avg_dct_df = df.groupby("Customer ID").agg({"Discount %": "mean"}).reset_index()
    avg_dct_df = avg_dct_df.round(2)
    avg_dct_df.columns = ["Customer ID", "Average Discount %"]

    # Calculating the number of unique payment methods for each customer
    unique_payments_df = (
        df.groupby("Customer ID")
        .agg(PreferredCategories=("Payment Method", "nunique"))
        .reset_index()
    )
    unique_payments_df.columns = ["Customer ID", "Payment Method Diversity"]

    # Calculating the number of unique product categories for each customer
    unique_categories_df = (
        df.groupby("Customer ID")
        .agg(PreferredCategories=("Product Category", "nunique"))
        .reset_index()
    )
    unique_categories_df.columns = ["Customer ID", "Product Category Diversity"]

    # Merge all the sub-dataframes into a single dataframe
    dfs = [
        recency_df,
        frequency_df,
        monetary_df,
        avg_quantity_df,
        avg_unitprice_df,
        avg_dct_df,
        unique_payments_df,
        unique_categories_df,
    ]
    agg_df = reduce(
        lambda left, right: pd.merge(left, right, on="Customer ID", how="left"), dfs
    )

    # Extract customer ID and store it separately
    customer_ids = agg_df["Customer ID"]
    features_df = agg_df.drop(columns=["Customer ID"])

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    # Cluster the data
    kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, batch_size=1024)
    features_df["Cluster"] = kmeans.fit_predict(scaled_features)

    # Calculate the mean profile for each cluster
    cluster_profiles = features_df.groupby("Cluster").mean()

    # Radar chart setup
    df_norm = (cluster_profiles - cluster_profiles.min()) / (
        cluster_profiles.max() - cluster_profiles.min()
    )
    labels = list(df_norm.columns)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    colors = ["#2c2ecc", "#ff7f0e", "#1bb11b", "#e01e1e"]

    radar_figs = []
    for idx, cluster in enumerate(df_norm.index):
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        values = df_norm.loc[cluster].tolist()
        values += values[:1]
        current_color = colors[idx % len(colors)]
        ax.plot(angles, values, color=current_color, linewidth=2, label=cluster)
        ax.fill(angles, values, color=current_color, alpha=0.25)
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.linspace(0, 1.1, num=5))
        ax.yaxis.grid(True, linestyle="dashed", alpha=0.6)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.15), frameon=False)
        radar_figs.append(fig)
        plt.close(fig)  # Prevents display in non-interactive environments

    # Insert the 'Customer ID' column back into the features DataFrame
    features_df.insert(loc=0, column="Customer ID", value=customer_ids)

    # Map cluster labels to descriptive names
    cluster_name_map = {
        0: "Occasional Low Spender",
        1: "VIP Loyalist",
        2: "Luxury Deal Seeker",
        3: "Budget Bulk Buyer",
    }

    features_df["Cluster Name"] = features_df["Cluster"].replace(cluster_name_map)
    features_df = features_df.drop(columns=["Cluster"])

    return features_df, radar_figs
