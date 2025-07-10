# Import necessary libraries
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


# Define the clustering function
def run_clustering(df, k=4):
    # Feature Engineering...

    # Convert 'Order Date' to datetime format
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # Set the reference date to one day after the last transaction
    latest_date = df["Order Date"].max()
    ref_date = latest_date + pd.Timedelta(days=1)

    # Using the reference date, calculate recency for each customer
    recency_df = (
        df.groupby("Customer ID")
        .agg({"Order Date": lambda x: (ref_date - x.max()).days})
        .reset_index()
    )
    recency_df.columns = ["Customer ID", "Recency"]

    # Calculate frequency for each customer
    frequency_df = df.groupby("Customer ID").size().reset_index()
    frequency_df.columns = ["Customer ID", "Frequency"]

    # Calculate monetary for each customer
    monetary_df = df.groupby("Customer ID").agg({"Grand Total": "sum"}).reset_index()
    monetary_df.columns = ["Customer ID", "Monetary"]
    monetary_df["Monetary"] = monetary_df["Monetary"].round(2)

    # Calculate average quantity per order for each customer
    avg_quantity_df = (
        df.groupby("Customer ID")
        .agg({"Quantity": "sum", "Order Date": "count"})
        .reset_index()
    )
    avg_quantity_df["Average Quantity Per Order"] = (
        avg_quantity_df["Quantity"] / avg_quantity_df["Order Date"]
    ).round(2)
    avg_quantity_df = avg_quantity_df[["Customer ID", "Average Quantity Per Order"]]

    # Calculate average unit price for each customer
    avg_unitprice_df = (
        df.groupby("Customer ID").agg({"Unit Price": "mean"}).reset_index()
    )
    avg_unitprice_df = avg_unitprice_df.round(2)
    avg_unitprice_df.columns = ["Customer ID", "Average Unit Price"]

    # Calculate average discount % for each customer
    avg_dct_df = df.groupby("Customer ID").agg({"Discount %": "mean"}).reset_index()
    avg_dct_df = avg_dct_df.round(2)
    avg_dct_df.columns = ["Customer ID", "Average Discount %"]

    # Calculate the number of unique payment methods for each customer
    unique_payments_df = (
        df.groupby("Customer ID")
        .agg(PreferredCategories=("Payment Method", "nunique"))
        .reset_index()
    )
    unique_payments_df.columns = ["Customer ID", "Payment Method Diversity"]

    # Calculate the number of unique product categories for each customer
    unique_categories_df = (
        df.groupby("Customer ID")
        .agg(PreferredCategories=("Product Category", "nunique"))
        .reset_index()
    )
    unique_categories_df.columns = ["Customer ID", "Product Category Diversity"]

    # Create a list of the sub-dataframes and perform left join cumulatively based on 'Customer ID'
    # to create the final aggregated dataframe
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

    # Feature Scaling...

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    # Clustering...

    # Cluster the data
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
    features_df["Cluster"] = kmeans.fit_predict(scaled_features)

    # Calculate the mean profiles for each cluster
    cluster_profiles = features_df.groupby("Cluster").mean()

    # Visualization...

    # Normalize the cluster profiles for radar chart visualization
    radar_scaler = MinMaxScaler()
    scaled_profiles = pd.DataFrame(
        radar_scaler.fit_transform(cluster_profiles),
        index=cluster_profiles.index,
        columns=cluster_profiles.columns,
    )

    # Set up the radar chart attrubutes
    labels = list(scaled_profiles.columns)
    label_count = len(labels)
    angles = np.linspace(0, 2 * np.pi, label_count, endpoint=False).tolist()

    # Append the first angle to the end to close the radar chart
    angles += angles[:1]
    colors = ["#2c2ecc", "#ff7f0e", "#1bb11b", "#e01e1e"]

    # Create a radar chart for each cluster
    radar_figs = []
    for idx, cluster in enumerate(scaled_profiles.index):
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # Retrieve the values for the current cluster and close the radar chart
        values = scaled_profiles.loc[cluster].tolist()
        values += values[:1]

        # Select color for the current cluster using modulo indexing
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
        plt.close(fig)

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
