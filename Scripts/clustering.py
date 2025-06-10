import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = pd.read_csv(
    "C:/GitHub/projects/Pakistan E-Commerce Customer Segmentation/Pakistan-E-Commerce-Customer-Segmentation/Sources/Pakistan Largest Ecommerce Dataset - Cleaned.csv"
)
df = data.copy()
# print(df.head())

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
avg_unitprice_df = df.groupby("Customer ID").agg({"Unit Price": "mean"}).reset_index()
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
scaled_features_df = pd.DataFrame(scaled_features, columns=features_df.columns)
# print(scaled_features_df.head())

# Visualizing the correlation matrix of the features
plt.figure(figsize=(12, 8))
sns.heatmap(scaled_features_df.corr(), annot=True, cmap="YlOrBr", fmt=".2f")
plt.title("Correlation Matrix of Features")
# plt.show()

# Using the Elbow Method to determine the optimal number of clusters
wcss = []
K = range(1, 9)

for k in K:
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, wcss, "bo-", markersize=6)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("Elbow Method")

for y in wcss:
    plt.axhline(y=y, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)

# plt.show()

# Define the optimal number of clusters based on the elbow method
optimal_k = 4

# Clustering
kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, batch_size=1024)
features_df["Cluster"] = kmeans.fit_predict(scaled_features)

# Evaluation
score = silhouette_score(scaled_features, features_df["Cluster"])
print(f"Silhouette Score: {score:.2f}")

cluster_labels = features_df["Cluster"]
sample_silhouette_values = silhouette_samples(scaled_features, cluster_labels)

# Create silhouette plot
n_clusters = len(np.unique(cluster_labels))
fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10

for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / n_clusters)
    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}")
    y_lower = y_upper + 10

ax.set_title("Silhouette Plot for Mini Batch K-Means Clustering (k=4)")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")

ax.axvline(x=score, color="red", linestyle="--", label="Avg Silhouette Score")
ax.set_yticks([])
ax.set_xlim([-0.1, 1])
ax.legend()
plt.tight_layout()
# plt.show()

# Visualizing the clusters using PCA
features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Average Quantity Per Order",
    "Average Unit Price",
    "Average Discount %",
    "Payment Method Diversity",
    "Product Category Diversity",
]

X = scaled_features_df[features]

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7
)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Customer Segments (PCA Projection)")
plt.legend(*scatter.legend_elements(), title="Cluster")
# plt.show()

# Insert the 'Customer ID' column back into the features DataFrame
features_df.insert(loc=0, column="Customer ID", value=customer_ids)

# Save the customer segments to a CSV file
features_df.to_csv(
    "C:/GitHub/projects/Pakistan E-Commerce Customer Segmentation/Pakistan-E-Commerce-Customer-Segmentation/Sources/Customer Segments.csv",
    index=False,
)

# Calculate the mean profile for each cluster
cluster_profiles = features_df.groupby("Cluster").mean()
cluster_profiles.drop(columns=["Customer ID"], inplace=True)
print(cluster_profiles)

# Plot a radar chart for each cluster profile
# Normalize the data for visual ease (min-max scaling)
df_norm = (cluster_profiles - cluster_profiles.min()) / (
    cluster_profiles.max() - cluster_profiles.min()
)

# Prepare the radar chart parameters
labels = list(df_norm.columns)
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

colors = ["#2c2ecc", "#ff7f0e", "#1bb11b", "#e01e1e"]

for idx, cluster in enumerate(df_norm.index):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Get the values for the current cluster and complete the loop
    values = df_norm.loc[cluster].tolist()
    values += values[:1]

    # Select the color for this cluster
    current_color = colors[idx % len(colors)]

    # Plot and fill the radar chart using the unique color
    ax.plot(angles, values, color=current_color, linewidth=2, label=cluster)
    ax.fill(angles, values, color=current_color, alpha=0.25)

    # Set the labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

    # Display the plot
    plt.show()
