from src.data_loader import load_data
from src.feature_engineering import create_customer_features
from src.clustering import run_kmeans, run_dbscan
from src.evaluation import evaluate_clusters
from src.visualization import plot_pca

# Load data
df = load_data("data/raw/transactions.csv")

# Feature engineering
customer_features, X_scaled = create_customer_features(df)

# Clustering
customer_features['kmeans_cluster'] = run_kmeans(X_scaled)
customer_features['dbscan_cluster'] = run_dbscan(X_scaled)

# Evaluation
score = evaluate_clusters(X_scaled, customer_features['dbscan_cluster'])
print("Silhouette Score (DBSCAN):", score)

# Visualization
plot_pca(X_scaled, customer_features['dbscan_cluster'])

# Marketing recommendations
recommendations = {
    0: "Offer weekend bulk discounts & bundles",
    1: "Send late-night flash offers after 10 PM",
    2: "Target with coupon-based campaigns",
    -1: "VIP or anomaly customers – personalized offers"
}

customer_features['marketing_action'] = customer_features['dbscan_cluster'].map(recommendations)

# Save output
customer_features.to_csv(
    "data/processed/customer_micro_trend_segments.csv",
    index=False
)
