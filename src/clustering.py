from sklearn.cluster import KMeans, DBSCAN

def run_kmeans(X_scaled, k=5):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    return labels

def run_dbscan(X_scaled, eps=0.9, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    return labels
