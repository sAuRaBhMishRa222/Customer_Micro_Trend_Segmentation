from sklearn.metrics import silhouette_score

def evaluate_clusters(X, labels):
    mask = labels != -1
    if len(set(labels[mask])) > 1:
        return silhouette_score(X[mask], labels[mask])
    return -1
