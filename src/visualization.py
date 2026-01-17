import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_pca(X_scaled, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Customer Micro-Trend Clusters")
    plt.colorbar(label="Cluster")
    plt.show()
