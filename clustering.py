import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Streamlit App
def main():
    st.title("K-Means Clustering Visualization")

    # Sidebar for user controls
    st.sidebar.header("Settings")
    n_samples = st.sidebar.slider("Number of Samples", min_value=100, max_value=1000, value=300, step=50)
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=4)
    cluster_std = st.sidebar.slider("Cluster Standard Deviation", min_value=0.1, max_value=2.0, value=0.60, step=0.1)
    random_state = st.sidebar.slider("Random State", min_value=0, max_value=100, value=0)

    # Generate random data points
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=cluster_std, random_state=random_state)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_

    # Plot the data points and centroids
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='X')
    ax.set_title("K-Means Clustering")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    # Display the plot
    st.pyplot(fig)

if __name__ == "__main__":
    main()
