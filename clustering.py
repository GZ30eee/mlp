import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Streamlit App
def main():
    st.set_page_config(page_title="K-Means Clustering", layout="wide")

    # Main title
    st.title("🔍 K-Means Clustering Visualization")
    st.write("This app demonstrates K-Means clustering on randomly generated data.")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, 50)
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
    cluster_std = st.sidebar.slider("Cluster Standard Deviation", 0.1, 2.0, 0.60, 0.1)
    random_state = st.sidebar.slider("Random State", 0, 100, 0)

    # Generate dataset
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=cluster_std, random_state=random_state)

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_

    # Layout: Two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Cluster Visualization")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.7, edgecolors='k')
        ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label="Centroids")
        ax.set_title("K-Means Clustering")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("📈 Cluster Information")
        st.write(f"🔹 **Inertia (Sum of Squared Distances):** `{kmeans.inertia_:.2f}`")
        st.write(f"🔹 **Cluster Centers:**")
        df_centers = pd.DataFrame(centers, columns=["Feature 1", "Feature 2"])
        st.dataframe(df_centers)

        # Download dataset button
        df_clustered = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
        df_clustered["Cluster"] = y_kmeans
        csv = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Clustered Data", data=csv, file_name="clustered_data.csv", mime="text/csv")

    # Elbow Method (optional)
    st.subheader("📌 Elbow Method for Optimal Clusters")
    distortions = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K_range, distortions, marker='o', linestyle='-', color='b')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia (Distortion)")
    ax.set_title("Elbow Method to Determine Optimal K")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
