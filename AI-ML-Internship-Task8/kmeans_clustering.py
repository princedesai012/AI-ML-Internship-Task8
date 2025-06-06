import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import os

# Step 1: Load and preprocess the dataset
def load_data(file_path):
    try:
        print(f"Attempting to load file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist in the specified directory.")
        df = pd.read_csv(file_path)
        # Select relevant features for clustering
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return df, X, X_scaled
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'Mall_Customers.csv' is in the same directory as this script or update the file_path variable.")
        raise
    except KeyError as e:
        print(f"Error: {e}")
        print("The dataset does not have the expected columns 'Annual Income (k$)' or 'Spending Score (1-100)'. Please check the dataset structure.")
        raise

# Step 2: Elbow Method to find optimal K
def elbow_method(X_scaled):
    inertias = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot Elbow Curve
    plt.figure(figsize=(8, 6))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.savefig('elbow_plot.png')
    plt.close()

# Step 3: Fit K-Means and assign cluster labels
def fit_kmeans(X_scaled, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return kmeans, cluster_labels

# Step 4: Visualize clusters using PCA
def visualize_clusters(df, X_scaled, cluster_labels):
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a DataFrame with PCA components and cluster labels
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels
    
    # Plot clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='deep')
    plt.title('Customer Segments (PCA-reduced)')
    plt.savefig('cluster_plot.png')
    plt.close()

# Step 5: Evaluate clustering using Silhouette Score
def evaluate_clustering(X_scaled, cluster_labels):
    score = silhouette_score(X_scaled, cluster_labels)
    return score

# Main function to run the clustering pipeline
def main():
    # Define file path
    file_path = 'Mall_Customers.csv'  # Update this if the file is in a different location, e.g., r'C:\path\to\Mall_Customers.csv'
    
    # Load data
    df, X, X_scaled = load_data(file_path)
    
    # Run Elbow Method
    elbow_method(X_scaled)
    
    # Fit K-Means (assuming K=5 based on typical Elbow Method results for this dataset)
    kmeans, cluster_labels = fit_kmeans(X_scaled, n_clusters=5)
    
    # Visualize clusters
    visualize_clusters(df, X_scaled, cluster_labels)
    
    # Evaluate clustering
    silhouette = evaluate_clustering(X_scaled, cluster_labels)
    print(f'Silhouette Score: {silhouette:.3f}')
    
    # Save results to a text file
    with open('results.txt', 'w') as f:
        f.write(f'Silhouette Score: {silhouette:.3f}\n')
        f.write(f'Number of clusters: 5\n')

if __name__ == '__main__':
    main()