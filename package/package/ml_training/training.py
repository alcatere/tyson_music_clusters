import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import mlflow
import mlflow.sklearn

class ClusteringModel:
    def __init__(self, model_name='kmeans', n_clusters=3):
        """
        Initialize the clustering model with specified parameters.
        
        :param model_name: The name of the clustering algorithm to use ('kmeans', 'dbscan', 'agglomerative').
        :param n_clusters: Number of clusters for clustering algorithms that require it (e.g., KMeans, Agglomerative).
        """
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.model = None
    
    def create_model(self):
        """
        Create a clustering model based on the specified model_name.
        """
        if self.model_name == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.model_name == 'dbscan':
            self.model = DBSCAN()
        elif self.model_name == 'agglomerative':
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        else:
            raise ValueError("Model name not recognized. Choose 'kmeans', 'dbscan', or 'agglomerative'.")
    
    def fit_predict(self, data):
        """
        Fit the clustering model and predict cluster labels.
        
        :param data: Data to fit the model on.
        :return: Predicted cluster labels.
        """
        return self.model.fit_predict(data)
    
    def log_results(self, data, labels, experiment_name):
        """
        Log clustering results to MLflow, including model, metrics, and visualizations.
        
        :param data: The data used for clustering.
        :param labels: Predicted cluster labels.
        :param experiment_name: Name of the MLflow experiment.
        """
        # Set the MLflow experiment
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            # Log model parameters
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("n_clusters", self.n_clusters if self.model_name != 'dbscan' else "N/A")
            
            # Calculate and log metrics
            if len(set(labels)) > 1:  # Metrics are undefined if there's only one cluster
                silhouette = silhouette_score(data, labels)
                calinski_harabasz = calinski_harabasz_score(data, labels)
                mlflow.log_metric("silhouette_score", silhouette)
                mlflow.log_metric("calinski_harabasz_score", calinski_harabasz)
            
            # Log the model
            mlflow.sklearn.log_model(self.model, "clustering_model")
            
            # Plot and log cluster visualization
            plt.figure(figsize=(8, 6))
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
            plt.title(f"{self.model_name} Clustering")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.colorbar(label="Cluster Label")
            plt.savefig("cluster_plot.png")
            mlflow.log_artifact("cluster_plot.png")
            plt.close()