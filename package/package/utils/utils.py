import mlflow

def log_preprocessed_data(df, scaling_method, imputation_strategy, encoding_method, experiment_name: str) -> None:
    """
    Log the preprocessed data to an MLflow experiment.
    
    :param df: Preprocessed data as a pandas DataFrame.
    :param experiment_name: Name of the MLflow experiment to log the data to.
    """
    experiment_id = mlflow.set_experiment(experiment_name).experiment_id
    
    with mlflow.start_run(run_name='testing_run', experiment_id=experiment_id):
        mlflow.log_param("scaling_method", scaling_method)
        mlflow.log_param("imputation_strategy", imputation_strategy)
        mlflow.log_param("encoding_method", encoding_method)
        # df.to_csv("preprocessed_data.csv", index=False)
        # mlflow.log_artifact("preprocessed_data.csv")


def log_results(self, data, labels, experiment_name:str):
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
