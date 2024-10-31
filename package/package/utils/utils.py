import mlflow

from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt


def log_preprocessed_data(df, model_name, scaling_method, imputation_strategy, encoding_method, experiment_name: str) -> None:
    """
    Log the preprocessed data to an MLflow experiment.
    
    :param df: Preprocessed data as a pandas DataFrame.
    :param experiment_name: Name of the MLflow experiment to log the data to.
    """
    experiment_id = mlflow.set_experiment(experiment_name).experiment_id
    
    with mlflow.start_run(run_name=model_name, experiment_id=experiment_id) as run:
        mlflow.log_param("scaling_method", scaling_method)
        mlflow.log_param("imputation_strategy", imputation_strategy)
        mlflow.log_param("encoding_method", encoding_method)
        # df.to_csv("preprocessed_data.csv", index=False)
        # mlflow.log_artifact("preprocessed_data.csv")
    return run.info.run_id

def log_results(data, labels, model, model_name, experiment_name:str, run_id:int, signature, n_clusters):
    """
    Log clustering results to MLflow, including model, metrics, and visualizations.
    
    :param data: The data used for clustering.
    :param labels: Predicted cluster labels.
    :param experiment_name: Name of the MLflow experiment.
    """
    # Set the MLflow experiment
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_id=run_id):
        # Log model parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_clusters", n_clusters if model_name != 'dbscan' else "N/A")
        
        # Calculate and log metrics
        if len(set(labels)) > 1:  # Metrics are undefined if there's only one cluster
            silhouette = silhouette_score(data, labels)
            calinski_harabasz = calinski_harabasz_score(data, labels)
            mlflow.log_metric("silhouette_score", silhouette)
            mlflow.log_metric("calinski_harabasz_score", calinski_harabasz)
        
        # Log the model
        mlflow.sklearn.log_model(model, "clustering_model", signature=signature)
        
        # Plot and log cluster visualization
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
        plt.title(f"{model_name} Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plot = plt.colorbar(label="Cluster Label")
        # plt.savefig("cluster_plot.png")
        # mlflow.log_artifact("cluster_plot.png")
        mlflow.log_image(plot)
        plt.close()
