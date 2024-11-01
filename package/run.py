from package.ml_training.training import ClusteringModel
from package.features.data_processing import DataPreprocessor
from package.utils.utils import log_preprocessed_data
from package.utils.utils import log_results

from mlflow.models.signature import infer_signature

import matplotlib.pyplot as plt

from pandas.core.common import flatten



if __name__ == '__main__':
    DATA_URL = 'https://storage.googleapis.com/kagglesdsdata/datasets/5890222/9781067/Tyler%20The%20Creator%20Dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241101%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241101T214729Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4e3531616180abb99172b7cd01396b0c07c55fd5769b8502e6ac32367edf479eeaa484b5ecd61a403b289d0698a4801f7b9d4b151c5130beefe020f220e7d878da2fa63eee18ed98476152f1354f55e0a77e8969bbaee951880778593781dd25f175f4e1f634a491d240f3f9d5fe62ef722cf17d2849008082b6d62af6a64a080cc898e783c8160aecc4cf97f6af6f14ceb510081b6d480bbd8aa5dfa2c7f382a52cf5d3503abdb05cab282f50b8dc54c1881bbb69c2fee5e99e180838c720a9032328f747036d3ea15ce7f00c72827aed39a24f17a8a4194aa5e4edda46116314049f32b4024668c8cb5523267095988fe3e08f460bd0dda2cf358e05370a7f'
    experiment_name = "Clustering Experiment"
    # run_name = 'clustering run'

    # Nums of clusters
    cluster_range = (3, 20)

    # Preprocess Data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(DATA_URL)
    process_df = preprocessor.preprocess_data(df)
    
    # Make models with different clusters
    kmeans_models = [ClusteringModel(model_name='kmeans', n_clusters=cluster) for cluster in range(cluster_range[0], cluster_range[1])]
    agglomerative_models = [ClusteringModel(model_name='agglomerative', n_clusters=cluster) for cluster in range(cluster_range[0], cluster_range[1])]


    # List of models to try
    models = [
        kmeans_models,
        ClusteringModel(model_name='dbscan'),
        agglomerative_models
    ]

    # Flatten the list
    models = list(flatten(models))

    silhouette_scores = {model.model_name: [] for model in models}
    calinski_scores = {model.model_name: [] for model in models}

    # Log every model
    for model in models:
        model.create_model()
        labels = model.fit_predict(process_df)
        signature = infer_signature(process_df, labels,)
        model_name = f'{model.model_name}_{model.n_clusters}' if model.model_name != 'dbscan' else 'dbscan'

        run_id = log_preprocessed_data(process_df,
                        model_name,
                        preprocessor.scaling_method,
                        preprocessor.imputation_strategy,
                        preprocessor.encoding_method,
                        experiment_name)
        
        silhouette_score, calinski_harabasz_score = log_results(process_df, 
                    labels, 
                    model, 
                    model_name, 
                    experiment_name, 
                    run_id,
                    signature, 
                    model.n_clusters)
        
        print(f'\n Finish to log the model: {model_name}')
        
        silhouette_scores[model.model_name].append(silhouette_score)
        calinski_scores[model.model_name].append(calinski_harabasz_score)

    # Plot the results
   
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for model_name in silhouette_scores:
        cluster_counts = list(range(cluster_range[0], cluster_range[1])) if model_name != 'dbscan' else [0]
        # print(model_name)
        # print(cluster_counts)
        # print(silhouette_scores[model_name])
        axes[0].plot(cluster_counts, silhouette_scores[model_name], marker='o', label=model_name)
        axes[1].plot(cluster_counts, calinski_scores[model_name], marker='o', label=model_name)
    
    # Customize plots
    axes[0].set_title("Silhouette Score by Number of Clusters")
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_title("Calinski-Harabasz Score by Number of Clusters")
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Calinski-Harabasz Score")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('./images/scores.png')
    # plt.show()