from package.ml_training.training import ClusteringModel
from package.features.data_processing import DataPreprocessor
from package.utils.utils import log_preprocessed_data
from package.utils.utils import log_results

from mlflow.models.signature import infer_signature

from pandas.core.common import flatten



if __name__ == '__main__':
    DATA_URL = 'https://storage.googleapis.com/kagglesdsdata/datasets/5890222/9708274/Tyler%20The%20Creator%20Dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241029%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241029T201158Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=654147980d111273ff19d55bfd20c7a3e1298be83b3043690816ce2f0dfe4a8a579815039a5519210b6c27f50089f1dfccab80c499fbf1fc0d91ff949394300eaf5684b524703efe543f2cd882b41f9f32ab5dd18c9a7f27f08bf3b7857491e071b93d3a73720df29e003ade70abf61bee4717927bcb00f5fbe72985f1263154d4bbf96b3e4c92dcf4b08a96ae689fe578d1d134e23a1c886c65e63f3eb0692b19a9fd9392780cbda48a88585092f2d62868a8ce1b1529dca40afd6f33fe8ccc0f07b9eec87ee7013c9029ebd694b5b863b73dca514517ab930b73b2546bd8aa8fda043ab02627e1d090f1becd0a42696ac43387c9b54afe209413eed2bc1c4c'
    
    experiment_name = "Clustering Experiment"
    # run_name = 'clustering run'

    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(DATA_URL)
    process_df = preprocessor.preprocess_data(df)



    
    kmeans_models = [ClusteringModel(model_name='kmeans', n_clusters=cluster) for cluster in range(3,10)]
    agglomerative_models = [ClusteringModel(model_name='agglomerative', n_clusters=cluster) for cluster in range(3,10)]


    # List of models to try
    models = [
        kmeans_models,
        ClusteringModel(model_name='dbscan'),
        agglomerative_models
    ]

    models = list(flatten(models))

    for model in models:
        model.create_model()
        labels = model.fit_predict(process_df)
        signature = infer_signature(process_df, labels)
        model_name = f'{model.model_name}_{model.n_clusters}' if model.model_name != 'dbscan' else 'dbscan'

        run_id = log_preprocessed_data(process_df,
                        model_name,
                        preprocessor.scaling_method,
                        preprocessor.imputation_strategy,
                        preprocessor.encoding_method,
                        experiment_name)
        
        log_results(process_df, 
                    labels, 
                    model, 
                    model_name, 
                    experiment_name, 
                    run_id,
                    signature, 
                    model.n_clusters)