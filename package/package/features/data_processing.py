import pandas as pd


class DataPreprocessor():
    def __init__():
        pass

    def load_data() -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        :param file_path: Path to the CSV file.
        :return: Loaded data as a pandas DataFrame.
        """
        data_url = 'https://storage.googleapis.com/kagglesdsdata/datasets/5890222/9708274/Tyler%20The%20Creator%20Dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241029%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241029T201158Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=654147980d111273ff19d55bfd20c7a3e1298be83b3043690816ce2f0dfe4a8a579815039a5519210b6c27f50089f1dfccab80c499fbf1fc0d91ff949394300eaf5684b524703efe543f2cd882b41f9f32ab5dd18c9a7f27f08bf3b7857491e071b93d3a73720df29e003ade70abf61bee4717927bcb00f5fbe72985f1263154d4bbf96b3e4c92dcf4b08a96ae689fe578d1d134e23a1c886c65e63f3eb0692b19a9fd9392780cbda48a88585092f2d62868a8ce1b1529dca40afd6f33fe8ccc0f07b9eec87ee7013c9029ebd694b5b863b73dca514517ab930b73b2546bd8aa8fda043ab02627e1d090f1becd0a42696ac43387c9b54afe209413eed2bc1c4c'
        
        dataset = pd.read_csv(data_url)

        print('Extraction succesfull')

        return dataset

    def preprocess_data(self, df) -> pd.DataFrame:
        """
        Preprocess data by imputing missing values and scaling features.
        
        :param df: Raw data as a pandas DataFrame.
        :return: Preprocessed data as a pandas DataFrame.
        """
        # Impute missing values
        df_imputed = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
        
        # Choose and apply the scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Scaling method not recognized. Choose 'standard' or 'minmax'.")
        
        scaled_data = pd.DataFrame(self.scaler.fit_transform(df_imputed), columns=df.columns)
        
        return scaled_data
    
    def log_preprocessed_data(self, df, experiment_name):
        """
        Log the preprocessed data to an MLflow experiment.
        
        :param df: Preprocessed data as a pandas DataFrame.
        :param experiment_name: Name of the MLflow experiment to log the data to.
        """
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_param("scaling_method", self.scaling_method)
            mlflow.log_param("imputation_strategy", self.imputation_strategy)
            mlflow.log_artifact(df.to_csv(index=False), "preprocessed_data.csv")
            mlflow.end_run()