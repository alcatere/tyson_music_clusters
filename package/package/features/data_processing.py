import pandas as pd


from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreprocessor:
    def __init__(self, scaling_method='standard', imputation_strategy='mean', encoding_method='onehot'):
        """
        Initialize the DataPreprocessor with chosen scaling method, imputation strategy, and encoding method.
        
        :param scaling_method: Scaling technique - 'standard' or 'minmax'
        :param imputation_strategy: Strategy for imputing missing values - 'mean', 'median', 'most_frequent'
        :param encoding_method: Encoding method for categorical features - 'onehot' or 'label'
        """
        self.scaling_method = scaling_method
        self.imputation_strategy = imputation_strategy
        self.encoding_method = encoding_method
        self.scaler = None
        self.imputer = SimpleImputer(strategy=self.imputation_strategy)
        self.encoder = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        :param file_path: Path to the CSV file.
        :return: Loaded data as a pandas DataFrame.
        """
        return pd.read_csv(file_path)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data by imputing missing values, encoding categorical features, and scaling features.
        
        :param df: Raw data as a pandas DataFrame.
        :return: Preprocessed data as a pandas DataFrame.
        """
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

        # Define transformers for numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.imputation_strategy)),
            ('scaler', StandardScaler() if self.scaling_method == 'standard' else MinMaxScaler())
        ])

        if self.encoding_method == 'onehot':
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False))  # Disable sparse output for DataFrame compatibility
            ])
        elif self.encoding_method == 'label':
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', LabelEncoder())
            ])
        else:
            raise ValueError("Encoding method not recognized. Choose 'onehot' or 'label'.")

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ], remainder='drop')
        
        # Fit and transform the data
        processed_data = preprocessor.fit_transform(df)

        # Handle column names after encoding
        cat_features = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
        all_columns = numeric_cols.tolist() + list(cat_features)
        
        processed_df = pd.DataFrame(processed_data, columns=all_columns)

        return processed_df