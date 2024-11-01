from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
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
