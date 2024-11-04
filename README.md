# Clustering Model Evaluation with MLflow

This project aims to evaluate multiple clustering models on Tyler The Producer Dataset using various metrics and to track model performance with MLflow. The goal is to identify the optimal number of clusters for each model, providing insights into cluster quality and stability.

## Project Overview

In this project, we explore different clustering models and log their performance metrics across a range of cluster numbers to MLflow for easy comparison. The following steps are performed:
1. Data preprocessing (handling categorical features and scaling).
2. Applying clustering models (e.g., KMeans, Agglomerative Clustering).
3. Logging metrics such as **Silhouette Score** and **Calinski-Harabasz Index** for each model and cluster count.
4. Visualizing metrics across cluster counts to aid in identifying the optimal number of clusters.
