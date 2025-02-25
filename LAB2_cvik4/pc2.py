import numpy as np
import matplotlib.pyplot as plt

def initialize_clusters(points: np.ndarray, k_clusters: int) -> np.ndarray:
    """
    Initializes and returns k random centroids from the given dataset.
    """
    shuffled_points = np.random.permutation(points)
    return shuffled_points[:k_clusters]

def calculate_metric(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance between each point and a given centroid.
    """
    return np.linalg.norm(points - centroid, axis=1)

def compute_distances(points: np.ndarray, centroids_points: np.ndarray) -> np.ndarray:
    """
    Computes and returns the distances from each point to each centroid.
    """
    return np.array([calculate_metric(points, centroid) for centroid in centroids_points])

def assign_centroids(distances: np.ndarray) -> np.ndarray:
    """
    Assigns each data point to the closest centroid.
    """
    return np.argmin(distances, axis=0)

def calculate_objective(assigned_centroids: np.ndarray, distances: np.ndarray) -> float:
    """
    Computes the objective function value based on distances and cluster assignments.
    """
    return np.sum([distances[cluster, i] for i, cluster in enumerate(assigned_centroids)])

def calculate_new_centroids(points: np.ndarray, assigned_centroids: np.ndarray, k_clusters: int) -> np.ndarray:
    """
    Computes new centroids by averaging all data points assigned to each cluster.
    """
    return np.array([points[assigned_centroids == k].mean(axis=0) for k in range(k_clusters)])

def fit(points: np.ndarray, k_clusters: int, max_iterations: int, error_threshold: float = 0.001) -> tuple:
    """
    Fits the k-means clustering model to the dataset.
    """
    centroids = initialize_clusters(points, k_clusters)
    prev_objective = float('inf')
    
    for _ in range(max_iterations):
        distances = compute_distances(points, centroids)
        assigned_centroids = assign_centroids(distances)
        objective_value = calculate_objective(assigned_centroids, distances)
        
        if abs(prev_objective - objective_value) < error_threshold:
            break
        
        centroids = calculate_new_centroids(points, assigned_centroids, k_clusters)
        prev_objective = objective_value
    
    return centroids, objective_value

# Load dataset
loaded_points = np.load('Data/k_mean_points.npy')
k = 3
centroids, final_objective = fit(loaded_points, k, 100)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(loaded_points[:, 0], loaded_points[:, 1], c='blue', alpha=0.5, label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Final Centroids')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("K-Means Clustering Result")
plt.legend()
plt.grid()
plt.show()
