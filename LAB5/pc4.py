import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.image import imread

# Load dataset
current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'Data', 'k_mean_points.npy')
loaded_points = np.load(file_path)

# Plot the dataset
plt.figure()
plt.scatter(loaded_points[:, 0], loaded_points[:, 1], label='Data Points')
plt.title('Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# K-means from scratch
def initialize_clusters(points: np.ndarray, k_clusters: int) -> np.ndarray:
    np.random.shuffle(points)
    return points[:k_clusters]

def calculate_metric(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points - centroid, axis=1)

def compute_distances(points: np.ndarray, centroids_points: np.ndarray) -> np.ndarray:
    return np.array([calculate_metric(points, centroid) for centroid in centroids_points])

def assign_centroids(distances: np.ndarray) -> np.ndarray:
    return np.argmin(distances, axis=0)

def calculate_objective(assigned_centroids: np.ndarray, distances: np.ndarray) -> float:
    min_distances = np.min(distances, axis=0)
    return np.sum(min_distances ** 2)

def calculate_new_centroids(points: np.ndarray, assigned_centroids: np.ndarray, k_clusters: int) -> np.ndarray:
    return np.array([points[assigned_centroids == k].mean(axis=0) for k in range(k_clusters)])

def fit(points: np.ndarray, k_clusters: int, n_of_iterations: int, error: float = 0.001) -> tuple:
    centroids = initialize_clusters(points, k_clusters)
    for _ in range(n_of_iterations):
        distances = compute_distances(points, centroids)
        assigned_centroids = assign_centroids(distances)
        new_centroids = calculate_new_centroids(points, assigned_centroids, k_clusters)

        if np.all(centroids == new_centroids) or np.linalg.norm(centroids - new_centroids) < error:
            break
        centroids = new_centroids

    last_objective = calculate_objective(assigned_centroids, distances)
    return centroids, last_objective

# Elbow method
def elbow_method(points: np.ndarray, max_k: int) -> None:
    k_all = range(2, max_k + 1)
    all_objective = []

    for k in k_all:
        _, objective = fit(points, k, 100)
        all_objective.append(objective)

    plt.figure()
    plt.plot(k_all, all_objective, marker='o')
    plt.xlabel('K clusters')
    plt.ylabel('Sum of squared distance')
    plt.title('Elbow Method For Optimal k')
    plt.show()

# Run the elbow method
elbow_method(loaded_points, 10)

# Fit K-means and plot centroids
k_clusters = 3
centroids, _ = fit(loaded_points, k_clusters, 100)

plt.figure()
plt.scatter(loaded_points[:, 0], loaded_points[:, 1], label='Data Points', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', label='Centroids', marker='X')
plt.title('K-means Clustering with Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Image compression
def compress_image(image: np.ndarray, number_of_colours: int) -> np.ndarray:
    height, width, channels = image.shape
    image_2d = image.reshape(-1, channels)

    kmeans = KMeans(n_clusters=number_of_colours)
    kmeans.fit(image_2d)
    compressed_2d = kmeans.cluster_centers_[kmeans.labels_]

    compressed_image = compressed_2d.reshape(height, width, channels).astype(np.uint8)
    return compressed_image

# Load image
loaded_image = imread(os.path.join(current_directory, 'Data', 'fish.jpg'))

# Compress the image
compressed_image = compress_image(loaded_image, 10)

# Plot the original and compressed images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(loaded_image)

plt.subplot(1, 2, 2)
plt.title('Compressed Image 10 colors)')
plt.imshow(compressed_image)

plt.show()
