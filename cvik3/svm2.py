# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random

# Generate Gaussian blobs (2D data)
random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.3, center_box=(4, 4))

# Train One-Class SVM
SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
SVMmodelOne.fit(x)

# Predict anomalies (outliers)
pred = SVMmodelOne.predict(x)
anom_index = where(pred == -1)
outliers = x[anom_index]

# Calculate anomaly scores
scores = SVMmodelOne.score_samples(x)

# Calculate the 1% quantile for threshold
thresh = quantile(scores, 0.01)
outliers_quantile = x[scores <= thresh]

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot the original data
axs[0].scatter(x[:, 0], x[:, 1])
axs[0].set_title("Generated Data Points")
axs[0].axis('equal')

# Plot the One-Class SVM outliers
axs[1].scatter(x[:, 0], x[:, 1])
axs[1].scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outliers')
axs[1].set_title("Outliers via One-Class SVM")
axs[1].axis('equal')
axs[1].legend()

# Plot the quantile-based outliers
axs[2].scatter(x[:, 0], x[:, 1])
axs[2].scatter(outliers_quantile[:, 0], outliers_quantile[:, 1], color='red', label='Outliers')
axs[2].set_title("Outliers based on Quantile Threshold")
axs[2].axis('equal')
axs[2].legend()

# Display the plot
plt.tight_layout()
plt.show()
