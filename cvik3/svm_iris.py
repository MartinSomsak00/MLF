# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load Iris dataset and select only the first two features
iris = load_iris()
X = iris.data[:, :2]  # Select first two features
y = iris.target

# Keep only classes 0 and 1 for the SVM model
mask = y < 2
X, y = X[mask], y[mask]

# Train SVM with Linear Kernel (Only Class 0 and Class 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel='linear', C=200)  # You can modify C to experiment
svm_model.fit(X_train, y_train)

# Print model accuracy
accuracy = svm_model.score(X_test, y_test)
print(f"SVM Model Accuracy: {accuracy:.2f}")

# Get separating line coefficients
W = svm_model.coef_[0]
b = svm_model.intercept_[0]

# Plot decision boundary
x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_values = -(W[0] * x_values + b) / W[1]

# Plot combined figure: Classes and decision boundary
plt.figure(figsize=(6,4))

# Scatter plot for Class 0 (blue) and Class 1 (red)
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='blue', label="Class 0", edgecolors='k')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='red', label="Class 1", edgecolors='k')

# Plot the SVM decision boundary
plt.plot(x_values, y_values, 'k--', label="Decision Boundary")

# Labels and legend
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.title("SVM Decision Boundary with Classes")
plt.show()
