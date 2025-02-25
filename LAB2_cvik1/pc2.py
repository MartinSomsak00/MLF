import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Defined 3 points in 2D-space:
X = np.array([[2, 1, 0], [4, 3, 0]])

# Calculate the covariance matrix
R = np.cov(X)

# Calculate the SVD decomposition and new basis vectors
U, D, V = np.linalg.svd(R)  # call SVD decomposition
u1 = U[:, 0]  # new basis vectors
u2 = U[:, 1]

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocessing: StandardScaler
Xscaler = preprocessing.StandardScaler()
Xpp = Xscaler.fit_transform(X)

# PCA with three components
pca = PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print("Covariance matrix:", pca.get_covariance())

# 3D Scatter plot of transformed feature space
fig = plt.figure(figsize=(8,6))
axes2 = fig.add_subplot(111, projection='3d')
axes2.scatter(Xpca[y == 0, 0], Xpca[y == 0, 1], Xpca[y == 0, 2], color='green', label='Class 0')
axes2.scatter(Xpca[y == 1, 0], Xpca[y == 1, 1], Xpca[y == 1, 2], color='blue', label='Class 1')
axes2.scatter(Xpca[y == 2, 0], Xpca[y == 2, 1], Xpca[y == 2, 2], color='magenta', label='Class 2')
axes2.set_xlabel('PC1')
axes2.set_ylabel('PC2')
axes2.set_zlabel('PC3')
axes2.legend()
plt.title('3D PCA Projection of Iris Dataset')
plt.show()

# Explained variance
print("Explained variance:", pca.explained_variance_)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# 2D Scatter plot of principal components
plt.figure(figsize=(8,6))
plt.scatter(Xpca[y == 0, 0], Xpca[y == 0, 1], color='green', label='Class 0')
plt.scatter(Xpca[y == 1, 0], Xpca[y == 1, 1], color='blue', label='Class 1')
plt.scatter(Xpca[y == 2, 0], Xpca[y == 2, 1], color='magenta', label='Class 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('2D PCA Projection of Iris Dataset')
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN classifier on original 4D data
knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_train, y_train)
y_pred1 = knn1.predict(X_test)

# Confusion matrix for full dataset
fig, ax = plt.subplots(figsize=(6,5))
cm1 = confusion_matrix(y_test, y_pred1)
ConfusionMatrixDisplay(cm1).plot(ax=ax)
plt.title('Confusion Matrix: KNN on Full 4D Data (k=3)')
plt.show()

# KNN classifier on PCA-transformed data (first two components)
X_pca2 = Xpca[:, :2]
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca2, y, test_size=0.3, random_state=42)

knn2 = KNeighborsClassifier(n_neighbors=3)
knn2.fit(X_pca_train, y_pca_train)
y_pred2 = knn2.predict(X_pca_test)

# Confusion matrix for PCA-transformed data
fig, ax = plt.subplots(figsize=(6,5))
cm2 = confusion_matrix(y_pca_test, y_pred2)
ConfusionMatrixDisplay(cm2).plot(ax=ax)
plt.title('Confusion Matrix: KNN on PCA-Reduced Data (k=3)')
plt.show()

# KNN classifier on first two dimensions of original data
X_2D = X[:, :2]
X_2D_train, X_2D_test, y_2D_train, y_2D_test = train_test_split(X_2D, y, test_size=0.3, random_state=42)

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_2D_train, y_2D_train)
y_pred3 = knn3.predict(X_2D_test)

# Confusion matrix for 2D original data
fig, ax = plt.subplots(figsize=(6,5))
cm3 = confusion_matrix(y_2D_test, y_pred3)
ConfusionMatrixDisplay(cm3).plot(ax=ax)
plt.title('Confusion Matrix: KNN on First Two Features of Original Data (k=3)')
plt.show()
