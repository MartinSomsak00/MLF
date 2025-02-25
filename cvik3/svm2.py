# Načtení knihoven
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, OneClassSVM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from numpy import quantile, where, random
import numpy as np

# 1. Načtení IRIS datasetu a kontrola obsahu
iris = load_iris()
print("Názvy vlastností:", iris.feature_names)
print("Prvních 5 řádků dat:\n", iris.data[0:5, :])
print("Prvních 5 cílových hodnot:", iris.target[0:5])

# 2. Rozdělení dat na trénovací a testovací sadu
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Trénovací data:", X_train.shape)
print("Testovací data:", X_test.shape)

# 3. Trénování SVM modelu pro klasifikaci
SVMmodel = SVC(kernel='linear')
SVMmodel.fit(X_train, y_train)
print("Přesnost modelu na testovacích datech:", SVMmodel.score(X_test, y_test))

# 4. Práce pouze s prvními dvěma vlastnostmi a odstranění třetí třídy
X = iris.data[:, :2]  # První dvě vlastnosti
y = iris.target

# Filtruj pouze řádky, kde y není 2
X = X[y != 2]  # Filtruj X podle y
y = y[y != 2]  # Filtruj y

# 5. Vizualizace dat
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Třída 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Třída 1')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.title("Rozdělení tříd")
plt.show()

# 6. Trénování SVM modelu pro dvě třídy
SVMmodel = SVC(kernel='linear', C=200)  # C je regularizační parametr
SVMmodel.fit(X, y)

# 7. Zobrazení rozhodovací hranice (bez podpůrných vektorů a s tlustší čárou)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)  # Scatterplot dat
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Vykreslení rozhodovací hranice
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = SVMmodel.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Změna: tlustší čára a odstranění podpůrných vektorů
plt.contour(xx, yy, Z, colors='k', levels=[0], linewidths=2, linestyles='-')  # Tlustší čára (linewidths=2)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Rozhodovací hranice")
plt.show()

# 8. Detekce anomálií pomocí one-class SVM
random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(4, 4))

# Trénování one-class SVM
SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
SVMmodelOne.fit(x)
pred = SVMmodelOne.predict(x)
anom_index = where(pred == -1)
values = x[anom_index]

# Vizualizace anomálií
plt.scatter(x[:, 0], x[:, 1])
plt.scatter(values[:, 0], values[:, 1], color='red', label='Anomálie')
plt.legend()
plt.title("Detekce anomálií")
plt.show()

# 9. Detekce anomálií s kontrolou pomocí kvantilu
scores = SVMmodelOne.decision_function(x)
thresh = quantile(scores, 0.05)  # 5% kvantil
index = where(scores <= thresh)
values = x[index]

plt.scatter(x[:, 0], x[:, 1])
plt.scatter(values[:, 0], values[:, 1], color='red', label='Anomálie (5% kvantil)')
plt.legend()
plt.title("Detekce anomálií s kontrolou pomocí kvantilu")
plt.show()