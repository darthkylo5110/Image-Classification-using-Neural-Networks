# -*- coding: utf-8 -*-
from sklearnex import get_patch_names, patch_sklearn
patch_sklearn()
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Load the Fashion-MNIST dataset 
fashion_mnist = datasets.fetch_openml(name="Fashion-MNIST")
X = np.array(fashion_mnist.data.astype("float32"))
y = np.array(fashion_mnist.target.astype("int"))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data (scale grayscale values from 0 to 255 to the range [0, 1])
X_train /= 255.0
X_test /= 255.0

# Standardize the data (optional but can improve the performance of linear SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and fit the linear SVM model
svm_model_linear = SVC(kernel="linear", C=1.0, random_state=42)
svm_model_linear.fit(X_train, y_train)

# Make predictions on the test set
y_pred_linear = svm_model_linear.predict(X_test)

# Evaluate the linear SVM model's performance
classification_report_linear = classification_report(y_test, y_pred_linear)

# Display the results for linear SVM
print("Linear SVM Classification Report:")
print(classification_report_linear)

# Ajustar el modelo SVM de base radial
svm_model_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model_rbf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_rbf = svm_model_rbf.predict(X_test)

# Evaluar el rendimiento del modelo SVM de base radial
report_rbf = classification_report(y_test, y_pred_rbf)

# Mostrar resultados para SVM de base radial
print("SVM RBF Classification Report:\n", report_rbf)

from sklearn.neural_network import MLPClassifier

# Crear y ajustar el modelo MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_mlp = mlp_model.predict(X_test)

# Evaluar el rendimiento del modelo MLP
report_mlp = classification_report(y_test, y_pred_mlp)

# Mostrar resultados para MLP
print("MLP Classification Report:\n", report_mlp)