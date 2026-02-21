"""
Model Comparison: Support Vector Machine (SVM)

The Support Vector Machine (SVM) is a classification algorithm that finds the optimal
hyperplane that maximizes the margin between two classes. It focuses on the data points
closest to the decision boundary, called support vectors, which determine the position
of the separating boundary. I chose SVM because it performs well on high-dimensional
datasets and often achieves strong accuracy on structured medical data.
"""

import torch
from sklearn.svm import SVC
from binary_classification import load_data, train, predict, accuracy


# Load data
X_train, X_test, y_train, y_test, _ = load_data()

# Train from-scratch model
w, b, _ = train(X_train, y_train, alpha=0.01, n_epochs=100, verbose=False)

custom_test_pred = predict(X_test, w, b)
custom_test_acc = accuracy(y_test, custom_test_pred)

# Convert to NumPy for sklearn
X_train_np = X_train.numpy()
X_test_np = X_test.numpy()
y_train_np = y_train.numpy()
y_test_np = y_test.numpy()

# Train SVM
svm_model = SVC()
svm_model.fit(X_train_np, y_train_np)

svm_test_acc = svm_model.score(X_test_np, y_test_np)

print(f"Custom Model Test Accuracy: {custom_test_acc:.4f}")
print(f"SVM Test Accuracy: {svm_test_acc:.4f}")

"""
Comparison Discussion:

The SVM model achieved slightly higher test accuracy than the
from-scratch implementation. This is likely because SVM maximizes
the margin between classes and benefits from optimized library
implementations. While the custom model provides insight into how
gradient descent works internally, the sklearn model is more
efficient and robust in practice.
"""

