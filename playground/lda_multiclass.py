import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X, y = make_classification(n_samples=1000,     # number of samples
                           n_features=20,       # number of features
                           n_informative=10,    # number of informative features
                           n_redundant=5,       # number of redundant features
                           n_classes=3,         # number of classes
                           random_state=42)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["class"] = y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_scaled, y_train)
y_pred = qda.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Plotting decision boundary
def plot_decision_boundary(X, y, model, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)

# Reduce to 2 features for visualization
X_train_2d = X_train_scaled[:, :2]
X_test_2d = X_test_scaled[:, :2]
qda_2d = QuadraticDiscriminantAnalysis()
qda_2d.fit(X_train_2d, y_train)

fig, ax = plt.subplots()
plot_decision_boundary(X_test_2d, y_test, qda_2d, ax)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("QDA Decision Boundary")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()