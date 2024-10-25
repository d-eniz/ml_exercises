import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

train = pd.read_csv("./MPHY0041/week1/mixture.csv")
test = pd.read_csv("./MPHY0041/week1/mixture_test.csv")

def KNN(train, test, k=1):
    result = []
    for i in range(test.shape[0]): # Iterate over each test sample
        distances = np.sqrt(np.sum((train.iloc[:, :2].values - test.iloc[i, :2].values) ** 2, axis=1)) # Euclidean distance
        k_indices = np.argsort(distances)[:k] # Indices of k nearest neighbors
        k_labels = train.iloc[k_indices, 2].astype(int) # Labels of k nearest neighbors
        k_label = Counter(k_labels).most_common(1)[0][0] # Most common label
        result.append(k_label)
    return result

k = 1 # Number of nearest neighbors

x1_test = test["X1"].values
x2_test = test["X2"].values
colors_test = KNN(train, test, k)
colors_test = pd.Series(colors_test).map({0: 'cyan', 1: 'orange'})
plt.scatter(x1_test, x2_test, c=colors_test, label='Test')

x1_train = train["X1"].values
x2_train = train["X2"].values
colors_train = train["Y"].map({0: 'blue', 1: 'red'})
plt.scatter(x1_train, x2_train, c=colors_train, label='Train')

plt.legend()
plt.xlabel("X1")
plt.ylabel("X2")
plt.title(f"KNN (k={k})")
plt.show()