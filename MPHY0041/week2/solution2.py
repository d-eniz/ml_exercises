import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train = pd.read_csv("./MPHY0041/week2/adni_adas13_train.csv", index_col=0)
test = pd.read_csv("./MPHY0041/week2/adni_adas13_test.csv", index_col=0)
print(train.head())
print(train.describe())

X = train.drop(columns=["ADAS13"])
y = train["ADAS13"]
X_test = test.drop(columns=["ADAS13"])
y_test = test["ADAS13"]

print(y.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_test_scaled = scaler.transform(X_test)
print(X_scaled_df.describe())

lm = LinearRegression()
lm.fit(X_scaled, y)

y_pred = lm.predict(X_test_scaled)
print(f"Coefficients: {lm.coef_}")
print(f"Intercept: {lm.intercept_}")
correlation = np.corrcoef(y_test, y_pred)[0, 1]
mse = mean_squared_error(y_test, y_pred)
print(f"Correlation: {correlation}, MSE: {mse}")

plt.scatter(y_test, y_pred)
plt.xlabel("ADAS13")
plt.ylabel("ADAS13 (Predicted)")
plt.show()

alphas = np.logspace(0, 5, 100)
coefficients = []
correlations = []
mses = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, y)
    y_pred_ridge = ridge.predict(X_test_scaled)

    coefficients.append(ridge.coef_)
    correlations.append(np.corrcoef(y_test, y_pred_ridge)[0, 1])
    mses.append(mean_squared_error(y_test, y_pred_ridge))

coefficients_df = pd.DataFrame(coefficients, columns=X.columns, index=alphas)
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
for feature in coefficients_df.columns:
    plt.plot(coefficients_df.index, coefficients_df[feature], marker="o", label=feature)
plt.xlabel("Alpha")
plt.ylabel("Coefficient Value")
plt.xscale("log")
plt.title("Ridge Coefficients over Alpha")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(alphas, correlations, marker="o", color="b", label="Correlation")
plt.xlabel("Alpha")
plt.ylabel("Correlation")
plt.xscale("log")
plt.title("Correlation over Alpha")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(alphas, mses, marker="o", color="r", label="MSE")
plt.xlabel("Alpha")
plt.ylabel("Mean Squared Error")
plt.xscale("log")
plt.title("MSE over Alpha")
plt.grid(True)

plt.tight_layout()
plt.show()
