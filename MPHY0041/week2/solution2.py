import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

train = pd.read_csv("./MPHY0041/week2/adni_adas13_train.csv", index_col=0)
test = pd.read_csv("./MPHY0041/week2/adni_adas13_test.csv", index_col=0)

X, y = train.drop(columns=["ADAS13"]), train["ADAS13"]
X_test, y_test = test.drop(columns=["ADAS13"]), test["ADAS13"]

scaler = StandardScaler()
X_scaled, X_test_scaled = scaler.fit_transform(X), scaler.transform(X_test)

lm = LinearRegression()
lm.fit(X_scaled, y)
y_pred = lm.predict(X_test_scaled)

print(f"Coefficients: {lm.coef_}\nIntercept: {lm.intercept_}")
print(f"Correlation: {np.corrcoef(y_test, y_pred)[0, 1]}, MSE: {mean_squared_error(y_test, y_pred)}")

plt.scatter(y_test, y_pred)
plt.xlabel("ADAS13")
plt.ylabel("ADAS13 (Predicted)")
plt.title("Linear Regression Predictions")
plt.show()

alphas = np.logspace(-1, 1, 100)
ridge_metrics, lasso_metrics = [], []

for alpha in alphas:
    # Ridge Regression
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, y)
    y_pred_ridge = ridge.predict(X_test_scaled)
    ridge_metrics.append((ridge.coef_, np.corrcoef(y_test, y_pred_ridge)[0, 1], mean_squared_error(y_test, y_pred_ridge)))

    # LASSO
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)
    y_pred_lasso = lasso.predict(X_test_scaled)
    lasso_metrics.append((lasso.coef_, np.corrcoef(y_test, y_pred_lasso)[0, 1], mean_squared_error(y_test, y_pred_lasso)))

ridge_df = pd.DataFrame(ridge_metrics, columns=["Coefficients", "Correlation", "MSE"], index=alphas)
lasso_df = pd.DataFrame(lasso_metrics, columns=["Coefficients", "Correlation", "MSE"], index=alphas)

plt.figure(figsize=(14, 12))

# Ridge Coefficients
plt.subplot(2, 3, 1)
for feature in range(X.shape[1]):
    plt.plot(alphas, [coef[feature] for coef in ridge_df["Coefficients"]], marker="o")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Coefficient Value")
plt.xscale("log")
plt.title("Ridge Coefficients")
plt.grid(True)

# Ridge Correlation and MSE
plt.subplot(2, 3, 2)
plt.plot(alphas, ridge_df["Correlation"], marker="o", color="b")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Correlation")
plt.xscale("log")
plt.title("Ridge Correlation")
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(alphas, ridge_df["MSE"], marker="o", color="r")
plt.xlabel("Alpha (log scale)")
plt.ylabel("MSE")
plt.xscale("log")
plt.title("Ridge MSE")
plt.grid(True)

# Lasso Coefficients
plt.subplot(2, 3, 4)
for feature in range(X.shape[1]):
    plt.plot(alphas, [coef[feature] for coef in lasso_df["Coefficients"]], marker="o")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Coefficient Value")
plt.xscale("log")
plt.title("Lasso Coefficients")
plt.grid(True)

# Lasso Correlation and MSE
plt.subplot(2, 3, 5)
plt.plot(alphas, lasso_df["Correlation"], marker="o", color="b")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Correlation")
plt.xscale("log")
plt.title("Lasso Correlation")
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(alphas, lasso_df["MSE"], marker="o", color="r")
plt.xlabel("Alpha (log scale)")
plt.ylabel("MSE")
plt.xscale("log")
plt.title("Lasso MSE")
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.4, wspace=0.3)
plt.show()