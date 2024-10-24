import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./ISLP/LinearRegression/test.csv")
y_values = data["y"].values
x_values = data["x"].values

sum_x = sum(x_values)
sum_y = sum(y_values)
sum_x_squared = sum(x**2 for x in x_values)
sum_xy = sum(x*y for x, y in zip(x_values, y_values))

n = len(x_values)
m = (n*sum_xy - sum_x*sum_y) / (n*sum_x_squared - sum_x**2)
b = (sum_y - m*sum_x) / n

residuals = [y - (m*x + b) for x, y in zip(x_values, y_values)]

for x, y, residual in zip(x_values, y_values, residuals):
    plt.vlines(x, y, m*x + b, color='green' if residual >= 0 else 'red', linewidth=2)

plt.scatter(x_values, y_values, color="blue", label="Data", s=5)
plt.plot(x_values, m*x_values + b, color="purple", label="Least Squares Line")
plt.show()
