import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("./playground/breast-cancer.csv", index_col=0)
print(df.head())

X, y = df.drop(columns="diagnosis"), df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.describe())

scaler = StandardScaler()
encoder = LabelEncoder()
X_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
y_train, y_test = encoder.fit_transform(y_train), encoder.transform(y_test)

model = LogisticRegression()
model.fit(X_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")