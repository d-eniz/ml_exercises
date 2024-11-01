import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

df = pd.read_csv("./playground/breast-cancer.csv", index_col=0)
print(df.head())

X, y = df.drop(columns="diagnosis"), df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
encoder = LabelEncoder()

X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
y_train_encoded, y_test_encoded = encoder.fit_transform(y_train), encoder.transform(y_test)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train_encoded)

y_pred = lda.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)

print(f"Accuracy: {accuracy:.2f}")


# Transform data onto the single LDA component for 1D plotting
X_train_lda = lda.transform(X_train_scaled)
X_test_lda = lda.transform(X_test_scaled)

# Plotting the LDA with only one component (LD1)
plt.figure(figsize=(8, 6))
for label in np.unique(y_train_encoded):
    plt.hist(
        X_train_lda[y_train_encoded == label],
        bins=20,
        alpha=0.5,
        label=encoder.inverse_transform([label])[0]
    )

plt.xlabel('LD1')
plt.title('LDA')
plt.legend()
plt.show()
