import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Define column names
column_names = [
    "BI-RADS",
    "Age",
    "Shape",
    "Margin",
    "Density",
    "Severity"
]

# Load the dataset
file_path = "mammographic_masses.data.txt"
dataset = pd.read_csv(file_path, na_values=["?"], names=column_names, delimiter=",")

# Data seems randomly distributed, so drop all rows having missing value
new_dataset = dataset.dropna()

attributes = new_dataset.drop(["Severity", "BI-RADS"], axis=1)
label = np.array(new_dataset["Severity"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(attributes, label, test_size=0.25, random_state=42)

# Normalized data
scaler = StandardScaler()
scaler_X = scaler.fit_transform(X_train)

# Create the KNN classifier
knn = KNeighborsClassifier()

loop_range = range(1, 51)

# Define a range of K values to search for
param_grid = {"n_neighbors": list(loop_range)}

# Use GridSearchCV to find the best K value
grid_search = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy")
grid_search.fit(scaler_X, y_train)

# Get the best K value
best_K = grid_search.best_params_["n_neighbors"]

print("Best K neighbor:", best_K)

knn_accuracies = []

for i in loop_range:
    X_train, X_test, y_train, y_test = train_test_split(attributes, label, test_size=0.25)

    # Standardize the features (normalize)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    knn_accuracies.append(accuracy)

average_accuracy = np.mean(knn_accuracies)
print(f"KNN (K={best_K}) Average Accuracy: {average_accuracy * 100:.3f}%")  # 80.240%

plt.figure(figsize=(8, 6))
plt.plot(loop_range, [accuracy * 100 for accuracy in knn_accuracies], marker="o")
plt.title(f"KNN (K={best_K}) Accuracy for each loop")
plt.xlabel("Loop")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()
