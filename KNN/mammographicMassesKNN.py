import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
file_path = "../dataset/mammographic_masses.data.txt"
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

for i in range(1, 21):
    knn_accuracies = []
    knn_precisions = []
    knn_recalls = []
    knn_f1_scores = []

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
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        knn_accuracies.append(accuracy)
        knn_precisions.append(precision)
        knn_recalls.append(recall)
        knn_f1_scores.append(f1)

    average_accuracy = np.mean(knn_accuracies)
    average_precision = np.mean(knn_precisions)
    average_recall = np.mean(knn_recalls)
    average_f1_score = np.mean(knn_f1_scores)

    print(f"KNN (K={best_K}) Average Accuracy: {average_accuracy * 100:.3f}%")  # 80.212%
    print(f"KNN (K={best_K}) Average Precision: {average_precision * 100:.3f}%")
    print(f"KNN (K={best_K}) Average Recall: {average_recall * 100:.3f}%")
    print(f"KNN (K={best_K}) Average F1 Score: {average_f1_score * 100:.3f}%")
    print("Confusion Matrix:")
    print(cm)
    print("=" * 40)

    plt.figure(figsize=(8, 6))
    plt.plot(loop_range, [metric * 100 for metric in knn_accuracies], label='Accuracy', marker='o')
    plt.plot(loop_range, [metric * 100 for metric in knn_precisions], label='Precision', marker='o')
    plt.plot(loop_range, [metric * 100 for metric in knn_recalls], label='Recall', marker='o')
    plt.plot(loop_range, [metric * 100 for metric in knn_f1_scores], label='F1 Score', marker='o')
    plt.title(f"KNN (K={best_K}) Metrics for Different Loop Values")
    plt.xlabel("Loop")
    plt.ylabel("Score (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig("images/details/KNN_acc_" + str(round(average_accuracy * 100, 3))
                + "_pre_" + str(round(average_precision * 100, 3))
                + "_rec_" + str(round(average_recall * 100, 3))
                + "_f1_" + str(round(average_f1_score * 100, 3))
                + ".png")

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix")

    # Save the confusion matrix as an image
    plt.savefig("images/matrix/Confusion_Matrix_" + str(round(average_accuracy * 100, 3))
                + "_pre_" + str(round(average_precision * 100, 3))
                + "_rec_" + str(round(average_recall * 100, 3))
                + "_f1_" + str(round(average_f1_score * 100, 3))
                + ".png")


    plt.show()


