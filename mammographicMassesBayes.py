import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
dataset = pd.read_csv(file_path, na_values=['?'], names=column_names, delimiter=",")

new_dataset = dataset.dropna()

attributes = new_dataset.drop(['Severity', 'BI-RADS'], axis=1)
label = np.array(new_dataset['Severity'])

scaler = StandardScaler()
scaler_X = scaler.fit_transform(attributes)

kf = KFold(n_splits=10, shuffle=True, random_state=100)

# Bayes

loop_range = range(1, 51)

gaussian_accuracies = []

# Split the data into training and testing sets
for i in loop_range:
    X_train, X_test, y_train, y_test = train_test_split(attributes, label, test_size=0.25)

    # Normalized data
    scaler = StandardScaler()
    scaler_X_train = scaler.fit_transform(X_train)
    scaler_X_test = scaler.transform(X_test)

    # Create the Gaussian Naive Bayes classifier
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(scaler_X_train, y_train)

    y_pred = gaussian_nb.predict(scaler_X_test)
    gaussian_accuracy = accuracy_score(y_test, y_pred)
    gaussian_accuracies.append(gaussian_accuracy)

average_gaussian_accuracy = np.mean(gaussian_accuracies)
print(f"Naive Bayes (GaussianNB) Average Accuracy: {average_gaussian_accuracy * 100:.3f}%")  # 79.529%

plt.figure(figsize=(8, 6))
plt.plot(loop_range, [accuracy * 100 for accuracy in gaussian_accuracies], marker="o")
plt.title("Gaussian Naive Bayes Accuracy for Different Loop Values")
plt.xlabel("Loop")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()