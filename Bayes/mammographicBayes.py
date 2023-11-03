import itertools

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
file_path = "../dataset/mammographic_masses.data.txt"
dataset = pd.read_csv(file_path, na_values=['?'], names=column_names, delimiter=",")

new_dataset = dataset.dropna()

attributes = new_dataset.drop(['Severity', 'BI-RADS'], axis=1)
label = np.array(new_dataset['Severity'])

scaler = StandardScaler()
scaler_X = scaler.fit_transform(attributes)

# Bayes
loop_range = range(1, 51)

gaussian_accuracies = []
gaussian_precisions = []
gaussian_recalls = []
gaussian_f1_scores = []
gaussian_cm_list = []

# Split the data into training and testing sets
for j in loop_range:
    X_train, X_test, y_train, y_test = train_test_split(attributes, label, test_size=0.25)

    # Normalized data
    scaler = StandardScaler()
    scaler_X_train = scaler.fit_transform(X_train)
    scaler_X_test = scaler.transform(X_test)

    # Create the Gaussian Naive Bayes classifier
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(scaler_X_train, y_train)

    y_pred = gaussian_nb.predict(scaler_X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    gaussian_accuracies.append(accuracy)
    gaussian_precisions.append(precision)
    gaussian_recalls.append(recall)
    gaussian_f1_scores.append(f1)
    gaussian_cm_list.append(cm)

average_gaussian_accuracy = np.mean(gaussian_accuracies)
average_gaussian_precision = np.mean(gaussian_precisions)
average_gaussian_recall = np.mean(gaussian_recalls)
average_gaussian_f1_score = np.mean(gaussian_f1_scores)
average_confusion_matrix = np.mean(gaussian_cm_list, axis=0)

print(f"Naive Bayes (GaussianNB) Average Accuracy: {average_gaussian_accuracy * 100:.3f}%")
print(f"Naive Bayes (GaussianNB) Average Precision: {average_gaussian_precision * 100:.3f}%")
print(f"Naive Bayes (GaussianNB) Average Recall: {average_gaussian_recall * 100:.3f}%")
print(f"Naive Bayes (GaussianNB) Average F1 Score: {average_gaussian_f1_score * 100:.3f}%")
print("Confusion Matrix:")
print(average_confusion_matrix)
print("=" * 40)

metrics_df = pd.DataFrame({
    'Accuracy': gaussian_accuracies,
    'Precision': gaussian_precisions,
    'Recall': gaussian_recalls,
    'F1 Score': gaussian_f1_scores
})

# Indicator
plt.figure(figsize=(10, 6))
plt.plot(loop_range, [metric * 100 for metric in metrics_df['Accuracy']], label='Accuracy', marker='o')
plt.plot(loop_range, [metric * 100 for metric in metrics_df['Precision']], label='Precision', marker='o')
plt.plot(loop_range, [metric * 100 for metric in metrics_df['Recall']], label='Recall', marker='o')
plt.plot(loop_range, [metric * 100 for metric in metrics_df['F1 Score']], label='F1 Score', marker='o')

plt.title("Naive Bayes (GaussianNB) Metrics for Different Loop Values")
plt.xlabel("Loop")
plt.ylabel("Score (%)")
plt.legend()
plt.grid(True)
plt.savefig("images/Bayes_acc_" + str(round(average_gaussian_accuracy * 100, 3))
            + "_pre_" + str(round(average_gaussian_precision * 100, 3))
            + "_rec_" + str(round(average_gaussian_recall * 100, 3))
            + "_f1_" + str(round(average_gaussian_f1_score * 100, 3))
            + ".png")

class_names = ["Negative", "Positive"]

plt.figure(figsize=(6, 4))
plt.imshow(average_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Average Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')

thresh = average_confusion_matrix.max() / 2.
for i, j in itertools.product(range(average_confusion_matrix.shape[0]), range(average_confusion_matrix.shape[1])):
    plt.text(j, i, f"{average_confusion_matrix[i, j]:.2f}", horizontalalignment="center",
             color="white" if average_confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("images/average_confusion_matrix_"
            + str(round(average_gaussian_accuracy * 100, 3))
            + ".png")

