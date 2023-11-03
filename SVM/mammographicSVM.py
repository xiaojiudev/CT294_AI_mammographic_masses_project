import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

column_names = [
    "BI-RADS",
    "Age",
    "Shape",
    "Margin",
    "Density",
    "Severity"
]

file_path = "../dataset/mammographic_masses.data.txt"
dataset = pd.read_csv(file_path, na_values=["?"], names=column_names, delimiter=",")

new_dataset = dataset.dropna()

X = new_dataset.drop(["Severity", "BI-RADS"], axis=1)
y = np.array(new_dataset["Severity"])

loop_range = range(1, 51)

# Find the best Kernel for SVM
svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

best_kernel = None
best_accuracy = 0.0
kernel_option_accuracies = {}

for kernel_option in svm_kernels:
    accuracies = []

    for j in loop_range:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm = SVC(kernel=kernel_option)
        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    kernel_option_accuracies[kernel_option] = average_accuracy

    print(f"Average SVM Accuracy with Kernel='{kernel_option}': {average_accuracy * 100:.3f}%")

    if average_accuracy > best_accuracy:
        best_accuracy = average_accuracy
        best_kernel = kernel_option

print(f"Best-performing Kernel: {best_kernel}")
print(f"Best SVM Accuracy: {best_accuracy * 100:.3f}%")
print("=" * 40)

# Draw chart for kernel - accuracy
kernels = list(kernel_option_accuracies.keys())
accuracies = list(kernel_option_accuracies.values())

plt.figure(figsize=(8, 6))
plt.bar(kernels, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.title("SVM Kernel Accuracies")
plt.xlabel("Kernel")
plt.ylabel("Accuracy")
plt.ylim(0.7, 0.9)  # Set y-axis limits
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, acc in enumerate(accuracies):
    plt.text(i, acc, f'{acc:.3f}', ha='center', va='bottom')

plt.savefig("images/svm_kernel_accuracies.png")

# Begin to split, train and test model
svm_accuracies = []
svm_precisions = []
svm_recalls = []
svm_f1_scores = []
svm_cm_list = []

for j in loop_range:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel=best_kernel)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    svm_accuracies.append(accuracy)
    svm_precisions.append(precision)
    svm_recalls.append(recall)
    svm_f1_scores.append(f1)
    svm_cm_list.append(cm)

average_svm_accuracy = np.mean(svm_accuracies)
average_svm_precision = np.mean(svm_precisions)
average_svm_recall = np.mean(svm_recalls)
average_svm_f1_score = np.mean(svm_f1_scores)
average_confusion_matrix = np.mean(svm_cm_list, axis=0)

print(f"Average SVM Accuracy: {average_svm_accuracy * 100:.3f}%")
print(f"Average SVM Precision: {average_svm_precision * 100:.3f}%")
print(f"Average SVM Recall: {average_svm_recall * 100:.3f}%")
print(f"Average SVM F1 Score: {average_svm_f1_score * 100:.3f}%")
print(f"Average SVM Confusion Matrix: {average_confusion_matrix}")
print("=" * 40)

svm_metrics_df = pd.DataFrame({
    'Accuracy': svm_accuracies,
    'Precision': svm_precisions,
    'Recall': svm_recalls,
    'F1 Score': svm_f1_scores
})

plt.figure(figsize=(10, 6))
plt.plot(loop_range, [metric * 100 for metric in svm_metrics_df['Accuracy']], label='Accuracy', marker='o')
plt.plot(loop_range, [metric * 100 for metric in svm_metrics_df['Precision']], label='Precision', marker='o')
plt.plot(loop_range, [metric * 100 for metric in svm_metrics_df['Recall']], label='Recall', marker='o')
plt.plot(loop_range, [metric * 100 for metric in svm_metrics_df['F1 Score']], label='F1 Score', marker='o')

plt.title("SVM Metrics for Different Loop Values")
plt.xlabel("Loop")
plt.ylabel("Score (%)")
plt.legend()
plt.grid(True)
plt.savefig("images/SVM_acc_" + str(round(average_svm_accuracy * 100, 3))
            + "_pre_" + str(round(average_svm_precision * 100, 3))
            + "_rec_" + str(round(average_svm_recall * 100, 3))
            + "_f1_" + str(round(average_svm_recall * 100, 3))
            + ".png")

class_names = ["Negative", "Positive"]

# Plot the average confusion matrix
plt.figure(figsize=(6, 4))
plt.imshow(average_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Average Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Add text to cells
thresh = average_confusion_matrix.max() / 2.
for i, j in itertools.product(range(average_confusion_matrix.shape[0]), range(average_confusion_matrix.shape[1])):
    plt.text(j, i, f"{average_confusion_matrix[i, j]:.2f}", horizontalalignment="center",
             color="white" if average_confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("images/average_confusion_matrix_" +
            str(round(average_svm_accuracy * 100, 3)) + ".png")
