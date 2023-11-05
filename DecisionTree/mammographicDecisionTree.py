import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
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

file_path = "../dataset/mammographic_masses.data.txt"
dataset = pd.read_csv(file_path, na_values=["?"], names=column_names, delimiter=",")

new_dataset = dataset.dropna()

# Split the data into attributes and labels
attributes = new_dataset.drop(["Severity", "BI-RADS"], axis=1)
label = np.array(new_dataset["Severity"])

X_train, X_test, y_train, y_test = train_test_split(attributes, label, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dt = DecisionTreeClassifier()

param_grid = {
    'max_depth': range(1, 11),
    'criterion': ['gini', 'entropy'],
}

grid_search = GridSearchCV(dt, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_max_depth = grid_search.best_params_['max_depth'] - 1
best_criterion = grid_search.best_params_['criterion']

print("Best Decision Tree - Max Depth:", best_max_depth)
print("Best Decision Tree - Criterion:", best_criterion)

max_depths = range(1, 11)
criterion_options = ['gini', 'entropy']

accuracy_matrix = np.zeros((len(max_depths), len(criterion_options)))

for i, max_depth in enumerate(max_depths):
    for j, criterion in enumerate(criterion_options):

        dt = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_matrix[i, j] = accuracy

plt.figure(figsize=(10, 6))
sns.heatmap(accuracy_matrix, annot=True, cmap="YlGnBu", xticklabels=criterion_options, yticklabels=max_depths)
plt.xlabel("Criterion")
plt.ylabel("Max Depth")
plt.title("Decision Tree Accuracy vs. Max Depth and Criterion")
plt.savefig("images/depth_criterion.png")


loop_range = range(1, 51)

dt_accuracies = []
dt_precisions = []
dt_recalls = []
dt_f1_scores = []
dt_cm_list = []

for j in loop_range:
    X_train, X_test, y_train, y_test = train_test_split(attributes, label, test_size=0.25)

    # Standardize the features (normalize)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    dt = DecisionTreeClassifier(max_depth=best_max_depth, criterion=best_criterion)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    dt_accuracies.append(accuracy)
    dt_precisions.append(precision)
    dt_recalls.append(recall)
    dt_f1_scores.append(f1)
    dt_cm_list.append(cm)

average_dt_accuracy = np.mean(dt_accuracies)
average_dt_precision = np.mean(dt_precisions)
average_dt_recall = np.mean(dt_recalls)
average_dt_f1_score = np.mean(dt_f1_scores)
average_confusion_matrix = np.mean(dt_cm_list, axis=0)

print(
    f"Decision Tree (Max Depth={best_max_depth}, Criterion={best_criterion}) Average Accuracy: {average_dt_accuracy * 100:.3f}%")
print(
    f"Decision Tree (Max Depth={best_max_depth}, Criterion={best_criterion}) Average Precision: {average_dt_precision * 100:.3f}%")
print(
    f"Decision Tree (Max Depth={best_max_depth}, Criterion={best_criterion}) Average Recall: {average_dt_recall * 100:.3f}%")
print(
    f"Decision Tree (Max Depth={best_max_depth}, Criterion={best_criterion}) Average F1 Score: {average_dt_f1_score * 100:.3f}%")
print("Confusion Matrix:")
print(average_confusion_matrix)
print("=" * 40)

dt_metrics_df = pd.DataFrame({
    'Accuracy': dt_accuracies,
    'Precision': dt_precisions,
    'Recall': dt_recalls,
    'F1 Score': dt_f1_scores
})

plt.figure(figsize=(10, 6))
plt.plot(loop_range, [metric * 100 for metric in dt_metrics_df['Accuracy']], label='Accuracy', marker='o')
plt.plot(loop_range, [metric * 100 for metric in dt_metrics_df['Precision']], label='Precision', marker='o')
plt.plot(loop_range, [metric * 100 for metric in dt_metrics_df['Recall']], label='Recall', marker='o')
plt.plot(loop_range, [metric * 100 for metric in dt_metrics_df['F1 Score']], label='F1 Score', marker='o')

plt.title("Decision Tree Metrics for Different Loop Values")
plt.xlabel("Loop")
plt.ylabel("Score (%)")
plt.legend()
plt.grid(True)
plt.savefig("images/DT_acc_" + str(round(average_dt_accuracy * 100, 3))
            + "_pre_" + str(round(average_dt_precision * 100, 3))
            + "_rec_" + str(round(average_dt_recall * 100, 3))
            + "_f1_" + str(round(average_dt_f1_score * 100, 3))
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
            + str(round(average_dt_accuracy * 100, 3))
            + ".png")
