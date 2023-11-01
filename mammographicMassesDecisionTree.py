# Define column names
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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

# Decision Tree Classifier (because label is classified)
dt_accuracies = []
dt_precisions = []
dt_recalls = []
dt_f1_scores = []

for train_index, test_index in kf.split(scaler_X):
    X_train, X_test = scaler_X[train_index], scaler_X[test_index]
    y_train, y_test = label[train_index], label[test_index]

    # Create and fit the Decision Tree model
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)

    # Predict and calculate metrics
    y_pred = decision_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append metrics to the lists
    dt_accuracies.append(accuracy)
    dt_precisions.append(precision)
    dt_recalls.append(recall)
    dt_f1_scores.append(f1)

# Calculate the average metrics over all folds for Decision Tree
average_accuracy_dt = np.mean(dt_accuracies)
average_precision_dt = np.mean(dt_precisions)
average_recall_dt = np.mean(dt_recalls)
average_f1_score_dt = np.mean(dt_f1_scores)

print("Decision Tree Average Accuracy:", round(average_accuracy_dt * 100, 2), "%")
print("Decision Tree Average Precision:", round(average_precision_dt * 100, 2), "%")
print("Decision Tree Average Recall:", round(average_recall_dt * 100, 2), "%")
print("Decision Tree Average F1 Score:", round(average_f1_score_dt * 100, 2), "%")

