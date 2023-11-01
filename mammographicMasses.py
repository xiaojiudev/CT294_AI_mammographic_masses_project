import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score, \
    recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

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


# Display the first 3 rows in dataset
# print(dataset.head(3))

# Display dataset to evaluate whether data needs cleaning
# print(dataset.describe().transpose())
# print(dataset['Severity'].value_counts())

# Display percentage of missing value
def percent_missing(df):
    percent_null = 100 * df.isnull().sum() / len(df)
    percent_null = percent_null[percent_null > 0].sort_values()
    return percent_null


percent_result = percent_missing(dataset)
sns.barplot(x=percent_result.index, y=percent_result)
plt.xlabel('Attributes')
plt.ylabel('% Missing')
plt.title('Percentage of Missing Values by Attribute')

# plt.show()

# Display all rows that having NaN value => 130 rows
# print(dataset[(dataset['Age'].isnull()) |
#               (dataset['Shape'].isnull()) |
#               (dataset['Margin'].isnull()) |
#               (dataset['Density'].isnull())])

# Display detail total NaN for each attribute
# print(dataset.isnull().sum())

# Data seems randomly distributed, so drop all rows having missing value
new_dataset = dataset.dropna()

# New dataset with 830 rows
# print(new_dataset)

# Check new dataset is having missing value?
# print(new_dataset.isnull().sum())

attributes = new_dataset.drop(['Severity', 'BI-RADS'], axis=1)
label = np.array(new_dataset['Severity'])
# print(attributes)
# print(label)

# Split dataset into 4 parts, 3 parts for train, 1 part for test
# X_train, X_test, y_train, y_test = train_test_split(attributes, label, test_size=0.25, random_state=100)

# Normalized data
scaler = StandardScaler()
scaler_X = scaler.fit_transform(attributes)

# KNN
# Because new dataset (dataset after drop missing value) having totals 830 elements (> 300) => use k-fold with k = 10
kf = KFold(n_splits=10, shuffle=True, random_state=100)

knn_average_accuracies = []

# Test different values of K
for K in range(1, 51):  # You can adjust the range as needed
    knn = KNeighborsClassifier(n_neighbors=K)
    knn_accuracies = []

    for train_index, test_index in kf.split(scaler_X):
        X_train, X_test = scaler_X[train_index], scaler_X[test_index]
        y_train, y_test = label[train_index], label[test_index]

        # Fit the KNN model
        knn.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        knn_accuracies.append(accuracy)

    # Calculate the average accuracy over all folds for this K
    average_accuracy = np.mean(knn_accuracies)
    knn_average_accuracies.append(average_accuracy)

# Find the best K with the highest average accuracy
best_K = np.argmax(knn_average_accuracies) + 1  # Add 1 to convert from 0-based index to actual K

print("Best K neighbor:", best_K)  # Neighbor = 15
print("KNN - Best K's Average Accuracy:", round(knn_average_accuracies[best_K - 1] * 100, 3), "%")  # 80.723%

# Bayes
bayes_accuracies = []
bayes_precisions = []
bayes_recalls = []
bayes_f1_scores = []

for train_index, test_index in kf.split(scaler_X):
    X_train, X_test = scaler_X[train_index], scaler_X[test_index]
    y_train, y_test = label[train_index], label[test_index]

    # Create and fit the Naive Bayes model
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = naive_bayes.predict(X_test)

    # cnf_matrix_gnb = confusion_matrix(y_test, y_pred)
    # print(cnf_matrix_gnb)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append metrics to the lists
    bayes_accuracies.append(accuracy)
    bayes_precisions.append(precision)
    bayes_recalls.append(recall)
    bayes_f1_scores.append(f1)

# Calculate the average accuracy over all folds
average_accuracy = np.mean(bayes_accuracies)
average_precision = np.mean(bayes_precisions)
average_recall = np.mean(bayes_recalls)
average_f1_score = np.mean(bayes_f1_scores)

print("Naive Bayes Average Accuracy:", round(average_accuracy * 100, 2), "%")
print("Naive Bayes Average Precision:", round(average_precision * 100, 2), "%")
print("Naive Bayes Average Recall:", round(average_recall * 100, 2), "%")
print("Naive Bayes Average F1 Score:", round(average_f1_score * 100, 2), "%")

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

# Random Forest
rf_accuracies = []
rf_precisions = []
rf_recalls = []
rf_f1_scores = []

for train_index, test_index in kf.split(scaler_X):
    X_train, X_test = scaler_X[train_index], scaler_X[test_index]
    y_train, y_test = label[train_index], label[test_index]

    # Create and fit the Random Forest model
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)

    # Predict and calculate metrics
    y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append metrics to the lists
    rf_accuracies.append(accuracy)
    rf_precisions.append(precision)
    rf_recalls.append(recall)
    rf_f1_scores.append(f1)

# Calculate the average metrics over all folds for Random Forest
average_accuracy_rf = np.mean(rf_accuracies)
average_precision_rf = np.mean(rf_precisions)
average_recall_rf = np.mean(rf_recalls)
average_f1_score_rf = np.mean(rf_f1_scores)

print("Random Forest Average Accuracy:", round(average_accuracy_rf * 100, 2), "%")
print("Random Forest Average Precision:", round(average_precision_rf * 100, 2), "%")
print("Random Forest Average Recall:", round(average_recall_rf * 100, 2), "%")
print("Random Forest Average F1 Score:", round(average_f1_score_rf * 100, 2), "%")

