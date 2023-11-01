import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

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
print(dataset.head(5))

# Display dataset to evaluate whether data needs cleaning
print(dataset.describe().transpose())
print(dataset['Severity'].value_counts())

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
print(dataset[(dataset['Age'].isnull()) |
              (dataset['Shape'].isnull()) |
              (dataset['Margin'].isnull()) |
              (dataset['Density'].isnull())])

# Display detail total NaN for each attribute
print(dataset.isnull().sum())

# Data seems randomly distributed, so drop all rows having missing value
new_dataset = dataset.dropna()

attributes = new_dataset.drop(['Severity', 'BI-RADS'], axis=1)
label = np.array(new_dataset['Severity'])

# New dataset with 830 rows
print(new_dataset)

# Check new dataset is having missing value?
print(new_dataset.isnull().sum())
