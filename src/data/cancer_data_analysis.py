import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the data
projectRoot = Path(__file__).parent.parent.parent
data = pd.read_csv(projectRoot / "data/raw/cancerData.csv")

# Convert diagnosis to numeric values
data['diagnosis_numeric'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Data exploration
print("Data Overview:")
head = data.head().to_string()

with open(projectRoot / "reports" / "info" / "data_head.txt", "w") as f:
    f.write(head)


print("\nData Types:")
print(data.dtypes)
"""
Data Types:
id                           int64
diagnosis                   object
radius_mean                float64
texture_mean               float64
perimeter_mean             float64
area_mean                  float64
smoothness_mean            float64
compactness_mean           float64
concavity_mean             float64
concave points_mean        float64
symmetry_mean              float64
fractal_dimension_mean     float64
radius_se                  float64
texture_se                 float64
perimeter_se               float64
area_se                    float64
smoothness_se              float64
compactness_se             float64
concavity_se               float64
concave points_se          float64
symmetry_se                float64
fractal_dimension_se       float64
radius_worst               float64
texture_worst              float64
perimeter_worst            float64
area_worst                 float64
smoothness_worst           float64
compactness_worst          float64
concavity_worst            float64
concave points_worst       float64
symmetry_worst             float64
fractal_dimension_worst    float64
Unnamed: 32                float64
dtype: object
"""
# Save descriptive statistics to a file
description_str = data.describe()
print("\nDescriptive Statistics:")
print(description_str)
description_str = description_str.to_string()
with open(projectRoot / "reports" / "info" / "data_description.txt", "w") as f:
    f.write(description_str)

print("\nMissing Values:")
print(data.isna().sum())
missing_values_str = data.isna().sum().to_string()
with open(projectRoot / "reports" / "info" / "data_missing.txt", "w") as f:
    f.write(missing_values_str)
"""
Missing Values:
id                           0
diagnosis                    0
radius_mean                  0
texture_mean                 0
perimeter_mean               0
area_mean                    0
smoothness_mean              0
compactness_mean             0
concavity_mean               0
concave points_mean          0
symmetry_mean                0
fractal_dimension_mean       0
radius_se                    0
texture_se                   0
perimeter_se                 0
area_se                      0
smoothness_se                0
perimeter_worst              0
area_worst                   0
smoothness_worst             0
compactness_worst            0
concavity_worst              0
concave points_worst         0
symmetry_worst               0
fractal_dimension_worst      0
Unnamed: 32                569
diagnosis_numeric            0
dtype: int64
"""

# Visualization
# Histograms or box plots
data.hist(figsize=(20, 20))
plt.show()

# # Scatter plots or pair plots
# sns.pairplot(data, hue='diagnosis')
# plt.show()

# Correlation matrix or heatmap
corr_matrix = data.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# # Feature-target relationship analysis
# # Distribution of the target variable (diagnosis)
# # Convert diagnosis to numeric values
# data['diagnosis_numeric'] = data['diagnosis'].map({'M': 1, 'B': 0})
# sns.countplot(data['diagnosis_numeric'])
# plt.show()

# Relationships between individual features and the target variable (diagnosis)
# Using violin plots
plt.figure(figsize=(20, 20))
for i, column in enumerate(data.columns[2:31], 1):
    plt.subplot(6, 5, i)
    sns.violinplot(x='diagnosis', y=column, data=data)
plt.show()
