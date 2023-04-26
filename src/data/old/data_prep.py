import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
from imblearn.over_sampling import SMOTE


def load_data(filename):
    data = pd.read_csv(filename)
    data.drop(columns=["Unnamed: 32", "id"], inplace=True)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data

def handle_outliers(data):
    # Separate features and target variable
    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"]
    # Calculate IQR
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    # Find outlier indices
    outlier_indices = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    # Remove outliers from the features
    X_no_outliers = X[~outlier_indices]
    # Remove the corresponding target values
    y_no_outliers = y[~outlier_indices]
    # Concatenate the features without outliers with the target variable
    data_no_outliers = pd.concat([X_no_outliers, y_no_outliers.reset_index(drop=True)], axis=1)
    return data_no_outliers

def handle_class_imbalance(data):
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"]
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # Combine resampled features and target into one DataFrame
    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
    return resampled_data

def scale_features(data):
    scaler = StandardScaler()
    features = data.drop(columns=["diagnosis"])
    scaled_features = scaler.fit_transform(features)
    return pd.DataFrame(scaled_features, columns=features.columns)

def handle_multicollinearity(data):
    pca = PCA(n_components=0.95)
    transformed_data = pca.fit_transform(data)
    return pd.DataFrame(transformed_data)

def main():
    data = load_data("data.csv")
    data = handle_class_imbalance(data)
    data = scale_features(data)
    data = handle_multicollinearity(data)
    data = handle_outliers(data)
    # Save preprocessed data to a new CSV file
    data.to_csv("preprocessed_cancer_data.csv", index=False)

if __name__ == "__main__":
    main()
