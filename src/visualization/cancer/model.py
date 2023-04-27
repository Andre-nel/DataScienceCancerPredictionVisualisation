from custom_transformers import DiagnosisEncoder, DataFrameSelector

import pickle
import numpy as np
# from joblib import load


def load_preprocessor(path='preprocessor_pipeline.pkl'):
    with open(path, 'rb') as file:
        preprocessor = pickle.load(file)
    return preprocessor


preprocessor = load_preprocessor(path="C:/Users/candr/OneDrive/Desktop/Masters/DataScience/DataScienceCancerPredictionVisualisation/src/visualization/cancer/preprocessor_pipeline.pkl")


def load_model(path='model/logistic_regression_model.pkl'):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model


def predict_diagnosis(model, features):
    features_array = np.array(features).reshape(1, -1)

    # Preprocess the input features using the preprocessor pipeline
    preprocessed_features = preprocessor.transform(features_array)

    prediction_proba = model.predict_proba(preprocessed_features)
    return prediction_proba
