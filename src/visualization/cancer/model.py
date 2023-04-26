from custom_transformers import DiagnosisEncoder, DataFrameSelector

import pickle
import numpy as np
# from joblib import load


def load_preprocessor():
    with open('preprocessor_pipeline.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
    return preprocessor


preprocessor = load_preprocessor()


def load_model():
    with open('model/logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


def predict_diagnosis(model, features):
    features_array = np.array(features).reshape(1, -1)

    # Preprocess the input features using the preprocessor pipeline
    preprocessed_features = preprocessor.transform(features_array)

    prediction_proba = model.predict_proba(preprocessed_features)
    return prediction_proba
