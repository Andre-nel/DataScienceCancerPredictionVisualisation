# from custom_transformers import DiagnosisEncoder, DataFrameSelector

import pickle
import numpy as np
# from joblib import load


def load_preprocessor(path):
    with open(path, 'rb') as file:
        preprocessor = pickle.load(file)
    return preprocessor


def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model


def predict_diagnosis(model, features, path_to_preprocessor_pipeline):
    features_array = np.array(features).reshape(1, -1)

    preprocessor = load_preprocessor(path=path_to_preprocessor_pipeline)

    # Preprocess the input features using the preprocessor pipeline
    preprocessed_features = preprocessor.transform(features_array)

    prediction_proba = model.predict_proba(preprocessed_features)
    return prediction_proba
