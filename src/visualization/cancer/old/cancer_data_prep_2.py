import pandas as pd
# import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pathlib import Path
import pickle

# from custom_transformers import DataFrameSelector


def create_pipeline(input_df):

    # num_features = [
    #     "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    #     "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    #     "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    #     "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    #     "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    #     "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst",
    #     "fractal_dimension_worst"
    # ]

    pipeline = ImbPipeline([
        # ("diagnosis_encoder", DiagnosisEncoder()),
        # ("selector", DataFrameSelector(num_features, input_df)),
        # ("smote", SMOTE(sampling_strategy="auto", random_state=42)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95))
    ])

    return pipeline


def preprocess_cancer_data(data, pipeline):
    # diagnosis_encoder = pipeline.named_steps["diagnosis_encoder"]
    # diagnosis_encoder.fit(data)
    # data_encoded = diagnosis_encoder.transform(data)

    # selector = pipeline.named_steps["selector"]
    # selector.fit(data)
    # data_selected = selector.transform(data)

    data_scaled = pipeline.named_steps["scaler"].fit_transform(data)
    data_pca = pipeline.named_steps["pca"].fit_transform(data_scaled)

    preprocessed_data = pd.DataFrame(data_pca, columns=[
        f'PC_{i}' for i in range(1, pipeline.named_steps['pca'].n_components_ + 1)
    ])

    return preprocessed_data


if __name__ == "__main__":
    projectRoot = Path.cwd()
    raw_data_path = projectRoot / "data/raw/cancerData.csv"
    # raw_hold_out_data_path = projectRoot / "data/raw/cancel_holdOut.csv"
    preprocessed_data_path = projectRoot / "data/preprocessed/cancer_data.csv"
    # hold_out_preprocessed_data_path = projectRoot / "data/preprocessed/hold_out_cancer_data.csv"
    pipeline_path = projectRoot / "src/visualization/cancer/preprocessor_pipeline.pkl"

    data = pd.read_csv(raw_data_path)
    data.drop(columns=["Unnamed: 32", "id"], inplace=True)

    diagnosis_column = data["diagnosis"].map({"M": 1, "B": 0})
    feature_data = data.drop(columns=["diagnosis"])

    preprocessor_pipeline = create_pipeline(feature_data)
    preprocessed_feature_data = preprocess_cancer_data(feature_data, preprocessor_pipeline)

    preprocessed_data = pd.concat([preprocessed_feature_data, diagnosis_column], axis=1)

    preprocessed_data.to_csv(preprocessed_data_path, index=False)
    # dump(preprocessor_pipeline, pipeline_path)

    # save the trained model to a pickle file
    with open(pipeline_path, 'wb') as f:
        pickle.dump(preprocessor_pipeline, f)
