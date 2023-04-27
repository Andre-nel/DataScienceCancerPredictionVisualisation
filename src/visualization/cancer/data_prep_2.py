import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from pathlib import Path
import pickle


def create_pipeline():

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("power_transformer", PowerTransformer()),
        ("smote", SMOTE(sampling_strategy="auto", random_state=42)),
        ("pca", PCA(n_components=0.95))
    ])

    return pipeline


def preprocess_cancer_data(data, pipeline, target=None, keep_all_features=False):

    data_scaled = pipeline.named_steps["scaler"].fit_transform(data)
    data_power_transformed = pipeline.named_steps["power_transformer"].fit_transform(data_scaled)

    if target is not None:
        data_resampled, target_resampled = pipeline.named_steps["smote"].fit_resample(data_power_transformed, target)
    else:
        data_resampled = data_power_transformed
        target_resampled = None

    if keep_all_features:
        return pd.DataFrame(data_resampled, columns=data.columns), target_resampled
    
    data_pca = pipeline.named_steps["pca"].fit_transform(data_resampled)

    preprocessed_data = pd.DataFrame(data_pca, columns=[
        f'PC_{i}' for i in range(1, pipeline.named_steps['pca'].n_components_ + 1)
    ])

    return preprocessed_data, target_resampled


if __name__ == "__main__":
    projectRoot = Path.cwd()
    raw_data_path = projectRoot / "data/raw/cancerData.csv"
    preprocessed_data_path = projectRoot / "data/preprocessed/cancer_data_all_features.csv"
    pipeline_path = projectRoot / "src/visualization/cancer/preprocessor_pipeline_options.pkl"

    data = pd.read_csv(raw_data_path)
    data.drop(columns=["Unnamed: 32", "id"], inplace=True)
    target_data = data["diagnosis"].map({"M": 1, "B": 0})
    data['diagnosis_numeric'] = data["diagnosis"].map({"M": 1, "B": 0})
    feature_data = data.drop(columns=["diagnosis", "diagnosis_numeric"])
    target_data = data["diagnosis_numeric"]

    preprocessor_pipeline = create_pipeline()
    preprocessed_feature_data, diagnosis_resampled = preprocess_cancer_data(
        feature_data, preprocessor_pipeline, target_data, keep_all_features=True)

    if diagnosis_resampled is not None:
        diagnosis_resampled = pd.Series(diagnosis_resampled, name="diagnosis")
        preprocessed_data = pd.concat([preprocessed_feature_data, diagnosis_resampled], axis=1)
    else:
        preprocessed_data = preprocessed_feature_data

    preprocessed_data.to_csv(preprocessed_data_path, index=False)

    # Save the trained model to a pickle file
    with open(pipeline_path, 'wb') as f:
        pickle.dump(preprocessor_pipeline, f)
