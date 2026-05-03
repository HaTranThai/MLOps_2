import os

MODEL_TYPE = os.getenv("MODEL_TYPE", "classification")

SUPPORTED_MODEL_TYPES = [
    "classification",
    "regression",
    "clustering",
    "time_series"
]

if MODEL_TYPE not in SUPPORTED_MODEL_TYPES:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")


RAW_DATA_PATHS = {
    "classification": "data/raw/classification.csv",
    "regression": "data/raw/regression.csv",
    "clustering": "data/raw/clustering.csv",
    "time_series": "data/raw/time_series.csv"
}


PROCESSED_DATA_DIRS = {
    "classification": "data/processed/classification",
    "regression": "data/processed/regression",
    "clustering": "data/processed/clustering",
    "time_series": "data/processed/time_series"
}


MODEL_PATHS = {
    "classification": "models/classification_model.pkl",
    "regression": "models/regression_model.pkl",
    "clustering": "models/clustering_model.pkl",
    "time_series": "models/time_series_model.pkl"
}


EXPERIMENT_NAMES = {
    "classification": "mlops-classification-experiment",
    "regression": "mlops-regression-experiment",
    "clustering": "mlops-clustering-experiment",
    "time_series": "mlops-time-series-experiment"
}


REGISTERED_MODEL_NAMES = {
    "classification": "classification_model",
    "regression": "regression_model",
    "clustering": "clustering_model",
    "time_series": "time_series_model"
}


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)


MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")