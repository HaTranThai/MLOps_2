import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import (
    MODEL_TYPE,
    MODEL_PATHS,
    EXPERIMENT_NAMES,
    REGISTERED_MODEL_NAMES,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    PROCESSED_DATA_DIRS
)
from src.tasks import get_task_components


def split_features_target(train_df, test_df):
    if MODEL_TYPE == "clustering":
        X_train = train_df
        X_test = test_df
        y_train = None
        y_test = None
    elif MODEL_TYPE == "time_series":
        feature_cols = ["lag_1", "lag_2", "lag_3"]
        target_col = "value"

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
    else:
        target_col = "target"

        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]

        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test


def train_model():
    processed_dir = PROCESSED_DATA_DIRS[MODEL_TYPE]
    train_path = f"{processed_dir}/train.csv"
    test_path = f"{processed_dir}/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, X_test, y_train, y_test = split_features_target(train_df, test_df)

    task = get_task_components()
    model = task["get_model"]()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAMES[MODEL_TYPE])

    with mlflow.start_run():
        if MODEL_TYPE == "clustering":
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)

        metrics = task["evaluate_model"](model, X_test, y_test)

        mlflow.log_param("model_type", MODEL_TYPE)
        mlflow.log_param("algorithm", model.__class__.__name__)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        model_path = MODEL_PATHS[MODEL_TYPE]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="registered_model_name"
        )

        print(f"Training completed for MODEL_TYPE={MODEL_TYPE}")
        print("Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    train_model()
