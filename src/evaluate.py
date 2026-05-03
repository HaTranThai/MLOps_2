import joblib
import pandas as pd

from src.config import MODEL_PATHS, MODEL_TYPE, PROCESSED_DATA_DIRS
from src.tasks import get_task_components
from src.train import split_features_target


def evaluate_saved_model():
    processed_dir = PROCESSED_DATA_DIRS[MODEL_TYPE]
    train_df = pd.read_csv(f"{processed_dir}/train.csv")
    test_df = pd.read_csv(f"{processed_dir}/test.csv")

    _, X_test, _, y_test = split_features_target(train_df, test_df)

    model = joblib.load(MODEL_PATHS[MODEL_TYPE])
    task = get_task_components()
    metrics = task["evaluate_model"](model, X_test, y_test)

    print(f"Evaluation completed for MODEL_TYPE={MODEL_TYPE}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    return metrics


if __name__ == "__main__":
    evaluate_saved_model()
