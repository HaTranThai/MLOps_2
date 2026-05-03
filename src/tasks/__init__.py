from src.config import MODEL_TYPE

from src.tasks.classification_task import evaluate_model as evaluate_classification_model
from src.tasks.classification_task import format_prediction as format_classification_prediction
from src.tasks.classification_task import get_model as get_classification_model
from src.tasks.clustering_task import evaluate_model as evaluate_clustering_model
from src.tasks.clustering_task import format_prediction as format_clustering_prediction
from src.tasks.clustering_task import get_model as get_clustering_model
from src.tasks.regression_task import evaluate_model as evaluate_regression_model
from src.tasks.regression_task import format_prediction as format_regression_prediction
from src.tasks.regression_task import get_model as get_regression_model
from src.tasks.time_series_task import evaluate_model as evaluate_time_series_model
from src.tasks.time_series_task import format_prediction as format_time_series_prediction
from src.tasks.time_series_task import get_model as get_time_series_model


def get_task_components():
    if MODEL_TYPE == "classification":
        return {
            "get_model": get_classification_model,
            "evaluate_model": evaluate_classification_model,
            "format_prediction": format_classification_prediction,
        }

    if MODEL_TYPE == "regression":
        return {
            "get_model": get_regression_model,
            "evaluate_model": evaluate_regression_model,
            "format_prediction": format_regression_prediction,
        }

    if MODEL_TYPE == "clustering":
        return {
            "get_model": get_clustering_model,
            "evaluate_model": evaluate_clustering_model,
            "format_prediction": format_clustering_prediction,
        }

    if MODEL_TYPE == "time_series":
        return {
            "get_model": get_time_series_model,
            "evaluate_model": evaluate_time_series_model,
            "format_prediction": format_time_series_prediction,
        }

    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")
