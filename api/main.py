import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from src.config import (
    MODEL_TYPE,
    REGISTERED_MODEL_NAMES,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS
)

from src.tasks import get_task_components


registered_model_name = REGISTERED_MODEL_NAMES[MODEL_TYPE]
MODEL_URI = f"models:/{registered_model_name}@{MODEL_ALIAS}"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="MLOps Multi-Task API with MLflow Registry",
    description="Serving classification, regression, clustering and time series models from MLflow Model Registry",
    version="2.0.0"
)


class InputData(BaseModel):
    features: Dict[str, Any]


model = mlflow.pyfunc.load_model(MODEL_URI)
task = get_task_components()


@app.get("/")
def home():
    return {
        "message": "MLOps Multi-Task API is running",
        "model_type": MODEL_TYPE,
        "registered_model_name": registered_model_name,
        "model_uri": MODEL_URI
    }


@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.features])

    prediction = model.predict(input_df)

    result = task["format_prediction"](prediction)
    result["model_type"] = MODEL_TYPE
    result["registered_model_name"] = registered_model_name
    result["model_uri"] = MODEL_URI

    return result