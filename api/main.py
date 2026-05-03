from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import MODEL_PATHS, MODEL_TYPE
from src.tasks import get_task_components


app = FastAPI(
    title="MLOps Multi-Task API",
    description="Serving classification, regression, clustering and time series models",
    version="1.0.0",
)


class InputData(BaseModel):
    features: Dict[str, Any]


def load_model():
    model_path = MODEL_PATHS[MODEL_TYPE]

    try:
        return joblib.load(model_path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model file not found at {model_path}. Run preprocessing and training first."
        ) from exc


model = load_model()
task = get_task_components()


@app.get("/")
def home():
    return {
        "message": "MLOps Multi-Task API is running",
        "model_type": MODEL_TYPE,
    }


@app.post("/predict")
def predict(data: InputData):
    try:
        input_df = pd.DataFrame([data.features])
        prediction = model.predict(input_df)
        return task["format_prediction"](prediction)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
