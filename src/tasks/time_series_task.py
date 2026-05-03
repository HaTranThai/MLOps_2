import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_model():
    return RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }

    return metrics


def format_prediction(prediction):
    return {
        "forecast": float(prediction[0]),
        "type": "time_series",
    }
