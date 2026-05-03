from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def get_model():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    return metrics


def format_prediction(prediction):
    return {
        "prediction": int(prediction[0]),
        "type": "classification",
    }
