from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_model():
    return KMeans(
        n_clusters=3,
        random_state=42,
        n_init="auto",
    )


def evaluate_model(model, X_test, y_test=None):
    labels = model.predict(X_test)
    unique_labels = set(labels)

    if len(unique_labels) < 2 or len(unique_labels) >= len(X_test):
        return {
            "silhouette_score": 0.0,
        }

    metrics = {
        "silhouette_score": silhouette_score(X_test, labels),
    }

    return metrics


def format_prediction(prediction):
    return {
        "cluster": int(prediction[0]),
        "type": "clustering",
    }
