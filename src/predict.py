import argparse
import json

import joblib
import pandas as pd

from src.config import MODEL_PATHS, MODEL_TYPE
from src.tasks import get_task_components


def predict_one(features):
    model = joblib.load(MODEL_PATHS[MODEL_TYPE])
    task = get_task_components()

    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)

    return task["format_prediction"](prediction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        required=True,
        help='JSON object, for example: {"feature_1": 5.1, "feature_2": 3.5, "feature_3": 1.4}',
    )
    args = parser.parse_args()

    features = json.loads(args.features)
    result = predict_one(features)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
