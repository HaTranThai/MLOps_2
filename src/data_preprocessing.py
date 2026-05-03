import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import MODEL_TYPE, PROCESSED_DATA_DIRS, RAW_DATA_PATHS


def preprocess_classification_or_regression(df):
    df = df.dropna()

    stratify = df["target"] if MODEL_TYPE == "classification" else None

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    return train_df, test_df


def preprocess_clustering(df):
    df = df.dropna()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )

    return train_df, test_df


def preprocess_time_series(df):
    df = df.dropna()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["lag_1"] = df["value"].shift(1)
    df["lag_2"] = df["value"].shift(2)
    df["lag_3"] = df["value"].shift(3)

    df = df.dropna()

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    return train_df, test_df


def preprocess_data():
    raw_path = RAW_DATA_PATHS[MODEL_TYPE]
    df = pd.read_csv(raw_path)

    if MODEL_TYPE in ["classification", "regression"]:
        train_df, test_df = preprocess_classification_or_regression(df)
    elif MODEL_TYPE == "clustering":
        train_df, test_df = preprocess_clustering(df)
    elif MODEL_TYPE == "time_series":
        train_df, test_df = preprocess_time_series(df)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

    output_dir = PROCESSED_DATA_DIRS[MODEL_TYPE]
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"Preprocessing completed for MODEL_TYPE={MODEL_TYPE}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")


if __name__ == "__main__":
    preprocess_data()
