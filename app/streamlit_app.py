import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

MODEL_TYPE = os.getenv("MODEL_TYPE", "classification")
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")


st.set_page_config(
    page_title="MLOps Multi-Task UI",
    page_icon=":bar_chart:",
    layout="centered",
)

st.title("MLOps Multi-Task Prediction App")

st.write(f"Current model type: **{MODEL_TYPE}**")

st.divider()


def get_input_features():
    if MODEL_TYPE in ["classification", "regression", "clustering"]:
        feature_1 = st.number_input("Feature 1", value=5.1)
        feature_2 = st.number_input("Feature 2", value=3.5)
        feature_3 = st.number_input("Feature 3", value=1.4)

        return {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_3": feature_3,
        }

    if MODEL_TYPE == "time_series":
        lag_1 = st.number_input("Lag 1", value=120.0)
        lag_2 = st.number_input("Lag 2", value=115.0)
        lag_3 = st.number_input("Lag 3", value=130.0)

        return {
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
        }

    st.error("Invalid MODEL_TYPE.")
    return {}


features = get_input_features()

st.subheader("Input Data")
st.dataframe(pd.DataFrame([features]))

if st.button("Predict"):
    payload = {
        "features": features,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            st.subheader("Result")
            st.success(result)
        else:
            st.error("API returned an error.")
            st.write(response.text)
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI.")
