import os
import streamlit as st
import requests
import pandas as pd

MODEL_TYPE = os.getenv("MODEL_TYPE", "classification")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


st.set_page_config(
    page_title="MLOps Multi-Task UI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 MLOps Multi-Task Prediction App")

st.write(f"Model type hiện tại: **{MODEL_TYPE}**")

st.info("FastAPI sẽ load model từ MLflow Model Registry bằng alias `champion`.")

st.divider()


def get_input_features():
    if MODEL_TYPE in ["classification", "regression", "clustering"]:
        feature_1 = st.number_input("Feature 1", value=5.1)
        feature_2 = st.number_input("Feature 2", value=3.5)
        feature_3 = st.number_input("Feature 3", value=1.4)

        return {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_3": feature_3
        }

    if MODEL_TYPE == "time_series":
        lag_1 = st.number_input("Lag 1", value=120.0)
        lag_2 = st.number_input("Lag 2", value=115.0)
        lag_3 = st.number_input("Lag 3", value=130.0)

        return {
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3
        }

    st.error("MODEL_TYPE không hợp lệ.")
    return {}


features = get_input_features()

st.subheader("Dữ liệu đầu vào")
st.dataframe(pd.DataFrame([features]))

if st.button("Dự đoán"):
    payload = {
        "features": features
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.subheader("Kết quả")
            st.success(result)

            if "model_uri" in result:
                st.caption(f"Model URI: {result['model_uri']}")

        else:
            st.error("API trả về lỗi.")
            st.code(response.text)

    except requests.exceptions.ConnectionError:
        st.error("Không kết nối được đến FastAPI.")