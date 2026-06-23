import streamlit as st
import requests

st.title("MNIST Digit Predictor")

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    files = {
        "file": uploaded_file.getvalue()
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files={"file": uploaded_file}
    )

    st.write("Prediction:")

    st.success(response.json()["prediction"])