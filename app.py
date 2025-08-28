import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Get latest run from MLflow
client = MlflowClient()
experiment_id = "1"
latest_run = client.search_runs([experiment_id], order_by=["start_time DESC"], max_results=1)[0]
model_uri = f"runs:/{latest_run.info.run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# UI
st.title("🧠 MLflow Model Predictor")
st.write(f"📌 Latest Run ID: `{latest_run.info.run_id}`")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame([{"age": age, "bmi": bmi}])
    prediction = model.predict(input_df)
    st.success(f"🎯 Prediction: {prediction[0]}")