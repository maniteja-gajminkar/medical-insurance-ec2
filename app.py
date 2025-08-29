from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import load_model
import logging
import os
import pandas as pd

app = FastAPI()

# 🔧 Environment config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "medical-insurance")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# 🧠 MLflow client
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# 🩺 Health check logic
def check_model_status():
    try:
        latest_versions = client.get_latest_versions(name=MODEL_NAME, stages=[MODEL_STAGE])
        if latest_versions:
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            model_status = "ready"
        else:
            all_versions = client.get_latest_versions(name=MODEL_NAME)
            if all_versions:
                latest_version = all_versions[0].version
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=latest_version,
                    stage=MODEL_STAGE
                )
                model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
                model_status = "promoted"
            else:
                model_uri = None
                model_status = "not found"
    except Exception as e:
        logging.error(f"MLflow error: {e}")
        model_uri = None
        model_status = "error"
    return model_uri, model_status

@app.get("/health")
def health():
    model_uri, model_status = check_model_status()
    return {
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_status": model_status
    }

# 🧾 Input schema
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.post("/predict")
def predict(input_data: InsuranceInput):
    model_uri, model_status = check_model_status()
    if model_status not in ["ready", "promoted"]:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        model = load_model(model_uri)
        input_df = pd.DataFrame([input_data.dict()])
        prediction = model.predict(input_df)[0]
        return {"prediction": round(prediction, 2)}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")