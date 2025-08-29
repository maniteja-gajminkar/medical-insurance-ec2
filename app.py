from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import load_model
import logging

app = FastAPI()

# 🔧 Configuration
MLFLOW_TRACKING_URI = "http://35.171.186.148:5000"
MODEL_NAME = "medical-insurance"
MODEL_STAGE = "Production"

# 🧠 Initialize MLflow client
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# 🩺 Health check logic
def check_model_status():
    try:
        latest_versions = client.get_latest_versions(name=MODEL_NAME, stages=[MODEL_STAGE])
        if latest_versions:
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            model_status = "ready"
        else:
            model_uri = None
            model_status = "not found"
    except Exception as e:
        logging.error(f"Error querying MLflow: {e}")
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

# 🧾 Define input schema
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
    if model_status != "ready":
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        model = load_model(model_uri)
        input_df = input_data.dict()
        prediction = model.predict([input_df])
        return {"prediction": prediction[0]}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")