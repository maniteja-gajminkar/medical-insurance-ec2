from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

app = FastAPI(title="MLflow Model API", description="FastAPI wrapper for MLflow model", version="1.0")

# Load latest model dynamically
client = MlflowClient()
experiment_id = "1"
latest_run = client.search_runs([experiment_id], order_by=["start_time DESC"], max_results=1)[0]
model_uri = f"runs:/{latest_run.info.run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

class InputData(BaseModel):
    age: int
    bmi: float

@app.get("/health")
def health_check():
    return {"status": "ok", "model_run_id": latest_run.info.run_id}

@app.get("/metadata")
def model_metadata():
    return {
        "run_id": latest_run.info.run_id,
        "experiment_id": experiment_id,
        "artifact_uri": latest_run.info.artifact_uri
    }

@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"prediction": prediction[0]}