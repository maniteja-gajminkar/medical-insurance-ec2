from fastapi import FastAPI
import mlflow
from mlflow.exceptions import RestException

app = FastAPI()

# Try to load the registered model
try:
    model = mlflow.pyfunc.load_model("models:/medical-insurance/Production")
    model_status = "loaded"
except RestException as e:
    print(f"⚠️ Model not found: {e}")
    model = None
    model_status = "not found"

@app.get("/health")
def health_check():
    return {
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "model_status": model_status
    }