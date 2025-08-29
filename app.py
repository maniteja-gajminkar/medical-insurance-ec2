from fastapi import FastAPI, Request
import mlflow
from mlflow.exceptions import RestException
import logging

app = FastAPI()

# 🔧 Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🌐 MLflow model registry config
MODEL_NAME = "medical-insurance"
MODEL_STAGE = "Production"
model = None
model_status = "not found"

# 🚀 Try to load the registered model
try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    model_status = "loaded"
    logger.info(f"✅ Loaded model from {model_uri}")
except RestException as e:
    logger.warning(f"⚠️ Model not found: {e}")
except Exception as e:
    logger.error(f"🔥 Unexpected error loading model: {e}")

# 🩺 Health check endpoint
@app.get("/health")
def health_check():
    return {
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_status": model_status
    }

# 🔮 Optional: Predict endpoint (ready for extension)
@app.post("/predict")
async def predict(request: Request):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        payload = await request.json()
        input_data = [list(payload.values())]
        prediction = model.predict(input_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}