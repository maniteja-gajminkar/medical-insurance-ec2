import mlflow
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("medical-insurance")

# Dummy training logic
df = pd.read_csv("insurance.csv")
X = df[["age", "bmi", "children"]]
y = df["charges"]

model = LinearRegression()
model.fit(X, y)
preds = model.predict(X)
mse = mean_squared_error(y, preds)

# Save model
joblib.dump(model, "model.pkl")

# Log to MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_artifact("model.pkl")
    mlflow.log_artifact("insurance.csv")