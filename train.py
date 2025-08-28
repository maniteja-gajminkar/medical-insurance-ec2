import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
import os

# ✅ Set MLflow tracking URI to your S3 bucket
mlflow.set_tracking_uri("s3://mlflow-artifacts-maniteja")
mlflow.set_experiment("medical-insurance")

# 🧠 Dummy training data
X = pd.DataFrame({'age': [25, 32, 47], 'bmi': [22.5, 28.1, 31.2]})
y = pd.Series([2500, 3200, 4100])

# 🚀 Train model
model = LinearRegression()
model.fit(X, y)

# 💾 Save model locally
model_path = "model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# 📦 Log to MLflow
with mlflow.start_run():
    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

print(f"✅ Model trained and logged to S3 via MLflow. Local path: {os.path.abspath(model_path)}")