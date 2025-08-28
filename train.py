import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import os
import pickle

# Connect to MLflow tracking server
mlflow.set_tracking_uri("http://35.171.186.148:5000")
mlflow.set_experiment("medical-insurance")

# Sample training data
X = pd.DataFrame({'age': [25, 32, 47], 'bmi': [22.0, 28.5, 31.2]})
y = np.array([2500, 3200, 4100])

# Start MLflow run
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X, y)

    # Log parameters and metrics
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("r2_score", model.score(X, y))

    # Log model to MLflow tracking server
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=X.iloc[:1]
    )

    # Save model locally for GitHub Actions to upload to S3
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)