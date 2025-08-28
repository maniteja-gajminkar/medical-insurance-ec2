import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import os
import pickle
from mlflow.models.signature import infer_signature

# Connect to MLflow tracking server
mlflow.set_tracking_uri("http://35.171.186.148:5000")
mlflow.set_experiment("medical-insurance")

# Sample training data
X = pd.DataFrame({'age': [25, 32, 47], 'bmi': [22.0, 28.5, 31.2]})
y = np.array([2500, 3200, 4100])

# Start MLflow run
with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X, y)

    # Log metadata
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("r2_score", model.score(X, y))
    mlflow.set_tag("owner", "Maniteja")
    mlflow.set_tag("purpose", "Interview demo")

    # Log model with signature
    signature = infer_signature(X, model.predict(X))
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=X.iloc[:1],
        signature=signature
    )

    # Save model locally for S3 upload
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save run ID for downstream use
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)