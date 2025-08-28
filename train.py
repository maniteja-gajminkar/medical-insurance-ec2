# train.py — Log MLflow artifacts to S3 (no server)

import os
import random
import hashlib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

# ---------------------------
# Config
# ---------------------------
SEED = 42
EXP_NAME = "medical_insurance_s3"   # use a fresh name so its artifact_location is S3
S3_BUCKET_URI = "s3://mlflow-artifacts-maniteja"  # must exist & be writeable

# Optional: make MLflow network calls a bit snappier
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

# Determinism
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Load & preprocess
# ---------------------------
df = pd.read_csv("raw_data/medical_insurance.csv")

target_col = "charges"
feature_cols = ["age", "sex", "bmi", "children", "smoker", "region"]

X = df[feature_cols].copy()
y = df[target_col].astype(float)

# one-hot encode categoricals
X = pd.get_dummies(X, columns=["sex", "smoker", "region"], drop_first=True)

# cast ints to float to avoid MLflow's missing-int warning later
for c in X.select_dtypes(include=["int32", "int64"]).columns:
    X[c] = X[c].astype("float64")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ---------------------------
# Train
# ---------------------------
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== Results ===")
print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2  : {r2:.4f}")

# ---------------------------
# MLflow: ensure experiment uses S3 artifact root
# ---------------------------
exp = mlflow.get_experiment_by_name(EXP_NAME)
if exp is None:
    exp_id = mlflow.create_experiment(name=EXP_NAME, artifact_location=S3_BUCKET_URI)
else:
    exp_id = exp.experiment_id
    if exp.artifact_location and not exp.artifact_location.startswith("s3://"):
        # Existing experiment points to local file store; create a fresh S3-backed experiment
        EXP_NAME = EXP_NAME + "_s3"
        exp_id = mlflow.create_experiment(name=EXP_NAME, artifact_location=S3_BUCKET_URI)

mlflow.set_experiment(EXP_NAME)

# ---------------------------
# Log run
# ---------------------------
with mlflow.start_run(run_name="linear_regression_simple"):

    mlflow.log_params({
        "model": "LinearRegression",
        "seed": SEED,
        "test_size": 0.2,
        "random_state": SEED,
    })

    mlflow.log_metrics({
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    print("Artifact URI =>", mlflow.get_artifact_uri())

    # IMPORTANT: use artifact_path (not name=)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.head(2),
        # signature/conda/requirements are auto-inferred; you can pass explicitly if you want
    )

print("✅ Training + MLflow logging complete.")
