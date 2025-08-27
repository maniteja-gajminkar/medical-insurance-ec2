# train.py (verbose + safe MLflow)
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("STEP 1: load CSV...")
df = pd.read_csv("raw_data/medical_insurance.csv")
print(f"  loaded {df.shape[0]} rows, {df.shape[1]} cols")

target_col = "charges"
feature_cols = ["age", "sex", "bmi", "children", "smoker", "region"]

print("STEP 2: build X,y...")
X = df[feature_cols]
y = df[target_col]

print("STEP 3: one-hot encode...")
X = pd.get_dummies(X, columns=["sex", "smoker", "region"], drop_first=True)
print(f"  X shape after OHE: {X.shape}")

print("STEP 4: split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("STEP 5: fit model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("  model.fit done")

print("STEP 6: evaluate...")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== Results ===")
print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2  : {r2:.4f}")

# -------- MLflow logging with fail-fast + fallback --------
print("STEP 7: MLflow logging...")
import mlflow, mlflow.sklearn

# Fail fast in CI (don’t hang for minutes)
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
print(f"  MLFLOW_TRACKING_URI={tracking_uri}")

def log_with_uri(uri):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("medical_insurance")
    with mlflow.start_run(run_name="linear_regression_simple"):
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        print("  logging model artifact...")
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head(2))
        run_id = mlflow.active_run().info.run_id
        print(f"  ✅ logged to {uri} | run_id={run_id}")

try:
    # try remote first
    log_with_uri(tracking_uri)
except Exception as e:
    print(f"  ⚠️ remote logging failed: {e}")
    # fallback to local file store so the job finishes
    fallback_uri = "file:./mlruns"
    print(f"  falling back to {fallback_uri}")
    try:
        log_with_uri(fallback_uri)
    except Exception as e2:
        print(f"  ❌ local logging also failed: {e2}")

print("✅ Training script finished.")
