# train.py — MLflow logs artifacts to S3, no server needed
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Load and preprocess data ----
df = pd.read_csv("raw_data/medical_insurance.csv")
target_col = "charges"
feature_cols = ["age", "sex", "bmi", "children", "smoker", "region"]

X = df[feature_cols]
y = df[target_col]
X = pd.get_dummies(X, columns=["sex", "smoker", "region"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== Results ===")
print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2  : {r2:.4f}")

# ---- MLflow logging ----
import mlflow
import mlflow.sklearn

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

EXP_NAME = "medical_insurance"
S3_BUCKET_URI = "s3://mlflow-artifacts-maniteja"

exp = mlflow.get_experiment_by_name(EXP_NAME)
if exp is None:
    exp_id = mlflow.create_experiment(name=EXP_NAME, artifact_location=S3_BUCKET_URI)
else:
    exp_id = exp.experiment_id

mlflow.set_experiment(EXP_NAME)

with mlflow.start_run(run_name="linear_regression_simple"):
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    print("Artifact URI =>", mlflow.get_artifact_uri())

    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_train.head(2),
    )

print("✅ Training + MLflow logging complete.")