import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Read MLflow URI from env (set by the workflow), fallback for local tests
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
print("Using MLflow URI:", MLFLOW_URI)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("medical_insurance")

# Load dataset
df = pd.read_csv("raw_data/medical_insurance.csv")
target_col = "charges"
feature_cols = ["age", "sex", "bmi", "children", "smoker", "region"]

X = pd.get_dummies(df[feature_cols], columns=["sex", "smoker", "region"], drop_first=True)
y = df[target_col].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== Results ===")
print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2  : {r2:.4f}")

# Log to MLflow
with mlflow.start_run(run_name="linear_regression"):
    mlflow.log_params({
        "model": "LinearRegression",
        "test_size": 0.2,
        "random_state": 42
    })
    mlflow.log_metrics({
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=X_train.head(2),
    )

print("✅ Training run complete. Check MLflow UI at:", MLFLOW_URI)