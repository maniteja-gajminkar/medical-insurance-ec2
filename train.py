import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "medical-insurance"
DATA_PATH = "raw_data/medical_insurance.csv"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print("📡 Tracking URI:", mlflow.get_tracking_uri())
print("🔬 Experiment:", EXPERIMENT_NAME)

if not os.path.exists(DATA_PATH):
    print(f"❌ Missing dataset at {DATA_PATH}", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
if "charges" not in df.columns:
    print("❌ 'charges' column not found in dataset.", file=sys.stderr)
    sys.exit(1)

X = pd.get_dummies(df.drop("charges", axis=1), drop_first=True)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
signature = infer_signature(X_test, preds)

with mlflow.start_run():
    mlflow.set_tags({
        "project": "medical-insurance",
        "developer": "maniteja",
        "model_type": "LinearRegression",
    })
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_artifact(DATA_PATH)
    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

print(f"✅ Training complete. RMSE: {rmse:.2f}")
print("🔗 MLflow UI:", MLFLOW_URI)
