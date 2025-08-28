import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import os
import sys

# 🔧 Use MLflow URI from environment (injected via GitHub secret)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("medical-insurance")

try:
    print(f"📡 Connecting to MLflow at {MLFLOW_URI}")

    # 📂 Load data
    data_path = "raw_data/medical_insurance.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset at {data_path}")
    df = pd.read_csv(data_path)

    # 🧼 Encode categorical features
    X = pd.get_dummies(df.drop("charges", axis=1), drop_first=True)
    y = df["charges"]

    # 🧪 Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🧠 Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 📊 Evaluation
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    signature = infer_signature(X_test, predictions)

    # 📝 MLflow logging
    with mlflow.start_run():
        mlflow.set_tags({
            "project": "medical-insurance",
            "developer": "maniteja",
            "model_type": "LinearRegression"
        })
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_artifact(data_path)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        print(f"✅ Training complete. RMSE: {rmse:.2f}")
        print("🔗 View run in MLflow UI:", MLFLOW_URI)

except Exception as e:
    print("❌ Training failed:", str(e))
    sys.exit(1)