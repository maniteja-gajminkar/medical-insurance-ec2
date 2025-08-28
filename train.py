import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys

# 🔧 MLflow setup
MLFLOW_URI = "http://http://35.171.186.148/:5000"  # Replace with actual IP or use os.getenv("MLFLOW_URI")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("medical-insurance")

try:
    print(f"📡 Connecting to MLflow at {MLFLOW_URI}")
    
    # 📂 Load data
    df = pd.read_csv("raw_data/medical_insurance.csv")
    X = df.drop("charges", axis=1)
    y = df["charges"]

    # 🧪 Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🧠 Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 📊 Evaluation
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    # 📝 MLflow logging
    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_artifact("raw_data/medical_insurance.csv")
        mlflow.sklearn.log_model(model, "model")
        print(f"✅ Training complete. RMSE: {rmse:.2f}")
        print("🔗 View run in MLflow UI:", MLFLOW_URI)

except Exception as e:
    print("❌ Training failed:", str(e))
    sys.exit(1)