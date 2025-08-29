import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import os
import pickle

# 🔧 MLflow config
mlflow.set_tracking_uri("http://35.171.186.148:5000")
mlflow.set_experiment("medical-insurance")

# 📥 Load dataset
df = pd.read_csv("raw_data/medical_insurance.csv")

# 🎯 Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# 🧼 Preprocessing
categorical = ["sex", "smoker", "region"]
numerical = ["age", "bmi", "children"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical),
    ("num", StandardScaler(), numerical)
])

# 🧠 Model pipeline
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

# 📊 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Start MLflow run
with mlflow.start_run() as run:
    pipeline.fit(X_train, y_train)

    # ✅ Log model
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("r2_score", pipeline.score(X_test, y_test))
    mlflow.sklearn.log_model(pipeline, artifact_path="model", input_example=X.iloc[:1])

    # 📦 Register model
    model_uri = f"runs:/{run.info.run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name="medical-insurance")

    # 🚦 Promote to Production
    client = MlflowClient()
    client.transition_model_version_stage(
        name="medical-insurance",
        version=result.version,
        stage="Production"
    )

    print(f"✅ Registered and promoted model: {result.name} v{result.version}")

# 💾 Save locally for S3 upload
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)