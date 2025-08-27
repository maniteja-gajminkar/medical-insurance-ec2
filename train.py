# train.py  (ultra simple)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# 1) Read data
df = pd.read_csv("raw_data/medical_insurance.csv")

# Expect these columns: age, sex, bmi, children, smoker, region, charges
# If your CSV uses different names, adjust below.
target_col = "charges"
feature_cols = ["age", "sex", "bmi", "children", "smoker", "region"]

# 2) Split features/target
X = df[feature_cols]
y = df[target_col]

# 3) One-hot encode categoricals the simple way
X = pd.get_dummies(X, columns=["sex", "smoker", "region"], drop_first=True)

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) Train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# 6) Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== Results ===")
print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2  : {r2:.4f}")

# 7) Log to MLflow (stores under ./mlruns by default)
mlflow.set_experiment("medical_insurance")
with mlflow.start_run(run_name="linear_regression_simple"):
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head(2))

print("\n✅ Training done. Metrics logged to MLflow (./mlruns).")
