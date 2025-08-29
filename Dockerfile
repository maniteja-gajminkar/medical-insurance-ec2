# 🐍 Use a lightweight Python base image
FROM python:3.10-slim

# 📁 Set working directory
WORKDIR /app

# 📦 Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 📂 Copy application code
COPY . .

# 🌍 Set environment variables for MLflow and AWS S3
ENV MLFLOW_TRACKING_URI=http://35.171.186.148:5000
ENV MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
ENV AWS_REGION=us-east-1

# 🔓 Expose FastAPI port
EXPOSE 8000

# 🚀 Run FastAPI app with Gunicorn + UvicornWorker
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8000"]