# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your training code and any local modules
COPY . .

# Set environment variables for MLflow and AWS
ENV MLFLOW_TRACKING_URI=http://35.171.186.148:5000
ENV AWS_REGION=us-east-1
ENV MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com

# Optional: expose port if serving via FastAPI or Streamlit later
EXPOSE 8000

# Default command to run trainings
CMD ["python", "train.py"]