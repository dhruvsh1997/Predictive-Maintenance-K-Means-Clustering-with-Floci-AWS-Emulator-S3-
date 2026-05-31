"""
Predictive Maintenance API
- Loads KMeans model + scaler from floci S3 on startup.
- POST /predict → returns cluster label, appends row to predictions CSV in S3.
"""

import csv
import io
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import boto3
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config — set these env vars before starting the API
# ---------------------------------------------------------------------------
S3_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
AWS_KEY_ID  = os.environ.get("AWS_ACCESS_KEY_ID", "test")
AWS_SECRET  = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
AWS_REGION  = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET   = os.environ.get("S3_BUCKET", "predictive-maintenance")
S3_PREFIX   = os.environ.get("S3_MODEL_PREFIX", "model")
MODEL_DIR   = os.environ.get("MODEL_DIR", "/tmp/model")

PREDICTIONS_CSV_KEY = "logs/predictions.csv"
CSV_COLUMNS = [
    "timestamp",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "cluster_id",
    "health_state",
]

# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------
s3_client     = None
kmeans_model  = None
scaler        = None
feature_names = None
state_map     = None


def build_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY_ID,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )


def download_model_artifacts(s3):
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifacts = ["kmeans_model.pkl", "scaler.pkl", "state_map.json", "features.json"]
    for artifact in artifacts:
        s3_key     = f"{S3_PREFIX}/{artifact}"
        local_path = os.path.join(MODEL_DIR, artifact)
        print(f"Downloading s3://{S3_BUCKET}/{s3_key} → {local_path}")
        s3.download_file(S3_BUCKET, s3_key, local_path)
    print("All model artifacts downloaded.")


def log_prediction_to_s3(s3, data, cluster_id: int, health_state: str):
    """
    Download existing predictions CSV from S3 (if any), append a new row, upload back.
    Creates the CSV with headers on first call.
    """
    existing_rows = []

    # Try to fetch existing CSV
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=PREDICTIONS_CSV_KEY)
        content  = response["Body"].read().decode("utf-8")
        reader   = csv.DictReader(io.StringIO(content))
        existing_rows = list(reader)
    except s3.exceptions.NoSuchKey:
        pass  # First prediction — CSV doesn't exist yet
    except Exception as exc:
        print(f"Warning: could not read predictions CSV: {exc}")

    # Build new row
    new_row = {
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "air_temperature_k":     data.air_temperature_k,
        "process_temperature_k": data.process_temperature_k,
        "rotational_speed_rpm":  data.rotational_speed_rpm,
        "torque_nm":             data.torque_nm,
        "tool_wear_min":         data.tool_wear_min,
        "cluster_id":            cluster_id,
        "health_state":          health_state,
    }
    existing_rows.append(new_row)

    # Write updated CSV back to S3
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(existing_rows)

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=PREDICTIONS_CSV_KEY,
        Body=output.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    print(f"Logged prediction #{len(existing_rows)} → s3://{S3_BUCKET}/{PREDICTIONS_CSV_KEY}")


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global s3_client, kmeans_model, scaler, feature_names, state_map

    s3_client = build_s3_client()
    download_model_artifacts(s3_client)

    kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "state_map.json")) as f:
        state_map = json.load(f)

    with open(os.path.join(MODEL_DIR, "features.json")) as f:
        feature_names = json.load(f)

    print(f"Model loaded. Features: {feature_names}")
    print(f"State map: {state_map}")

    yield

    print("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description="K-Means clustering of industrial machine sensor data. Every prediction is logged to S3.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SensorData(BaseModel):
    air_temperature_k:     float = Field(..., example=298.1,  description="Air temperature in Kelvin")
    process_temperature_k: float = Field(..., example=308.6,  description="Process temperature in Kelvin")
    rotational_speed_rpm:  float = Field(..., example=1551.0, description="Rotational speed in RPM")
    torque_nm:             float = Field(..., example=42.8,   description="Torque in Nm")
    tool_wear_min:         float = Field(..., example=0.0,    description="Tool wear in minutes")


class PredictionResponse(BaseModel):
    cluster_id:     int
    health_state:   str
    input_features: dict
    logged_to_s3:   bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Predictive Maintenance API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": kmeans_model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: SensorData):
    if kmeans_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    input_values = [
        data.air_temperature_k,
        data.process_temperature_k,
        data.rotational_speed_rpm,
        data.torque_nm,
        data.tool_wear_min,
    ]

    X_scaled    = scaler.transform(np.array(input_values).reshape(1, -1))
    cluster_id  = int(kmeans_model.predict(X_scaled)[0])
    health_state = state_map.get(str(cluster_id), "Unknown")

    # Log prediction to S3 CSV
    logged = False
    try:
        log_prediction_to_s3(s3_client, data, cluster_id, health_state)
        logged = True
    except Exception as exc:
        print(f"Warning: could not log prediction to S3: {exc}")

    return PredictionResponse(
        cluster_id=cluster_id,
        health_state=health_state,
        input_features={feature_names[i]: input_values[i] for i in range(len(feature_names))},
        logged_to_s3=logged,
    )


@app.get("/predictions", tags=["Logs"])
def get_predictions():
    """Download and return all logged predictions from S3 as JSON."""
    if s3_client is None:
        raise HTTPException(status_code=503, detail="S3 client not ready.")
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=PREDICTIONS_CSV_KEY)
        content  = response["Body"].read().decode("utf-8")
        reader   = csv.DictReader(io.StringIO(content))
        rows     = list(reader)
        return {"total": len(rows), "predictions": rows}
    except Exception:
        return {"total": 0, "predictions": [], "note": "No predictions logged yet."}
