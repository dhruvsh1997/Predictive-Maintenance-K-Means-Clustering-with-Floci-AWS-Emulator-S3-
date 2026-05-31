# Predictive Maintenance — K-Means Clustering with Floci (AWS Emulator)

A complete ML pipeline that clusters industrial machine sensor data into health states — **Optimal**, **Degrading**, **Critical Failure Risk** — and serves predictions via a FastAPI REST API.

**Every prediction is automatically logged** (input + output) to a CSV file in S3 that grows with each call. Model artifacts are stored in S3 and the API runs locally, with S3 emulated using **floci**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Tech Stack](#tech-stacks)
4. [What is Floci?](#what-is-floci)
5. [Setup — Prerequisites](#setup--prerequisites)
6. [Step 1 — Start Floci](#step-1--start-floci)
7. [Step 2 — Configure AWS Credentials](#step-2--configure-aws-credentials)
8. [Step 3 — Install Python Dependencies](#step-3--install-python-dependencies)
9. [Step 4 — Train the Model (Jupyter Notebook)](#step-4--train-the-model-jupyter-notebook)
10. [Step 5 — Verify S3 Upload](#step-5--verify-s3-upload)
11. [Step 6 — Run the API](#step-6--run-the-api)
12. [Step 7 — Test the API](#step-7--test-the-api)
13. [Step 8 — Inspect the Predictions CSV in S3](#step-8--inspect-the-predictions-csv-in-s3)
14. [API Reference](#api-reference)
15. [Floci Command Reference](#floci-command-reference)
16. [Migrating to Real AWS (S3)](#migrating-to-real-aws-s3)
17. [Troubleshooting](#troubleshooting)

---

> **Reading this guide:** Every section that involves a terminal command shows two variants — **Windows (PowerShell)** and **Linux / macOS (bash)**. Run the one that matches your OS. Commands identical on both OSes are shown once without a label.

---

## Project Overview

**Dataset:** AI4I 2020 Predictive Maintenance Dataset (10,000 rows, 14 columns)  
**Algorithm:** K-Means Clustering (k=3, unsupervised)  
**Features used for clustering:**
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

**Output clusters:**
| Health State | Description |
|-------------|-------------|
| Optimal | Machine running normally — low wear, stable readings |
| Degrading | Schedule maintenance soon — wear accumulating |
| Critical Failure Risk | Immediate attention required — high wear / abnormal readings |

**S3 layout (all inside floci at `http://localhost:4566`):**
```
s3://predictive-maintenance/
├── model/
│   ├── kmeans_model.pkl      ← trained model
│   ├── scaler.pkl            ← StandardScaler (must match training)
│   ├── state_map.json        ← cluster ID → health state label
│   └── features.json         ← feature column order
└── logs/
    └── predictions.csv       ← appended on every /predict call
```

---

## Project Structure

```
predictive-maintenance/
├── Dataset/
│   └── ai4i2020.csv              # Source data
├── notebook/
│   └── train_model.ipynb         # Full training pipeline
├── model/                        # Auto-created during training
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── state_map.json
│   ├── features.json
│   └── *.png                     # EDA / cluster plots
├── api/
│   └── main.py                   # FastAPI application
├── requirements.txt              # All dependencies — training + API
└── README.md
```

Dataset - https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Training | scikit-learn (KMeans, StandardScaler), pandas, numpy |
| Model Persistence | joblib |
| API Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Cloud Storage | AWS S3 (via floci) |
| AWS SDK | boto3 |
| AWS CLI | aws-cli v2 |
| Local AWS Emulator | floci |

---

## What is Floci?

Floci is a **local AWS cloud emulator** — it runs 47 AWS services (S3, Lambda, RDS, etc.) on your machine on port `4566`. No AWS account, no billing. You use the same `aws` CLI commands and `boto3` SDK calls as real AWS; you just point them at `http://localhost:4566` instead.

> Think of it as "AWS on your laptop."

---

## Setup — Prerequisites

Before starting, make sure you have:

- [ ] Python 3.10+
- [ ] `aws` CLI v2 installed — verify: `aws --version`
- [ ] `floci` installed (see below)
- [ ] `jupyter` or `jupyterlab` available

### Install floci

**Windows (PowerShell):**
```powershell
iwr https://floci.io/install.ps1 | iex
```

**Linux / macOS (bash):**
```bash
curl -fsSL https://floci.io/install.sh | sh
```

### Verify floci installed (both OS)
```bash
floci --version
```

### Install AWS CLI v2

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://awscli.amazonaws.com/AWSCLIV2.msi" -OutFile "$env:TEMP\AWSCLIV2.msi"
Start-Process msiexec.exe -Wait -ArgumentList "/I $env:TEMP\AWSCLIV2.msi /quiet"
```
Then **close and reopen PowerShell** so PATH updates.

**Linux (bash):**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**macOS (bash):**
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```

Verify: `aws --version`

---

## Step 1 — Start Floci

Open a **dedicated terminal** and run (same on both OS):

```bash
floci start
```

Expected output:
```
✓ Floci started in 24ms
✓ Endpoint: http://localhost:4566
✓ Services: S3, Lambda, ...
```

**Keep this terminal open** — floci must stay running for the entire project session.

Other useful commands (same on both OS):
```bash
floci status          # check if running
floci logs --follow   # stream live service logs
floci doctor          # run diagnostics
floci stop            # stop the emulator when done
```

---

## Step 2 — Configure AWS Credentials

Floci accepts any dummy credentials. Open a **second terminal** (your project terminal):

**Windows (PowerShell):**
```powershell
$env:AWS_ENDPOINT_URL      = "http://localhost:4566"
$env:AWS_ACCESS_KEY_ID     = "test"
$env:AWS_SECRET_ACCESS_KEY = "test"
$env:AWS_DEFAULT_REGION    = "us-east-1"
```

**Linux / macOS (bash):**
```bash
eval $(floci env)
```

> `eval $(floci env)` auto-exports all four env vars. On Windows, `floci env` only prints them — set them manually as shown above.

**Verify connection (both OS):**
```bash
aws --endpoint-url http://localhost:4566 s3 ls
```
Empty output = connected. Error = floci not running.

> **Important:** Env vars set in one terminal are not visible in other terminals. Repeat the env var block in every new terminal you open.

---

## Step 3 — Install Python Dependencies

**Windows (PowerShell):**
```powershell
cd predictive-maintenance
pip install -r requirements.txt
```

**Linux / macOS (bash):**
```bash
cd predictive-maintenance
pip install -r requirements.txt
```

---

## Step 4 — Train the Model (Jupyter Notebook)

The env vars from Step 2 must be active in the **same terminal** you use to launch Jupyter, so boto3 inside the notebook can reach floci S3.

**Windows (PowerShell):**
```powershell
$env:AWS_ENDPOINT_URL      = "http://localhost:4566"
$env:AWS_ACCESS_KEY_ID     = "test"
$env:AWS_SECRET_ACCESS_KEY = "test"
$env:AWS_DEFAULT_REGION    = "us-east-1"

jupyter notebook notebook/train_model.ipynb
```

**Linux / macOS (bash):**
```bash
eval $(floci env)
jupyter notebook notebook/train_model.ipynb
```

**Run all cells top to bottom.** The notebook does:

1. Loads `Dataset/ai4i2020.csv`
2. Plots feature distributions and correlation heatmap
3. Scales features with `StandardScaler` (critical for K-Means)
4. Runs elbow method + silhouette analysis (k=2..8) — confirms k=3
5. Trains `KMeans(n_clusters=3, random_state=42)`
6. Profiles clusters by tool wear → assigns Optimal / Degrading / Critical Failure Risk
7. Saves 4 artifacts locally to `model/`: `kmeans_model.pkl`, `scaler.pkl`, `state_map.json`, `features.json`
8. Connects to floci S3 via boto3
9. Creates bucket `predictive-maintenance`
10. Uploads all 4 artifacts to `s3://predictive-maintenance/model/`

Expected output from upload cells:
```
Using endpoint: http://localhost:4566
Bucket created: s3://predictive-maintenance
Uploaded: ../model/kmeans_model.pkl → s3://predictive-maintenance/model/kmeans_model.pkl
Uploaded: ../model/scaler.pkl → s3://predictive-maintenance/model/scaler.pkl
Uploaded: ../model/state_map.json → s3://predictive-maintenance/model/state_map.json
Uploaded: ../model/features.json → s3://predictive-maintenance/model/features.json
```

---

## Step 5 — Verify S3 Upload

AWS CLI S3 commands are **identical on Windows and Linux**.

Confirm model files are in floci S3:
```bash
aws --endpoint-url http://localhost:4566 s3 ls s3://predictive-maintenance/model/
```

List all buckets:
```bash
aws --endpoint-url http://localhost:4566 s3 ls
```

Print a file to terminal to verify content:
```bash
aws --endpoint-url http://localhost:4566 s3 cp s3://predictive-maintenance/model/state_map.json -
```

---

## Step 6 — Run the API

The API downloads model artifacts from floci S3 on startup, then serves predictions. Every `/predict` call appends a row to `s3://predictive-maintenance/logs/predictions.csv`.

Set env vars and start the server from the `predictive-maintenance/` folder:

**Windows (PowerShell):**
```powershell
$env:AWS_ENDPOINT_URL      = "http://localhost:4566"
$env:AWS_ACCESS_KEY_ID     = "test"
$env:AWS_SECRET_ACCESS_KEY = "test"
$env:AWS_DEFAULT_REGION    = "us-east-1"
$env:S3_BUCKET             = "predictive-maintenance"
$env:S3_MODEL_PREFIX       = "model"
$env:MODEL_DIR             = "$PWD\model_cache"

cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Linux / macOS (bash):**
```bash
eval $(floci env)
export S3_BUCKET=predictive-maintenance
export S3_MODEL_PREFIX=model
export MODEL_DIR=./model_cache

cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Expected startup output:
```
Downloading s3://predictive-maintenance/model/kmeans_model.pkl → ./model_cache/kmeans_model.pkl
Downloading s3://predictive-maintenance/model/scaler.pkl → ./model_cache/scaler.pkl
Downloading s3://predictive-maintenance/model/state_map.json → ./model_cache/state_map.json
Downloading s3://predictive-maintenance/model/features.json → ./model_cache/features.json
All model artifacts downloaded.
Model loaded. Features: [...]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 7 — Test the API

### Check API is alive (both OS)
```bash
curl http://localhost:8000/health
```
Expected: `{"status":"ok","model_loaded":true}`

---

### Predict — Optimal machine

**Windows (PowerShell) — using `Invoke-RestMethod`:**
```powershell
$body = @{
    air_temperature_k     = 298.1
    process_temperature_k = 308.6
    rotational_speed_rpm  = 1551.0
    torque_nm             = 42.8
    tool_wear_min         = 0.0
} | ConvertTo-Json

Invoke-RestMethod -Method POST `
    -Uri "http://localhost:8000/predict" `
    -ContentType "application/json" `
    -Body $body
```

**Windows (PowerShell) — using `curl.exe` (note the `.exe`, not the PowerShell alias):**
```powershell
curl.exe -X POST http://localhost:8000/predict `
    -H "Content-Type: application/json" `
    -d "{\"air_temperature_k\":298.1,\"process_temperature_k\":308.6,\"rotational_speed_rpm\":1551.0,\"torque_nm\":42.8,\"tool_wear_min\":0.0}"
```

**Linux / macOS (bash):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551.0,
    "torque_nm": 42.8,
    "tool_wear_min": 0.0
  }'
```

Expected response:
```json
{
  "cluster_id": 0,
  "health_state": "Optimal",
  "input_features": {
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551.0,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0.0
  },
  "logged_to_s3": true
}
```

`"logged_to_s3": true` confirms the prediction was written to `s3://predictive-maintenance/logs/predictions.csv`.

---

### Predict — Degrading machine

**Windows (PowerShell):**
```powershell
$body = @{
    air_temperature_k     = 302.5
    process_temperature_k = 312.0
    rotational_speed_rpm  = 1350.0
    torque_nm             = 55.0
    tool_wear_min         = 140.0
} | ConvertTo-Json

Invoke-RestMethod -Method POST `
    -Uri "http://localhost:8000/predict" `
    -ContentType "application/json" `
    -Body $body
```

**Linux / macOS (bash):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature_k": 302.5,
    "process_temperature_k": 312.0,
    "rotational_speed_rpm": 1350.0,
    "torque_nm": 55.0,
    "tool_wear_min": 140.0
  }'
```

---

### Predict — Critical machine

**Windows (PowerShell):**
```powershell
$body = @{
    air_temperature_k     = 305.0
    process_temperature_k = 315.0
    rotational_speed_rpm  = 1200.0
    torque_nm             = 68.0
    tool_wear_min         = 230.0
} | ConvertTo-Json

Invoke-RestMethod -Method POST `
    -Uri "http://localhost:8000/predict" `
    -ContentType "application/json" `
    -Body $body
```

**Linux / macOS (bash):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature_k": 305.0,
    "process_temperature_k": 315.0,
    "rotational_speed_rpm": 1200.0,
    "torque_nm": 68.0,
    "tool_wear_min": 230.0
  }'
```

---

### Open interactive Swagger UI (both OS)

Navigate to: **http://localhost:8000/docs**

FastAPI auto-generates Swagger UI — send test requests directly from the browser without any terminal commands.

---

## Step 8 — Inspect the Predictions CSV in S3

Every `/predict` call appends a row to `s3://predictive-maintenance/logs/predictions.csv`.

| Column | Description |
|--------|-------------|
| `timestamp` | UTC ISO-8601 timestamp |
| `air_temperature_k` | Input |
| `process_temperature_k` | Input |
| `rotational_speed_rpm` | Input |
| `torque_nm` | Input |
| `tool_wear_min` | Input |
| `cluster_id` | Raw cluster number (0, 1, or 2) |
| `health_state` | Optimal / Degrading / Critical Failure Risk |

### Option A — via API endpoint (both OS)
```bash
curl http://localhost:8000/predictions
```

### Option B — download CSV via AWS CLI (both OS)
```bash
aws --endpoint-url http://localhost:4566 s3 cp s3://predictive-maintenance/logs/predictions.csv predictions.csv
```
Then open `predictions.csv` in Excel (Windows) or any CSV viewer.

### Option C — print CSV to terminal (both OS)
```bash
aws --endpoint-url http://localhost:4566 s3 cp s3://predictive-maintenance/logs/predictions.csv -
```

---

## API Reference

### `GET /`
Health ping. Returns `{"status": "ok", "message": "..."}`.

### `GET /health`
Returns `{"status": "ok", "model_loaded": true/false}`.

### `POST /predict`

**Request body:**
```json
{
  "air_temperature_k":     298.1,
  "process_temperature_k": 308.6,
  "rotational_speed_rpm":  1551.0,
  "torque_nm":             42.8,
  "tool_wear_min":         0.0
}
```

**Response body:**
```json
{
  "cluster_id":     0,
  "health_state":   "Optimal",
  "input_features": { "Air temperature [K]": 298.1, "...": "..." },
  "logged_to_s3":   true
}
```

`logged_to_s3` is `true` if prediction was appended to the S3 CSV, `false` if the S3 write failed (prediction still returned either way).

### `GET /predictions`
Returns all logged predictions from `s3://predictive-maintenance/logs/predictions.csv` as JSON.
Returns `{"total": 0, "predictions": []}` if nothing logged yet.

---

## Floci Command Reference

All floci commands are identical on Windows and Linux:

| Command | Purpose |
|---------|---------|
| `floci start` | Start the AWS emulator |
| `floci start --persist ./data` | Start with persistent storage (S3 data survives restarts) |
| `floci status` | Check if emulator is running |
| `floci logs --follow` | Stream live service logs |
| `floci stop` | Stop the emulator |
| `floci env` | Print AWS env vars to export |
| `floci doctor` | Run diagnostics |
| `floci snapshot save <name>` | Save current S3 state |
| `floci snapshot restore <name>` | Restore a saved snapshot |

**S3 via AWS CLI — identical on Windows and Linux (always include `--endpoint-url http://localhost:4566`):**

| Command | Purpose |
|---------|---------|
| `aws ... s3 ls` | List all buckets |
| `aws ... s3 mb s3://bucket` | Create bucket |
| `aws ... s3 cp file.txt s3://bucket/key` | Upload file |
| `aws ... s3 cp s3://bucket/key file.txt` | Download file |
| `aws ... s3 cp s3://bucket/key -` | Print file to stdout |
| `aws ... s3 ls s3://bucket/prefix/` | List objects under prefix |
| `aws ... s3 rm s3://bucket/key` | Delete object |
| `aws ... s3 rb s3://bucket --force` | Delete bucket and all contents |

---

## Migrating to Real AWS (S3)

When you're ready to use real AWS S3 instead of floci, here is every change required.

---

### 1. Create a Real AWS Account & IAM User

1. Sign up at https://aws.amazon.com
2. Go to **IAM → Users → Create user**
3. Attach the managed policy: `AmazonS3FullAccess`
4. Under the user → **Security credentials → Create access key** → choose "CLI" → download the CSV

---

### 2. Configure Real AWS Credentials

Run on both OS (same command):
```bash
aws configure
```
Enter:
```
AWS Access Key ID:     <your real key id>
AWS Secret Access Key: <your real secret key>
Default region name:   us-east-1
Default output format: json
```

Remove the floci env vars:

**Windows (PowerShell):**
```powershell
Remove-Item Env:AWS_ENDPOINT_URL        # Critical — this redirects traffic to floci
Remove-Item Env:AWS_ACCESS_KEY_ID
Remove-Item Env:AWS_SECRET_ACCESS_KEY
Remove-Item Env:AWS_DEFAULT_REGION
```

**Linux / macOS (bash):**
```bash
unset AWS_ENDPOINT_URL        # Critical — this redirects traffic to floci
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_DEFAULT_REGION
```

> `AWS_ENDPOINT_URL` is the **only** thing that routes boto3 and aws-cli to floci. Delete it and all traffic goes to real AWS automatically.

---

### 3. Changes in the Jupyter Notebook

In Cell 8 (S3 config), change:

```python
# BEFORE (floci)
s3 = boto3.client(
    's3',
    endpoint_url=FLOCI_ENDPOINT,      # remove this line
    aws_access_key_id=AWS_ACCESS_KEY, # remove this line
    aws_secret_access_key=AWS_SECRET, # remove this line
    region_name=AWS_REGION
)

# AFTER (real AWS) — reads credentials from ~/.aws/credentials automatically
s3 = boto3.client('s3', region_name='us-east-1')
```

S3 bucket names must be **globally unique** on real AWS:
```python
S3_BUCKET = 'predictive-maintenance-yourname-2024'  # must be unique across all AWS
```

S3 CLI commands — just drop `--endpoint-url`:

**Windows (PowerShell):**
```powershell
# floci
aws --endpoint-url http://localhost:4566 s3 ls s3://predictive-maintenance/model/

# real AWS
aws s3 ls s3://predictive-maintenance-yourname-2024/model/
```

**Linux / macOS (bash):**
```bash
# floci
aws --endpoint-url http://localhost:4566 s3 ls s3://predictive-maintenance/model/

# real AWS
aws s3 ls s3://predictive-maintenance-yourname-2024/model/
```

---

### 4. Changes in `api/main.py`

Remove `endpoint_url` and hardcoded credentials from `build_s3_client()`:

```python
# BEFORE (floci)
def build_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,         # remove
        aws_access_key_id=AWS_KEY_ID,     # remove
        aws_secret_access_key=AWS_SECRET, # remove
        region_name=AWS_REGION,
    )

# AFTER (real AWS)
def build_s3_client():
    return boto3.client("s3", region_name="us-east-1")
```

Update the `S3_BUCKET` default:
```python
S3_BUCKET = os.environ.get("S3_BUCKET", "predictive-maintenance-yourname-2024")
```

---

### 5. Changes in env vars when running the API

**Windows (PowerShell):**
```powershell
# Remove AWS_ENDPOINT_URL — do NOT set it for real AWS
Remove-Item Env:AWS_ENDPOINT_URL

$env:S3_BUCKET       = "predictive-maintenance-yourname-2024"
$env:S3_MODEL_PREFIX = "model"
$env:MODEL_DIR       = "$PWD\model_cache"

cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Linux / macOS (bash):**
```bash
# Do not set AWS_ENDPOINT_URL for real AWS
export S3_BUCKET=predictive-maintenance-yourname-2024
export S3_MODEL_PREFIX=model
export MODEL_DIR=./model_cache

cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

### 6. Summary — What Actually Changes

| Thing | Floci | Real AWS |
|-------|-------|----------|
| `AWS_ENDPOINT_URL` env var | `http://localhost:4566` | **Delete it entirely** |
| AWS credentials | `test` / `test` | Real IAM access key + secret |
| `boto3.client()` | Pass `endpoint_url=` | Remove `endpoint_url=` |
| `aws` CLI | Append `--endpoint-url http://localhost:4566` | Remove that flag |
| Bucket name uniqueness | Local — any name works | Must be globally unique across all AWS |
| Cost | Free | S3 ~$0.023/GB/month |

---

## Troubleshooting

**`aws` is not recognized:**  
AWS CLI not installed or terminal not restarted after install. Install via the Prerequisites section, then **close and reopen the terminal**.

**`Connection refused` on any `aws` command:**  
Floci not running. Run `floci start` and wait for the ready message.

**`Could not connect to the endpoint URL`:**  
`AWS_ENDPOINT_URL` not set in current terminal. Re-run the env var block from Step 2.

**Notebook cell 8 fails with `NoCredentialsError` or connection error:**  
Env vars set in a different terminal. Set them in the same terminal used to launch Jupyter.

**`BucketAlreadyOwnedByYou` in notebook:**  
Normal — bucket exists from a previous run. The notebook catches this and continues.

**API starts but `model_loaded: false`:**  
Notebook S3 upload did not complete. Re-run cells 8–10 in the notebook, then restart the API.

**`logged_to_s3: false` in predict response:**  
API cannot reach floci S3. Confirm `AWS_ENDPOINT_URL=http://localhost:4566` is set in the terminal running the API.

**`curl` in PowerShell sends wrong JSON:**  
PowerShell's `curl` is an alias for `Invoke-WebRequest` — it does not behave like Linux curl. Use `Invoke-RestMethod` (shown in Step 7) or call `curl.exe` explicitly (note the `.exe`).

**Data lost after `floci stop`:**  
Floci state is ephemeral by default. Use `floci start --persist ./floci-data` to persist S3 data across restarts.
