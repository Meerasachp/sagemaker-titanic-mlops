# 🛟 Titanic Survival — Your First **AWS SageMaker MLOps** Project (Pipelines + GitHub Actions) 

![AWS](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazon-aws&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-2088FF)

End-to-end MLOps on **AWS SageMaker** using the Titanic dataset:  
**data → preprocessing → training → evaluation → registry → deployment (endpoint) → monitoring → CI/CD**.

✅ Data prep to S3  
✅ SageMaker **Pipeline** (Preprocess → Train → Evaluate → Gate → Register)  
✅ Model **Registry** (Pending approval)  
✅ **Smoke** test against a real endpoint  
✅ Monitoring-ready (data capture + Model Monitor helpers)

---

## 📊 Problem Statement
Predict **survival** based on:

- **Pclass**, **Sex**, **Age**, **SibSp**, **Parch**, **Fare**, **Embarked**
- Model: **XGBoost (built-in, 1.7-1)**

---

## 🚀 Quick Start

### ✅ Phase 1 — Project Initialization
- Local env & repo structure (`src/`, `pipelines/`, `.github/workflows/`)
- `requirements.txt`, `.gitignore`
- AWS access configured (IAM execution role for SageMaker jobs)

**Quickstart**

git clone https://github.com/Meerasachp/sagemaker-titanic-mlops.git
cd sagemaker-titanic-mlops
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
aws configure   # use us-east-1

### ✅ Phase 2 — Data Ingestion (feature-engineering ready)

Titanic CSV stored in your S3 bucket.
Preprocess expects label in first column; writes train.csv & test.csv.

# Create a bucket you own (unique) and upload the CSV
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="sagemaker-us-east-1-${ACCOUNT}-mlops"
aws s3api create-bucket --bucket "$BUCKET"
aws s3 cp ./data/titanic.csv "s3://${BUCKET}/raw/titanic.csv"
aws s3 ls "s3://${BUCKET}/raw/"
ℹ️ Feature Store is optional for v1.0; preprocess is feature-store compatible if you add it later.

### ✅ Phase 3 — Training & Evaluation (XGBoost)

- Training uses SageMaker built-in XGBoost (1.7-1).
- Evaluation produces metrics.json with auc & accuracy.

core scripts
src/preprocess.py → reads CSV from input dir; writes:
/opt/ml/processing/train/train.csv
/opt/ml/processing/test/test.csv
src/evaluate.py → loads model.tar.gz, computes metrics → /opt/ml/processing/output/metrics.json

### ✅ Phase 4 — Model Registry & Deployment

- Registered models land in Model Package Group (e.g., titanic-xgboost) with PendingManualApproval.
- A live endpoint is used by the CI Smoke test (boto3 invoke).

### ✅ Phase 5 — Monitoring

- Data capture & Model Monitor schedule (hourly) supported via helpers:
  src/enable_data_capture.py
  src/monitor_setup.py
  src/register_model.py
 Cost guardrails: sampling ≤ 25% and S3 lifecycle (30d) for monitoring/ & datacapture/.

### ✅ Phase 6 — CI/CD Automation (GitHub Actions + SageMaker Pipelines)

Smoke: invokes endpoint on push/PR.
SageMaker Pipeline: Preprocess → Train → Evaluate → Quality Gate (AUC) → Register.
Gate defaults to AUC ≥ 0.80 (override via env).
Instance types (safe defaults, override via env):
   Processing/Eval: ml.t3.medium
   Training (XGBoost allow-list): ml.m4.xlarge
Pipeline file: pipelines/pipeline_up.py (defines & starts the run).

### 🧰 Tech Stack

AWS SageMaker: Processing, Training (XGBoost), Pipelines, Model Registry, Endpoints
CI/CD: GitHub Actions (smoke + pipeline jobs; OIDC supported)
Storage & Logs: Amazon S3 (artifacts, capture, metrics), CloudWatch Logs
Libraries: XGBoost, pandas, scikit-learn, boto3, SageMaker SDK
Security: IAM execution role, OIDC role assumption for CI (no static keys), fork-PR secret gating

## 🧱 Project Structure
sagemaker-titanic-mlops/
├── data/
│   └── titanic.csv                  # example local dataset (uploaded to S3)
├── src/
│   ├── preprocess.py                # writes train/test CSVs (headerless, label first col)
│   ├── evaluate.py                  # writes metrics.json {auc, accuracy}
│   ├── enable_data_capture.py       # optional monitoring helper
│   ├── monitor_setup.py             # optional monitoring helper
│   └── register_model.py            # optional registry helper
├── pipelines/
│   └── pipeline_up.py               # SageMaker Pipeline definition + start
├── .github/workflows/
│   └── mlops.yml                    # Smoke + Pipeline CI
├── requirements.txt
└── README.md

### ▶️ How to Run

A) End-to-end via CI

Push to main or open Actions → “MLOps Pipeline (Smoke)” → Run workflow
Watch smoke logs (prints JSON prediction)
Watch sagemaker-pipeline logs (upserts & starts TitanicXGBPipeline)
In Studio → Pipelines, open latest execution to see steps & metrics

B) Quick CLI checks

# Latest pipeline execution status
aws sagemaker list-pipeline-executions \
  --pipeline-name TitanicXGBPipeline --max-results 1 \
  --query "PipelineExecutionSummaries[0].[PipelineExecutionArn,PipelineExecutionStatus]"

# Latest registered model package
aws sagemaker list-model-packages \
  --model-package-group-name titanic-xgboost \
  --query "ModelPackageSummaryList[0].[ModelPackageArn,ModelApprovalStatus]"

  ### 🧪 Example Prediction (from Smoke)

  {"predictions": [{"score": 0.8977}]}

Meerasa — DevOps / MLOps Engineer :-)





