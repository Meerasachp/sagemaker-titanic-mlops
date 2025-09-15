# 🛟 Titanic Survival — **AWS SageMaker MLOps** Project (Pipelines + GitHub Actions) 

![AWS](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazon-aws&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-2088FF)

End-to-end MLOps on **AWS SageMaker** using the Titanic dataset:  
**data → preprocessing → training → evaluation → registry → deployment (endpoint) → monitoring → CI/CD**.

✅ Data prep to S3  
✅ SageMaker Pipeline (Preprocess → Train → Evaluate → Gate → Register)  
✅ Model Registry (Pending/Manual approval by default)  
✅ Smoke test against a real endpoint  
✅ Monitoring-ready (data capture + Model Monitor helpers)  
✅ OIDC for GitHub Actions (no long-lived keys)  

---

## 📊 Problem Statement
Predict **survival** based on:

- **Pclass**, **Sex**, **Age**, **SibSp**, **Parch**, **Fare**, **Embarked**
- Model: **XGBoost (built-in, 1.7-1)**

---

## 🧱 Repo Structure

```bash
sagemaker-titanic-mlops/
├── data/
│   └── titanic.csv                     # sample local dataset (uploaded to S3)
├── src/
│   ├── preprocess.py                   # writes train/test (label first col, no header)
│   ├── evaluate.py                     # writes metrics.json { auc, accuracy }
│   ├── enable_data_capture.py          # optional: turn on endpoint capture
│   ├── monitor_setup.py                # optional: Model Monitor schedule
│   └── register_model.py               # optional: extra registry helpers
├── pipelines/
│   └── pipeline_up.py                  # defines & triggers SageMaker Pipeline
├── .github/workflows/
│   └── mlops.yml                       # smoke + pipeline jobs (OIDC supported)
├── requirements.txt
├── Makefile                         
└── README.md

▶️ To Start the Project 

git clone https://github.com/Meerasachp/sagemaker-titanic-mlops.git
cd sagemaker-titanic-mlops
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
aws configure   # use us-east-1

🚀 Quick Start

✅Phase 1 — Project Initialization
Local env & repo structure (src/, pipelines/, .github/workflows/)
requirements.txt, .gitignore
AWS access configured (IAM execution role for SageMaker jobs)

✅ Phase 2 — Data Ingestion (feature-engineering ready)
Titanic CSV stored in your S3 bucket. Preprocess expects label in first column; writes train.csv & test.csv.

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="sagemaker-us-east-1-${ACCOUNT}-mlops"
aws s3api create-bucket --bucket "$BUCKET"
aws s3 cp ./data/titanic.csv "s3://${BUCKET}/raw/titanic.csv"
aws s3 ls "s3://${BUCKET}/raw/"

✅ Phase 4 — Model Registry & Deployment
Data capture & Model Monitor (hourly) via helpers:
src/enable_data_capture.py, src/monitor_setup.py, src/register_model.py
Cost guardrails: sampling ≤ 25% + S3 lifecycle (30d).

✅ Phase 6 — CI/CD Automation (GitHub Actions + SageMaker Pipelines)
moke test: invokes endpoint on push/PR.
Pipeline: Preprocess → Train → Evaluate → Gate (AUC ≥ 0.80) → Register.
Instance defaults:
   Processing/Eval → ml.t3.medium
   Training → ml.m4.xlarge

## 🧰 Tech Stack:
AWS SageMaker: Processing, Training, Pipelines, Model Registry, Endpoints
CI/CD: GitHub Actions (smoke + pipeline jobs; OIDC supported)
Storage/Logs: Amazon S3, CloudWatch Logs
Libraries: XGBoost, pandas, scikit-learn, boto3, SageMaker SDK
Security: IAM execution role, OIDC role assumption (no static keys)

▶️ How to Run

A) End-to-end via CI
Push to main or open Actions → “MLOps Pipeline (Smoke)” → Run workflow
Watch smoke logs (prints JSON prediction)
View pipeline execution in SageMaker Studio

B) Quick CLI checks
# Latest pipeline execution status
aws sagemaker list-pipeline-executions \
  --pipeline-name TitanicXGBPipeline --max-results 1 \
  --query "PipelineExecutionSummaries[0].[PipelineExecutionArn,PipelineExecutionStatus]"

# Latest registered model package 
aws sagemaker list-model-packages \
  --model-package-group-name titanic-xgboost \
  --query "ModelPackageSummaryList[0].[ModelPackageArn,ModelApprovalStatus]"

## 🧪 Example Prediction (from Smoke)

{"predictions": [{"score": 0.8977}]}


# 🧩 Known Gotchas / Troubleshooting
ValidationError: Endpoint not found
→ Check region/profile; confirm endpoint is InService; no typos; region var set.

Invalid endpoint URL (double dot)
→ Ensure AWS_REGION is correct.

S3 bucket errors
→ Bucket names must be unique + region-matching.
→ --create-bucket-configuration only needed outside us-east-1.

GitHub Actions credentials
→ Prefer OIDC role. If secrets used, set:
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION


# Meerasa — DevOps / MLOps Engineer :-)

