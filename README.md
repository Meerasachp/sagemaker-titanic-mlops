# 🚀 SageMaker Titanic MLOps

![AWS](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazon-aws&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-blue)
![Status](https://img.shields.io/badge/Status-Phase%206%20Complete-brightgreen)

End-to-end MLOps on **AWS SageMaker** using the Titanic dataset:
**data → preprocessing → training → evaluation → registry → deployment (endpoint) → monitoring → CI/CD**.

---

## 📌 Phases (Roadmap)

### ✅ Phase 1 — Project Initialization
- Local env & repo structure (`src/`, `pipelines/`, `.github/workflows/`)
- `requirements.txt`, `.gitignore`
- AWS access configured (IAM role for SageMaker jobs)

**Quickstart**
```bash
git clone https://github.com/Meerasachp/sagemaker-titanic-mlops.git
cd sagemaker-titanic-mlops
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
aws configure   # use us-east-1

✅ Phase 2 — Data Ingestion (Feature engineering ready)
Titanic CSV stored in S3 under your bucket.

Preprocess script expects label first column; writes train.csv & test.csv.

Quickstart

bash
Copy code
# Create a bucket you own (unique) and upload the CSV
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="sagemaker-us-east-1-${ACCOUNT}-mlops"
aws s3api create-bucket --bucket "$BUCKET"
aws s3 cp ./data/titanic.csv "s3://${BUCKET}/raw/titanic.csv"
aws s3 ls "s3://${BUCKET}/raw/"
ℹ️ Feature Store is optional for v1.0. Preprocess is feature-store compatible if you want to add it later.

✅ Phase 3 — Training & Evaluation (XGBoost)
Training uses SageMaker built-in XGBoost (1.7-1)

Evaluation step produces metrics.json with auc & accuracy.

Core scripts

src/preprocess.py — reads CSV from input dir, writes:

/opt/ml/processing/train/train.csv

/opt/ml/processing/test/test.csv

src/evaluate.py — loads model.tar.gz, computes metrics → /opt/ml/processing/output/metrics.json

✅ Phase 4 — Model Registry & Deployment
Registered models go to Model Package Group (e.g., titanic-xgboost) with PendingManualApproval.

A live endpoint is used by the CI Smoke test.

Deployment scripts are optional; the endpoint can be managed via Studio or your existing infra.
CI smoke step invokes the endpoint using boto3.

✅ Phase 5 — Monitoring
Data capture & Model Monitor schedule (hourly) supported via helper scripts:

src/enable_data_capture.py

src/monitor_setup.py

src/register_model.py (registers the deployed model artifact/image)

Cost guardrails recommended: sampling ≤ 25% and S3 lifecycle (30d) for monitoring/ & datacapture/.

✅ Phase 6 — CI/CD Automation (GitHub Actions + SageMaker Pipelines)
Smoke: invokes endpoint on push/PR

SageMaker Pipeline: Preprocess → Train → Evaluate → Quality Gate (AUC) → Register

Pipeline AUC gate defaults to 0.80 (configurable)

Instance types (safe defaults; override via env):

Processing/Eval: ml.t3.medium

Training (XGBoost allow-list): ml.m4.xlarge

Pipeline file

pipelines/pipeline_up.py — defines & starts the pipeline run

⚙️ CI/CD (GitHub Actions)
Workflow: .github/workflows/mlops.yml has two jobs:

smoke — checks AWS gate, configures creds, invokes endpoint

sagemaker-pipeline — creates/updates TitanicXGBPipeline and starts an execution

Secrets required

Name	Used for	Where
SAGEMAKER_ENDPOINT_NAME	Smoke test invoke	Repo → Settings → Secrets
(either) AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY	Static creds (if not using OIDC)	Repo → Secrets
(or) OIDC role-to-assume	Preferred (no static keys)	IAM Role + workflow permissions
(optional) SAGEMAKER_ROLE_ARN	Pipeline execution role	Repo → Secrets
(optional) INPUT_DATA_URI	Point pipeline to your S3 CSV	Repo → Secrets/Env

Run the pipeline job with env overrides (recommended)

yaml
Copy code
- name: Create/Update & Start SageMaker Pipeline
  env:
    AWS_REGION: us-east-1
    INPUT_DATA_URI: s3://sagemaker-us-east-1-<ACCOUNT>-mlops/raw/titanic.csv
    PROC_INSTANCE_TYPE: ml.t3.medium
    EVAL_INSTANCE_TYPE: ml.t3.medium
    TRAIN_INSTANCE_TYPE: ml.m4.xlarge
  run: python pipelines/pipeline_up.py
Using OIDC instead of static keys (recommended)

yaml
Copy code
permissions:
  id-token: write
  contents: read

- name: Configure AWS credentials (OIDC)
  uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: arn:aws:iam::<ACCOUNT_ID>:role/github-oidc-sagemaker
    aws-region: us-east-1
Create the IAM role with trust to token.actions.githubusercontent.com and attach SageMaker/S3/Logs access to required resources.


🧱 Project Structure
bash
Copy code
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

▶️ How to Run
A) End-to-end via CI

Push to main or click Actions → “MLOps Pipeline (Smoke)” → Run workflow

Watch smoke logs (should print JSON prediction)

Watch sagemaker-pipeline: it upserts & starts TitanicXGBPipeline

In Studio → Pipelines, open the latest execution to see steps & metrics

B) Quick CLI checks

bash
Copy code
# Latest pipeline execution status
aws sagemaker list-pipeline-executions \
  --pipeline-name TitanicXGBPipeline --max-results 1 \
  --query "PipelineExecutionSummaries[0].[PipelineExecutionArn,PipelineExecutionStatus]"

# Latest registered model package
aws sagemaker list-model-packages \
  --model-package-group-name titanic-xgboost \
  --query "ModelPackageSummaryList[0].[ModelPackageArn,ModelApprovalStatus]"
🧪 Example Prediction (from Smoke)
json
Copy code
{"predictions": [{"score": 0.8977}]}
🛡️ Troubleshooting
AccessDenied / No S3 objects under URL

Use a bucket you own (e.g., sagemaker-us-east-1-<ACCOUNT>-mlops)

Upload raw/titanic.csv

Give your SageMaker execution role S3 List/Get/Put on that bucket/prefix

Service quota (SageMaker)

Switch to defaults provided (Processing/Eval: ml.t3.medium, Train: ml.m4.xlarge)

Or request a quota increase in Service Quotas → SageMaker

XGBoost instance type validation

Only certain training types are allowed. To list:

bash
Copy code
aws sagemaker describe-algorithm --algorithm-name xgboost \
  --query 'TrainingSpecification.SupportedTrainingInstanceTypes'
Forked PRs skip AWS steps

By design. Use branches in the same repo or workflow_dispatch.

Quality gate blocks registration

Lower the AUC threshold (env AUC_THRESHOLD, default 0.80) or improve model.

🔒 Security Notes
Prefer OIDC in CI (no long-lived keys).

Scope IAM to only needed actions/prefixes.

Rotate any static access keys if exposed.

👤 Author
Meerasa (Max) — DevOps / MLOps Engineer
🔗 GitHub: https://github.com/Meerasachp







