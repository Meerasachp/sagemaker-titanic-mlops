# 🚀 SageMaker Titanic MLOps

![AWS](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazon-aws&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-blue)
![Status](https://img.shields.io/badge/Status-Phase%204%20Complete-brightgreen)

End-to-end MLOps on **AWS SageMaker** using the Titanic dataset: data → feature store → training → deployment → prediction → monitoring → CI/CD.

---

## 📌 Phases

### ✅ Phase 1 – Project Initialization
- Local env + venv  
- Repo structure (`src/`, `data/`)  
- `requirements.txt`, `.gitignore`  
- AWS creds + SageMaker access  

#### Quickstart (Phase 1)
```bash
git clone https://github.com/Meerasachp/sagemaker-titanic-mlops.git
cd sagemaker-titanic-mlops

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

aws configure   # provide IAM creds
# verify region = us-east-1

mkdir data src

# place Titanic dataset (train.csv, test.csv) in data/
✅ Phase 2 – Feature Store
Feature Group created (online + offline)

Ingestion from data/train.csv

IAM + S3 permissions fixed

Verified rows in Studio Feature Store

Quickstart (Phase 2)
bash
Copy code
python src/feature_store_setup.py
python src/check_feature_group.py   # confirm ACTIVE + schema + row count
AWS Console → SageMaker → Feature Store → Feature groups → titanic-feature-group-*

✅ Phase 3 – Training & Deployment (XGBoost)
Training: Built with SageMaker XGBoost (script mode)

Artifacts: Stored in S3 (model.tar.gz)

Deployment: Real-time endpoint on SageMaker

Prediction: Live inference working

Example result:

json
Copy code
{"predictions": [{"score": 0.8977}]}
Quickstart (Phase 3)
bash
Copy code
python src/run_training.py     # train XGBoost
python src/deploy.py           # deploy endpoint
python src/predict.py          # test prediction
python src/delete_endpoint.py  # cleanup


✅ Phase 4 – Monitoring & CI/CD
Serverless endpoint (cost-efficient, $0 idle)

Batch inference with batch_invoke.py

Optional Model Monitor setup (baseline + drift detection)

CI/CD via GitHub Actions:

Smoke test on push/PR → invokes endpoint

Manual Train job (launch SageMaker training)

Manual Canary Deploy (10% traffic to new model)

Quickstart (Phase 4)
Batch predictions

bash
Copy code
python src/batch_invoke.py --input data/sample_rows.csv --output preds.jsonl
head preds.jsonl
Serverless switch (script)

bash
Copy code
./serverless_recreate.sh \
  --endpoint titanic-xgboost-endpoint-1757260241 \
  --model sagemaker-xgboost-2025-09-07-15-50-46-469 \
  --region us-east-1 --mem 2048 --maxc 5
GitHub Actions (MLOps Pipeline)

smoke-test → auto on push/PR

train → manual run, launches SageMaker training (cicd/train_job.py)

canary-deploy → manual run, 10% traffic shift (cicd/deploy_canary.py)

📂 Structure
plaintext
Copy code
sagemaker-titanic-mlops/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_rows.csv
│   └── baseline_features.csv
├── src/
│   ├── feature_store_setup.py
│   ├── check_feature_group.py
│   ├── run_training.py
│   ├── train.py
│   ├── deploy.py
│   ├── predict.py
│   ├── delete_endpoint.py
│   ├── invoke_test.py
│   ├── batch_invoke.py
│   └── monitor_setup.py
├── cicd/
│   ├── train_job.py
│   └── deploy_canary.py
├── .github/workflows/mlops.yml
├── serverless_recreate.sh
└── README.md

🛠️ Tech
AWS SageMaker (Feature Store, Training, Endpoints, Serverless Inference)

XGBoost (1.5+ → 1.7-1)

Python / Pandas / scikit-learn

GitHub Actions (CI/CD)

S3, IAM, CloudWatch

👤 Author
Meerasa (Max) — DevOps / MLOps Engineer
🔗 GitHub
