# 🚀 SageMaker Titanic MLOps

![AWS](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazon-aws&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-blue)
![Status](https://img.shields.io/badge/Status-Phase%203%20Complete-brightgreen)

End-to-end MLOps on **AWS SageMaker** using the Titanic dataset: data → feature store → training → deployment → prediction.

---

## 📌 Phases

### ✅ Phase 1 – Project Initialization
- Local env + venv  
- Repo structure (`src/`, `data/`)  
- `requirements.txt`, `.gitignore`  
- AWS creds + SageMaker access  


**Quickstart (Phase 1)**  
```bash
# Clone the repo
git clone https://github.com/Meerasachp/sagemaker-titanic-mlops.git
cd sagemaker-titanic-mlops

# Create virtual env
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
bash
Copy code
# Configure AWS
aws configure
# Provide IAM user credentials with SageMaker + S3 access
# Verify default region = us-east-1
bash
Copy code
# Prepare project structure
mkdir data src
# Place Titanic dataset (train.csv, test.csv) in data/


✅ Phase 2 – Feature Store
Feature Group created (online + offline)

Ingestion from data/train.csv

IAM + S3 permissions fixed

Verified rows in Studio Feature Store

Quickstart (Phase 2)

bash
Copy code
# Create Feature Group
python src/feature_store_setup.py
bash
Copy code
# Check Feature Group status
python src/check_feature_group.py
Confirms FG is ACTIVE

Lists schema + record count

text
Copy code
AWS Console → SageMaker → Feature Store → Feature groups  
Search for titanic-feature-group-* and explore rows (Name, Sex, Age, Ticket, Cabin, etc.)


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
# Train (XGBoost, script mode)
python src/run_training.py
bash
Copy code
# Deploy endpoint
python src/deploy.py
bash
Copy code
# Predict
python src/predict.py
bash
Copy code
# Delete endpoint (stop costs)
python src/delete_endpoint.py


🔜 Phase 4 – Monitoring & CI/CD
Model Monitor (data/quality/drift)

CI/CD with SageMaker Pipelines & GitHub Actions

Auto retraining triggers
 

📂 Structure
css
Copy code
sagemaker-titanic-mlops/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── feature_store_setup.py
│   ├── check_feature_group.py
│   ├── run_training.py
│   ├── train.py
│   ├── deploy.py
│   ├── predict.py
│   └── delete_endpoint.py
├── requirements.txt
└── README.md


🛠️ Tech
AWS SageMaker (Feature Store, Training, Endpoints)

XGBoost (1.5+)

Python / Pandas / scikit-learn

S3, IAM, CloudWatch



👤 Author
Meerasa (Max) — DevOps/MLOps
🔗 GitHub
