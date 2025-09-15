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
