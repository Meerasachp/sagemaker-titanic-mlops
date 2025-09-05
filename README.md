# 🚀 SageMaker Titanic MLOps

![AWS](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazon-aws&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-blue)
![Status](https://img.shields.io/badge/Status-Phase%202%20Complete-brightgreen)

This repository demonstrates an **end-to-end MLOps pipeline** on **AWS SageMaker** using the Titanic dataset.  
The goal is to build a reproducible, automated pipeline that handles data ingestion, feature engineering, model training, deployment, and monitoring.

---

## 📌 Project Phases

### ✅ Phase 1 – Project Initialization  
- Setup local environment and GitHub repo  
- Added requirements and project structure  
- Verified AWS CLI & SageMaker Studio setup  

### ✅ Phase 2 – Feature Store Setup  
- Created SageMaker Feature Store  
- Defined feature groups (passenger details, survival labels)  
- Verified feature ingestion and retrieval in Studio  

### 🔄 Phase 3 – Model Training & Deployment (In Progress)  
- Build training pipeline with SageMaker Processing & Training Jobs  
- Register model artifacts in Model Registry  
- Deploy model to SageMaker Endpoint  

### 🔜 Phase 4 – Monitoring & CI/CD  
- Setup model monitoring for drift and bias  
- Integrate CI/CD pipelines (GitHub Actions + SageMaker Pipelines)  
- Enable automated retraining triggers  

---

## 📂 Repository Structure

sagemaker-titanic-mlops/
├── src/ # Source code
│ ├── data_ingestion/ # Data preprocessing scripts
│ ├── feature_store/ # Feature store setup
│ ├── training/ # Model training scripts
│ └── deployment/ # Model deployment configs
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## 🛠️ Tech Stack
- **AWS SageMaker** – ML training, deployment, and feature store  
- **Python** – Core development language  
- **Docker** – Containerization  
- **GitHub Actions** – CI/CD pipelines  
- **Terraform (Planned)** – Infrastructure as Code  

---

## 📈 Roadmap
- [x] Phase 1 – Initialization  
- [x] Phase 2 – Feature Store Setup  
- [ ] Phase 3 – Model Training & Deployment  
- [ ] Phase 4 – Monitoring & CI/CD  

---

## 👨‍💻 Author
**Meerasa** – DevOps/MLOps Engineer  
🔗 [GitHub Profile](https://github.com/Meerasachp)  


