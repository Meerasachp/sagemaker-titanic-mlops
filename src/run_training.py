import sagemaker
from sagemaker.session import Session
from sagemaker.xgboost import XGBoost
import boto3

# SageMaker session & role
session = sagemaker.Session()
role = "arn:aws:iam::605134434521:role/SageMakerExecutionRole"

# S3 bucket & prefix
bucket = session.default_bucket()  # or hardcode: "sagemaker-us-east-1-605134434521"
prefix = "titanic-xgboost"

# Upload local data to S3
s3 = boto3.Session().resource("s3")
s3.Bucket(bucket).upload_file("data/train.csv", f"{prefix}/data/train.csv")
s3.Bucket(bucket).upload_file("data/test.csv", f"{prefix}/data/test.csv")

# XGBoost Estimator
xgb_estimator = XGBoost(
    entry_point="src/train.py",
    framework_version="1.5-1",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/{prefix}/output",
    sagemaker_session=session,
    hyperparameters={
        "max_depth": 5,
        "eta": 0.2,
        "objective": "binary:logistic",
        "num_round": 100,
    },
)

# Launch training job
xgb_estimator.fit({
    "train": f"s3://{bucket}/{prefix}/data/train.csv",
    "validation": f"s3://{bucket}/{prefix}/data/test.csv"
})

