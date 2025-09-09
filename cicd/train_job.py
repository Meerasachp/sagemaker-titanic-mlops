import os, boto3, sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris

REGION  = os.getenv("AWS_REGION", "us-east-1")
ROLE    = os.environ["SAGEMAKER_ROLE_ARN"]          # secret
BUCKET  = os.environ["S3_BUCKET"]                   # secret (e.g., sagemaker-us-east-1-<acct>)
PREFIX  = os.getenv("PROJECT_PREFIX", "titanic-mlops")
TRAIN_S3 = os.environ["S3_TRAIN_DATA"]              # secret: s3://.../train.csv (features+label, no header)
VAL_S3   = os.getenv("S3_VAL_DATA", TRAIN_S3)

session = sagemaker.Session(boto3.session.Session(region_name=REGION))
xgb_image = image_uris.retrieve(framework='xgboost', region=REGION, version='1.7-1')

est = Estimator(
    image_uri=xgb_image,
    role=ROLE,
    instance_count=1,
    instance_type="ml.m5.large",
    volume_size=10,
    output_path=f"s3://{BUCKET}/{PREFIX}/models/",
    sagemaker_session=session,
    base_job_name=f"{PREFIX}-xgb-train",
)

# Minimal demo hyperparameters
est.set_hyperparameters(
    objective="binary:logistic",
    num_round=100,
    max_depth=5,
    eta=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
)

inputs = {
    "train": TrainingInput(TRAIN_S3, content_type="text/csv"),
    "validation": TrainingInput(VAL_S3, content_type="text/csv"),
}

est.fit(inputs, logs=True, wait=True)
artifact = est.model_data
print(f"MODEL_ARTIFACT_S3={artifact}")
