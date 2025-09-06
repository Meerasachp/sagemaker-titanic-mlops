import sagemaker
from sagemaker import Session
from sagemaker.xgboost.model import XGBoostModel
import pandas as pd

# 🔹 Initialize SageMaker session
session = Session()
role = "arn:aws:iam::605134434521:role/SageMakerExecutionRole" 
region = session.boto_region_name

# 🔹 Get the latest completed training job
sm_client = session.sagemaker_client
response = sm_client.list_training_jobs(SortBy="CreationTime", SortOrder="Descending", MaxResults=1)
latest_job_name = response["TrainingJobSummaries"][0]["TrainingJobName"]

print(f"Latest training job: {latest_job_name}")

# 🔹 Get model artifact S3 path from training job
desc = sm_client.describe_training_job(TrainingJobName=latest_job_name)
model_artifact = desc["ModelArtifacts"]["S3ModelArtifacts"]
print(f"Model artifact: {model_artifact}")

# 🔹 Create SageMaker Model object
xgb_model = XGBoostModel(
    model_data=model_artifact,
    role=role,
    framework_version="1.5-1",
    sagemaker_session=session
)

# 🔹 Deploy model as endpoint
predictor = xgb_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="titanic-xgboost-endpoint",
    serializer=sagemaker.serializers.CSVSerializer(),   # 👈 ensures CSV format
    deserializer=sagemaker.deserializers.JSONDeserializer()
)

print("✅ Model deployed at endpoint: titanic-xgboost-endpoint")

# 🔹 Test inference with one passenger
test_passenger = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 1,      # already encoded
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 0
}])

# Convert to CSV-friendly row
prediction = predictor.predict(test_passenger.to_csv(header=False, index=False))
print("🔮 Prediction result:", prediction)

