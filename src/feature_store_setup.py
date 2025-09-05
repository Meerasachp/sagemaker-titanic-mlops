import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
import pandas as pd
from time import strftime, gmtime
import time

# -----------------------------
# Step 1: Session & Role
# -----------------------------
session = sagemaker.Session()
region = session.boto_region_name
print("SageMaker session region:", region)

# Replace with your IAM role ARN
role = "arn:aws:iam::605134434521:role/SageMakerExecutionRole"

bucket = "mlops-sagemaker-project-meerasa"
s3_prefix = "titanic-feature-store"

# -----------------------------
# Step 2: Load Titanic dataset
# -----------------------------
df = pd.read_csv("data/train.csv")

# Add EventTime column (mandatory for Feature Store)
df["EventTime"] = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%SZ")

# Ensure PassengerId is string (for record identifier)
df["PassengerId"] = df["PassengerId"].astype(str)

# -----------------------------
# Step 3: Define Feature Group
# -----------------------------
feature_group_name = f"titanic-feature-group-{strftime('%Y%m%d%H%M%S', gmtime())}"
feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=session)
print(f"Creating Feature Group: {feature_group_name}")

# ✅ Load feature definitions from DataFrame
feature_group.load_feature_definitions(data_frame=df)

# -----------------------------
# Step 4: Create Feature Group
# -----------------------------
feature_group.create(
    s3_uri=f"s3://{bucket}/{s3_prefix}",
    record_identifier_name="PassengerId",
    event_time_feature_name="EventTime",
    role_arn=role,
    enable_online_store=True
)

# -----------------------------
# Step 5: Wait for ACTIVE
# -----------------------------
sm_client = boto3.client("sagemaker", region_name=region)
print("Waiting for Feature Group to become ACTIVE...")

for i in range(30):  # wait up to 15 minutes
    desc = sm_client.describe_feature_group(FeatureGroupName=feature_group_name)
    status = desc["FeatureGroupStatus"]
    print(f"Attempt {i+1}: Status = {status}")
    if status == "Created":
        print("✅ Feature Group is ACTIVE!")
        break
    elif status == "CreateFailed":
        raise RuntimeError(f"❌ Feature Group creation failed: {desc}")
    time.sleep(30)
else:
    raise TimeoutError("❌ Feature Group did not become ACTIVE within 15 minutes.")

# -----------------------------
# Step 6: Ingest Data
# -----------------------------
print("Ingesting records into Feature Store...")
feature_group.ingest(data_frame=df, max_workers=3, wait=True)

print(f"✅ Feature Group {feature_group_name} created and data ingested!")

