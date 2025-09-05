import boto3

# -----------------------------
# Step 1: Set up SageMaker client
# -----------------------------
region = "us-east-1"
fg_name = "titanic-feature-group-20250904192821"  # Replace with your FG name
sm_client = boto3.client("sagemaker", region_name=region)

# -----------------------------
# Step 2: Describe Feature Group
# -----------------------------
try:
    response = sm_client.describe_feature_group(FeatureGroupName=fg_name)
    print("=== Feature Group Status Check ===")
    print("Name:", response["FeatureGroupName"])
    print("Status:", response["FeatureGroupStatus"])
    print("CreationTime:", response["CreationTime"])
    print("OfflineStoreStatus:", response.get("OfflineStoreStatus", {}))
    print("OnlineStoreConfig:", response.get("OnlineStoreConfig", {}))
except sm_client.exceptions.ResourceNotFound:
    print(f"❌ Feature Group '{fg_name}' not found in region {region}.")

