import boto3

# Init SageMaker client
sm_client = boto3.client("sagemaker", region_name="us-east-1")

# Replace with your actual FG name from logs
fg_name = "titanic-feature-group-20250904144218"

# Describe FG
response = sm_client.describe_feature_group(FeatureGroupName=fg_name)

print("=== Feature Group Status Check ===")
print("Name:", response["FeatureGroupName"])
print("Status:", response["FeatureGroupStatus"])
print("CreationTime:", response["CreationTime"])
print("OfflineStoreStatus:", response.get("OfflineStoreStatus", {}))
print("OnlineStoreStatus:", response.get("OnlineStoreConfig", {}))

