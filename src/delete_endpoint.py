import sagemaker

# Initialize session
session = sagemaker.Session()

# Name of the endpoint to delete
endpoint_name = "titanic-xgboost-endpoint"

# Delete endpoint
print(f"🛑 Deleting endpoint: {endpoint_name}")
session.delete_endpoint(endpoint_name=endpoint_name)

print("✅ Endpoint deleted successfully.")

