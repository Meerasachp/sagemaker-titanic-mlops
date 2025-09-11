# src/register_model.py
import os, boto3, time, botocore

REGION = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT = os.environ["SAGEMAKER_ENDPOINT_NAME"]
GROUP = os.getenv("MODEL_PACKAGE_GROUP", "titanic-xgboost")

sm = boto3.client("sagemaker", region_name=REGION)

def ensure_group(name):
    try:
        sm.describe_model_package_group(ModelPackageGroupName=name)
    except botocore.exceptions.ClientError as e:
        if "ModelPackageGroupNotFoundException" in str(e):
            sm.create_model_package_group(
                ModelPackageGroupName=name,
                ModelPackageGroupDescription="Registered from CI",
            )
        else:
            raise

def main():
    ep = sm.describe_endpoint(EndpointName=ENDPOINT)
    cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
    model_name = cfg["ProductionVariants"][0]["ModelName"]
    model = sm.describe_model(ModelName=model_name)
    container = model["PrimaryContainer"]
    image_uri = container["Image"]
    model_data = container["ModelDataUrl"]

    ensure_group(GROUP)
    print("Registering model to group:", GROUP)
    resp = sm.create_model_package(
        ModelPackageGroupName=GROUP,
        ModelPackageDescription=f"From endpoint {ENDPOINT} at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        InferenceSpecification={
            "Containers": [{"Image": image_uri, "ModelDataUrl": model_data}],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="Approved",
        Tags=[{"Key":"project","Value":"titanic-mlops"}],
    )
    print("ModelPackageArn:", resp["ModelPackageArn"])

if __name__ == "__main__":
    main()

