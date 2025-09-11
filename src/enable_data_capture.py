# src/enable_data_capture.py
import os, time, boto3

REGION = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT = os.environ["SAGEMAKER_ENDPOINT_NAME"]
BUCKET = os.getenv("ARTIFACT_BUCKET")  # optional; if not set we'll fall back to default bucket
CAPTURE_PREFIX = os.getenv("CAPTURE_PREFIX", "datacapture")

sm = boto3.client("sagemaker", region_name=REGION)

def default_bucket():
    s3 = boto3.client("s3", region_name=REGION)
    acct = boto3.client("sts").get_caller_identity()["Account"]
    name = f"sagemaker-{REGION}-{acct}"
    # create if missing
    try: s3.head_bucket(Bucket=name)
    except: s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": REGION})
    return name

def main():
    print(f"Enabling data capture on endpoint: {ENDPOINT}")
    ep = sm.describe_endpoint(EndpointName=ENDPOINT)
    old_cfg = ep["EndpointConfigName"]
    cfg = sm.describe_endpoint_config(EndpointConfigName=old_cfg)

    bucket = BUCKET or default_bucket()
    dest = f"s3://{bucket}/{CAPTURE_PREFIX}/{ENDPOINT}/"

    new_cfg_name = f"{old_cfg}-capture-{int(time.time())}"
    sm.create_endpoint_config(
        EndpointConfigName=new_cfg_name,
        ProductionVariants=cfg["ProductionVariants"],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": dest,
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {"CsvContentTypes": ["text/csv"]},
        },
        Tags=cfg.get("Tags", []),
    )
    sm.update_endpoint(EndpointName=ENDPOINT, EndpointConfigName=new_cfg_name)
    print(f"Updated endpoint to config: {new_cfg_name}\nCapture destination: {dest}")

if __name__ == "__main__":
    main()
