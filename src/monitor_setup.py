# src/monitor_setup.py
import os, json, boto3, pathlib, tempfile
from sagemaker import Session
from sagemaker.model_monitor import DefaultModelMonitor, DatasetFormat, EndpointInput

REGION = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT = os.environ["SAGEMAKER_ENDPOINT_NAME"]
BUCKET = os.getenv("ARTIFACT_BUCKET")  # optional
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")  # optional; will be inferred if empty
SCHEDULE_NAME = os.getenv("MONITOR_SCHEDULE_NAME", f"{ENDPOINT}-dataquality")
BASELINE_LOCAL = os.getenv("BASELINE_LOCAL", "data/baseline.csv")

b3 = boto3.Session(region_name=REGION)
sm = b3.client("sagemaker")
s3 = b3.client("s3")
sess = Session(boto_session=b3)

def ensure_bucket(name=None):
    if name:
        try: s3.head_bucket(Bucket=name)
        except: s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": REGION})
        return name
    # default bucket
    name = sess.default_bucket()
    try: s3.head_bucket(Bucket=name)
    except: s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": REGION})
    return name

def infer_role_from_endpoint(endpoint):
    ep = sm.describe_endpoint(EndpointName=endpoint)
    cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
    model_name = cfg["ProductionVariants"][0]["ModelName"]
    model = sm.describe_model(ModelName=model_name)
    return model.get("ExecutionRoleArn")

def main():
    bucket = ensure_bucket(BUCKET)
    role = ROLE_ARN or infer_role_from_endpoint(ENDPOINT)
    print("Using role:", role)
    print("Using bucket:", bucket)

    # Prepare a baseline dataset
    local_path = pathlib.Path(BASELINE_LOCAL)
    if not local_path.exists():
        print(f"{BASELINE_LOCAL} not found. Creating a tiny CSV baseline matching inference shape.")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        with open(tmp.name, "w") as f:
            row = "3,1,34,0,0,7.8292,2\n"
            f.writelines([row for _ in range(50)])
        local_path = pathlib.Path(tmp.name)

    baseline_s3 = f"s3://{bucket}/monitoring/baseline/{ENDPOINT}/baseline.csv"
    sess.upload_data(str(local_path), bucket, f"monitoring/baseline/{ENDPOINT}/baseline.csv")
    print("Baseline uploaded to:", baseline_s3)

    mon = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=sess,
        base_job_name=f"{ENDPOINT}-dq",
    )

    baseline_job = mon.suggest_baseline(
        baseline_dataset=baseline_s3,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=f"s3://{bucket}/monitoring/baseline/{ENDPOINT}/outputs",
        wait=True,
    )

    stats = baseline_job.baseline_statistics()
    constraints = baseline_job.suggested_constraints()
    print("Stats:", stats)
    print("Constraints:", constraints)

    mon.create_monitoring_schedule(
        monitor_schedule_name=SCHEDULE_NAME,
        endpoint_input=EndpointInput(endpoint_name=ENDPOINT),
        output_s3_uri=f"s3://{bucket}/monitoring/reports/{ENDPOINT}",
        statistics=stats,
        constraints=constraints,
        schedule_cron_expression="cron(0 * * * ? *)",  # hourly
    )
    desc = sm.describe_monitoring_schedule(MonitoringScheduleName=SCHEDULE_NAME)
    print("Monitoring schedule status:", desc["MonitoringScheduleStatus"])

if __name__ == "__main__":
    main()

