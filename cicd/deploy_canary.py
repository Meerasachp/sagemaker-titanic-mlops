import os, json, time, boto3, sagemaker

REGION      = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT    = os.environ["SAGEMAKER_ENDPOINT_NAME"]     # from GitHub secret
MODEL_DATA  = os.environ["MODEL_ARTIFACT_S3"]           # set by train step or workflow input
ROLE        = os.environ["SAGEMAKER_ROLE_ARN"]          # from secret
VARIANT_OLD = os.getenv("VARIANT_OLD", "AllTraffic")
VARIANT_NEW = os.getenv("VARIANT_NEW", "Canary")
NEW_WEIGHT  = float(os.getenv("CANARY_WEIGHT", "0.1"))

sm = boto3.client("sagemaker", region_name=REGION)
rt = boto3.client("sagemaker-runtime", region_name=REGION)

# 1) Describe current endpoint
econf_name = sm.describe_endpoint(EndpointName=ENDPOINT)["EndpointConfigName"]
econf = sm.describe_endpoint_config(EndpointConfigName=econf_name)
pv = econf["ProductionVariants"][0]
old_model_name = pv["ModelName"]
variant_name = pv["VariantName"]

# 2) Create a new Model backed by MODEL_DATA
model_name = f"{ENDPOINT}-xgb-{int(time.time())}"
container = {
    "Image": sagemaker.image_uris.retrieve("xgboost", region=REGION, version="1.7-1"),
    "Mode": "SingleModel",
    "ModelDataUrl": MODEL_DATA,
    "Environment": {}
}
sm.create_model(ModelName=model_name, PrimaryContainer=container, ExecutionRoleArn=ROLE)

# 3) Create new EndpointConfig with two variants (old + new)
new_ec = f"{ENDPOINT}-canary-ec-{int(time.time())}"
variants = [
    {
        "VariantName": variant_name,           # old
        "ModelName": old_model_name,
        "InitialInstanceCount": pv.get("InitialInstanceCount", 1),
        "InstanceType": pv.get("InstanceType", "ml.m5.large"),
        "InitialVariantWeight": 1.0 - NEW_WEIGHT
    },
    {
        "VariantName": VARIANT_NEW,            # new
        "ModelName": model_name,
        "InitialInstanceCount": pv.get("InitialInstanceCount", 1),
        "InstanceType": pv.get("InstanceType", "ml.m5.large"),
        "InitialVariantWeight": NEW_WEIGHT
    }
]
sm.create_endpoint_config(EndpointConfigName=new_ec, ProductionVariants=variants)

# 4) Update endpoint
sm.update_endpoint(EndpointName=ENDPOINT, EndpointConfigName=new_ec)

# 5) Wait until InService
while True:
    status = sm.describe_endpoint(EndpointName=ENDPOINT)["EndpointStatus"]
    print("Endpoint status:", status, flush=True)
    if status in ("InService", "Failed"):
        break
    time.sleep(30)

if status != "InService":
    raise RuntimeError("Endpoint update failed; check CloudWatch logs")

print(json.dumps({
    "new_model": model_name,
    "endpoint_config": new_ec,
    "canary_weight": NEW_WEIGHT
}, indent=2))
