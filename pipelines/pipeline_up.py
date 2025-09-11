# pipelines/pipeline_up.py
import os, sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.functions import Join
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker import image_uris

REGION = os.getenv("AWS_REGION", "us-east-1")
sess   = sagemaker.Session()

def infer_role_from_endpoint(endpoint_name: str) -> str:
    import boto3
    sm = boto3.client("sagemaker", region_name=REGION)
    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
    model = sm.describe_model(ModelName=cfg["ProductionVariants"][0]["ModelName"])
    return model["ExecutionRoleArn"]

# ---- Execution role for jobs in this pipeline ----
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
if not ROLE_ARN:
    ep = os.getenv("SAGEMAKER_ENDPOINT_NAME")  # we already have this secret
    if not ep:
        raise RuntimeError("Set SAGEMAKER_ROLE_ARN env or SAGEMAKER_ENDPOINT_NAME to infer role.")
    ROLE_ARN = infer_role_from_endpoint(ep)

# ---- Parameters (override at runtime if you want) ----
InputDataUri = ParameterString("InputDataUri", default_value="s3://mlops-demo-project/raw/titanic.csv")
ModelPackageGroup = ParameterString("ModelPackageGroup", default_value="titanic-xgboost")
AUCThreshold = ParameterFloat("AUCThreshold", default_value=0.80)

cache = CacheConfig(enable_caching=True, expire_after="30d")

# ---- Step: Preprocess (your existing script) ----
sk = SKLearnProcessor(
    framework_version="1.2-1",
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=sess,
)
pre = ProcessingStep(
    name="Preprocess",
    processor=sk,
    code="src/preprocess.py",
    inputs=[ProcessingInput(source=InputDataUri, destination="/opt/ml/processing/input")],
    job_arguments=["--input","/opt/ml/processing/input/titanic.csv","--target","Survived"],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/train", output_name="train"),
        ProcessingOutput(source="/opt/ml/processing/test",  output_name="test"),
    ],
    cache_config=cache,
)

# ---- Step: Train (built-in XGBoost) ----
xgb_image = image_uris.retrieve("xgboost", region=REGION, version="1.7-1")
est = Estimator(
    image_uri=xgb_image,
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{sess.default_bucket()}/xgb/output/",
    sagemaker_session=sess,
    hyperparameters={
        "objective":"binary:logistic",
        "eval_metric":"auc",
        "num_round":200,
        "max_depth":5,
        "eta":0.2,
        "subsample":0.8,
        "min_child_weight":2,
    },
)
train = TrainingStep(
    name="Train",
    estimator=est,
    inputs={
        "train": TrainingInput(
            s3_data=pre.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": TrainingInput(
            s3_data=pre.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            content_type="text/csv"
        ),
    },
    cache_config=cache,
)

# ---- Step: Evaluate -> write metrics.json ----
eval_prop = PropertyFile(name="EvalMetrics", output_name="metrics", path="metrics.json")
eval_proc = SKLearnProcessor(
    framework_version="1.2-1",
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=sess,
)
evaluate = ProcessingStep(
    name="Evaluate",
    processor=eval_proc,
    code="src/evaluate.py",
    job_arguments=[
        "--test", Join(on="", values=[pre.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, "/test.csv"]),
        "--model_artifact", train.properties.ModelArtifacts.S3ModelArtifacts,
        "--out", "/opt/ml/processing/output/metrics.json",
    ],
    outputs=[ProcessingOutput(source="/opt/ml/processing/output", output_name="metrics")],
    property_files=[eval_prop],
)

# ---- Model metrics (for Registry) ----
metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(on="", values=[evaluate.properties.ProcessingOutputConfig.Outputs["metrics"].S3Output.S3Uri, "/metrics.json"]),
        content_type="application/json",
    )
)

# ---- Register model when AUC passes threshold ----
from sagemaker.workflow.model_step import RegisterModel
register = RegisterModel(
    name="RegisterModel",
    estimator=est,
    model_data=train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large","ml.t2.medium"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=ModelPackageGroup,
    model_metrics=metrics,
    approval_status="PendingManualApproval"  # keep human in the loop
)

gate = ConditionStep(
    name="QualityGate",
    conditions=[ConditionGreaterThan(left=eval_prop.prop("auc"), right=AUCThreshold)],
    if_steps=[register],
    else_steps=[],
)

pipeline = Pipeline(
    name="TitanicXGBPipeline",
    parameters=[InputDataUri, ModelPackageGroup, AUCThreshold],
    steps=[pre, train, evaluate, gate],
    sagemaker_session=sess,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=ROLE_ARN)
    ex = pipeline.start()
    print("Started execution:", ex.arn)

