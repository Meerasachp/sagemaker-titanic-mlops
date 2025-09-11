import os
import boto3
import sagemaker

from sagemaker import image_uris
from sagemaker.session import Session
from sagemaker.estimator import Estimator

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.functions import Join
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.model_metrics import ModelMetrics, MetricsSource

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# Compatible import across SDK versions
from sagemaker.workflow.step_collections import RegisterModel


REGION = os.getenv("AWS_REGION", "us-east-1")
sess: Session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
sm = boto3.client("sagemaker", region_name=REGION)


def infer_role_from_endpoint(endpoint_name: str) -> str:
    """Infer the execution role from the currently deployed model behind an endpoint."""
    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
    model_name = cfg["ProductionVariants"][0]["ModelName"]
    model = sm.describe_model(ModelName=model_name)
    return model["ExecutionRoleArn"]


# -------- Execution Role --------
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
if not ROLE_ARN:
    EP = os.getenv("SAGEMAKER_ENDPOINT_NAME")
    if not EP:
        raise RuntimeError(
            "Provide SAGEMAKER_ROLE_ARN env (preferred) or SAGEMAKER_ENDPOINT_NAME "
            "to infer a role from the deployed model."
        )
    ROLE_ARN = infer_role_from_endpoint(EP)

# -------- Pipeline Parameters --------
InputDataUri = ParameterString(
    name="InputDataUri",
    default_value=os.getenv("INPUT_DATA_URI", "s3://mlops-demo-project/raw/titanic.csv"),
)
ModelPackageGroup = ParameterString(
    name="ModelPackageGroup",
    default_value=os.getenv("MODEL_PACKAGE_GROUP", "titanic-xgboost"),
)
AUCThreshold = ParameterFloat(
    name="AUCThreshold",
    default_value=float(os.getenv("AUC_THRESHOLD", "0.80")),
)

cache = CacheConfig(enable_caching=True, expire_after="30d")

# -------- Step: Preprocess (uses YOUR src/preprocess.py) --------
sk_proc = SKLearnProcessor(
    framework_version="1.2-1",
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=sess,
)

preprocess_step = ProcessingStep(
    name="Preprocess",
    processor=sk_proc,
    code="src/preprocess.py",
    inputs=[
        ProcessingInput(
            source=InputDataUri,  # s3://.../titanic.csv
            destination="/opt/ml/processing/input",
        )
    ],
    job_arguments=[
        "--input", "/opt/ml/processing/input/titanic.csv",
        "--target", "Survived",
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/train", output_name="train"),
        ProcessingOutput(source="/opt/ml/processing/test",  output_name="test"),
    ],
    cache_config=cache,
)

# -------- Step: Train (built-in XGBoost) --------
xgb_image = image_uris.retrieve("xgboost", region=REGION, version="1.7-1")
est = Estimator(
    image_uri=xgb_image,
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{sess.default_bucket()}/xgb/output/",
    sagemaker_session=sess,
    hyperparameters={
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "num_round": 200,
        "max_depth": 5,
        "eta": 0.2,
        "subsample": 0.8,
        "min_child_weight": 2,
    },
)

train_step = TrainingStep(
    name="Train",
    estimator=est,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            content_type="text/csv",
        ),
    },
    cache_config=cache,
)

# -------- Step: Evaluate --------
eval_prop = PropertyFile(name="EvalMetrics", output_name="metrics", path="metrics.json")

eval_proc = SKLearnProcessor(
    framework_version="1.2-1",
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=sess,
)

evaluate_step = ProcessingStep(
    name="Evaluate",
    processor=eval_proc,
    code="src/evaluate.py",
    inputs=[
        ProcessingInput(
            source=preprocess_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        )
    ],
    job_arguments=[
        "--test", "/opt/ml/processing/test/test.csv",
        "--model_artifact", train_step.properties.ModelArtifacts.S3ModelArtifacts,
        "--out", "/opt/ml/processing/output/metrics.json",
    ],
    outputs=[ProcessingOutput(source="/opt/ml/processing/output", output_name="metrics")],
    property_files=[eval_prop],
)

# -------- Metrics & Registration --------
metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(
            on="",
            values=[
                evaluate_step.properties.ProcessingOutputConfig.Outputs["metrics"].S3Output.S3Uri,
                "/metrics.json",
            ],
        ),
        content_type="application/json",
    )
)

register_step = RegisterModel(
    name="RegisterModel",
    estimator=est,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["application/json"],
    inference_instances=["ml.t2.medium", "ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=ModelPackageGroup,
    model_metrics=metrics,
    approval_status="PendingManualApproval",
)

gate_step = ConditionStep(
    name="QualityGate",
    conditions=[ConditionGreaterThan(left=eval_prop.prop("auc"), right=AUCThreshold)],
    if_steps=[register_step],
    else_steps=[],
)

# -------- Assemble Pipeline --------
pipeline = Pipeline(
    name="TitanicXGBPipeline",
    parameters=[InputDataUri, ModelPackageGroup, AUCThreshold],
    steps=[preprocess_step, train_step, evaluate_step, gate_step],
    sagemaker_session=sess,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=ROLE_ARN)
    execution = pipeline.start()
    print("Started execution:", execution.arn)

