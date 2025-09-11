# pipelines/pipeline_up.py
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

# NOTE: RegisterModel import path varies by SDK version; this one is most compatible.
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
    # Fallback: infer from your existing endpoint (we already use this in smoke tests)
    EP = os.getenv("SAGEMAKER_ENDPOINT_NAME")
    if not EP:
        raise RuntimeError(
            "Provide SAGEMAKER_ROLE_ARN env (preferred) or SAGEMAKER_ENDPOINT_NAME "
            "to infer a role from the deployed model."
        )
    ROLE_ARN = infer_role_from_endpoint(EP)

# -------- Pipeline Parameters (runtime-overridable) --------
InputDataUri = ParameterString(
    name="InputDataUr
