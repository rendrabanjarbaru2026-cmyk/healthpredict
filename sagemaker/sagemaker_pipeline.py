"""
sagemaker_pipeline.py
SageMaker Pipelines — HealthPredict AI
3-Step Pipeline: SKLearn Processing → XGBoost Training → Model Registration

Run from SageMaker Studio terminal or AWS CloudShell:
  python sagemaker_pipeline.py

AWS Academy Learner Lab Notes:
  - Uses LabRole for SageMaker execution
  - ml.m5.large for training (avoid large instances to preserve credits)
  - ml.t2.medium for endpoint (cheapest inference option)
  - After running, go to SageMaker Studio > Pipelines to trigger execution
"""

import boto3
import sagemaker
import logging
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep, TrainingStep, TransformStep
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── CONFIG — update STUDENT_NAME before running ──────────────
STUDENT_NAME   = "yourname"          # ← CHANGE THIS
ACCOUNT_ID     = boto3.client('sts').get_caller_identity()['Account']
REGION         = "us-east-1"
ROLE_ARN       = f"arn:aws:iam::{ACCOUNT_ID}:role/LabRole"

PROJECT        = "healthpredict"
DATASET_BUCKET = f"{PROJECT}-dataset-{STUDENT_NAME}-2026"
MODELS_BUCKET  = f"{PROJECT}-models-{STUDENT_NAME}-2026"
PIPELINE_NAME  = f"{PROJECT}-training-pipeline"
MODEL_GROUP    = f"{PROJECT}-model-group"

session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

logger.info(f"Account ID:   {ACCOUNT_ID}")
logger.info(f"Role ARN:     {ROLE_ARN}")
logger.info(f"Dataset S3:   s3://{DATASET_BUCKET}")
logger.info(f"Models S3:    s3://{MODELS_BUCKET}")


# ════════════════════════════════════════════════════════════
# STEP 0 — Create Model Package Group (if not exists)
# ════════════════════════════════════════════════════════════

def ensure_model_group():
    sm = boto3.client('sagemaker', region_name=REGION)
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=MODEL_GROUP,
            ModelPackageGroupDescription=(
                "HealthPredict AI — XGBoost diabetes risk prediction models"
            )
        )
        logger.info(f"Created model package group: {MODEL_GROUP}")
    except sm.exceptions.ClientError as e:
        if "already exists" in str(e).lower():
            logger.info(f"Model package group already exists: {MODEL_GROUP}")
        else:
            raise


# ════════════════════════════════════════════════════════════
# PIPELINE PARAMETERS
# ════════════════════════════════════════════════════════════

p_processing_instance = ParameterString(
    name="ProcessingInstanceType",
    default_value="ml.m5.large"
)
p_training_instance = ParameterString(
    name="TrainingInstanceType",
    default_value="ml.m5.large"
)
p_min_auc = ParameterFloat(
    name="MinimumAUCThreshold",
    default_value=0.75      # Lowered for Lab environment
)


# ════════════════════════════════════════════════════════════
# STEP 1 — PROCESSING STEP (SKLearnProcessor)
# Reads processed Parquet → ensures correct format for XGBoost
# ════════════════════════════════════════════════════════════

def build_processing_step():
    logger.info("Building Processing Step...")

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=p_processing_instance,
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=session,
        base_job_name=f"{PROJECT}-processing",
    )

    step = ProcessingStep(
        name="DataPreparationStep",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{DATASET_BUCKET}/train/",
                destination="/opt/ml/processing/input/train",
                input_name="train-data",
            ),
            ProcessingInput(
                source=f"s3://{DATASET_BUCKET}/validation/",
                destination="/opt/ml/processing/input/validation",
                input_name="val-data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train-output",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{DATASET_BUCKET}/pipeline-train/",
            ),
            ProcessingOutput(
                output_name="val-output",
                source="/opt/ml/processing/output/validation",
                destination=f"s3://{DATASET_BUCKET}/pipeline-validation/",
            ),
        ],
        code="sagemaker/processing_script.py",
    )

    return step


# ════════════════════════════════════════════════════════════
# STEP 2 — TRAINING STEP (XGBoost Built-in)
# ════════════════════════════════════════════════════════════

def build_training_step(processing_step):
    logger.info("Building Training Step...")

    xgb_image = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=REGION,
        version="1.7-1"
    )

    estimator = Estimator(
        image_uri=xgb_image,
        role=ROLE_ARN,
        instance_count=1,
        instance_type=p_training_instance,
        volume_size=10,
        max_run=3600,
        output_path=f"s3://{MODELS_BUCKET}/pipeline-output/",
        sagemaker_session=session,
        base_job_name=f"{PROJECT}-training",
        hyperparameters={
            "num_round":             100,
            "max_depth":             5,
            "eta":                   0.2,
            "objective":             "binary:logistic",
            "eval_metric":           "auc",
            "subsample":             0.8,
            "colsample_bytree":      0.8,
            "min_child_weight":      6,
            "gamma":                 0.1,
            "early_stopping_rounds": 10,
            "seed":                  42,
        }
    )

    step = TrainingStep(
        name="XGBoostTrainingStep",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig
                        .Outputs["train-output"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig
                        .Outputs["val-output"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

    return step


# ════════════════════════════════════════════════════════════
# STEP 3 — MODEL REGISTRATION STEP
# Registers model in SageMaker Model Registry with PendingManualApproval
# ════════════════════════════════════════════════════════════

def build_register_step(training_step):
    logger.info("Building Model Registration Step...")

    xgb_image = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=REGION,
        version="1.7-1"
    )

    model = Model(
        image_uri=xgb_image,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=ROLE_ARN,
        sagemaker_session=session,
        name=f"{PROJECT}-model",
    )

    register_step = ModelStep(
        name="ModelRegistrationStep",
        step_args=model.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium", "ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=MODEL_GROUP,
            approval_status="PendingManualApproval",
            description="XGBoost 1.7 diabetes risk prediction model",
        )
    )

    return register_step


# ════════════════════════════════════════════════════════════
# ASSEMBLE & REGISTER PIPELINE
# ════════════════════════════════════════════════════════════

def build_pipeline():
    ensure_model_group()

    processing_step = build_processing_step()
    training_step   = build_training_step(processing_step)
    register_step   = build_register_step(training_step)

    # Chain dependencies
    training_step.add_depends_on([processing_step])
    register_step.add_depends_on([training_step])

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            p_processing_instance,
            p_training_instance,
            p_min_auc,
        ],
        steps=[processing_step, training_step, register_step],
        sagemaker_session=session,
    )

    return pipeline


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("HEALTHPREDICT AI — REGISTERING SAGEMAKER PIPELINE")
    logger.info("=" * 60)

    pipeline = build_pipeline()

    # Upsert pipeline definition (create or update)
    pipeline.upsert(role_arn=ROLE_ARN)
    logger.info(f"\nPipeline registered: {PIPELINE_NAME}")
    logger.info("Pipeline definition saved. Execute from SageMaker Studio → Pipelines.")
