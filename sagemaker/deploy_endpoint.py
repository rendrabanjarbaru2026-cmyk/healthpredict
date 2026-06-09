"""
deploy_endpoint.py
Deploy an Approved model from SageMaker Model Registry to a real-time endpoint.

Run this from SageMaker Studio terminal (File → New → Terminal):
  python deploy_endpoint.py

AWS Academy Learner Lab Notes:
  - Must run from SageMaker Studio (CLI outside Studio is restricted)
  - Uses ml.t2.medium — cheapest inference instance
  - ALWAYS delete endpoint after testing to preserve credits
"""

import boto3
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────
STUDENT_NAME   = "Rendra"          # ← CHANGE THIS
ACCOUNT_ID     = boto3.client("sts").get_caller_identity()["Account"]
REGION         = "us-east-1"
ROLE_ARN       = f"arn:aws:iam::{973364355372}:role/LabRole"

PROJECT        = "healthpredict"
MODEL_GROUP    = f"{PROJECT}-model-group"
ENDPOINT_NAME  = f"{PROJECT}-endpoint-diabetes"
CONFIG_NAME    = f"{PROJECT}-endpoint-config"
MODEL_NAME     = f"{PROJECT}-model-diabetes"

sm     = boto3.client("sagemaker",          region_name=REGION)
sm_rt  = boto3.client("sagemaker-runtime",  region_name=REGION)


def get_latest_approved_model_arn() -> str:
    pkgs = sm.list_model_packages(
        ModelPackageGroupName=MODEL_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["ModelPackageSummaryList"]
    if not pkgs:
        raise RuntimeError(
            f"No Approved model packages in {MODEL_GROUP}. "
            "Approve the model in Model Registry first."
        )
    arn = pkgs[0]["ModelPackageArn"]
    logger.info(f"Latest approved model: ...{arn[-40:]}")
    return arn


def deploy():
    pkg_arn = get_latest_approved_model_arn()

    # Clean up existing resources
    for res, fn, key in [
        (ENDPOINT_NAME, sm.delete_endpoint,        "EndpointName"),
        (CONFIG_NAME,   sm.delete_endpoint_config, "EndpointConfigName"),
        (MODEL_NAME,    sm.delete_model,           "ModelName"),
    ]:
        try:
            fn(**{key: res})
            logger.info(f"  Deleted existing {key}: {res}")
            time.sleep(5)
        except Exception:
            pass

    # Create model
    sm.create_model(
        ModelName=MODEL_NAME,
        ExecutionRoleArn=ROLE_ARN,
        Containers=[{"ModelPackageName": pkg_arn}],
    )
    logger.info(f"Model created: {MODEL_NAME}")

    # Create endpoint config
    sm.create_endpoint_config(
        EndpointConfigName=CONFIG_NAME,
        ProductionVariants=[{
            "VariantName":          "primary",
            "ModelName":            MODEL_NAME,
            "InstanceType":         "ml.t2.medium",
            "InitialInstanceCount": 1,
        }],
    )
    logger.info(f"Endpoint config created: {CONFIG_NAME}")

    # Create endpoint
    sm.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=CONFIG_NAME,
    )
    logger.info(f"Endpoint creation started: {ENDPOINT_NAME}")
    logger.info("Waiting for InService (~5 min)...")

    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=ENDPOINT_NAME,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},
    )

    # Smoke test
    score = float(sm_rt.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body="6,148,72,35,0,33.6,0.627,50",
    )["Body"].read().decode().strip())

    logger.info("=" * 60)
    logger.info(f"Endpoint InService: {ENDPOINT_NAME}")
    logger.info(f"Smoke test score:   {score:.4f}")
    logger.info("IMPORTANT: Delete the endpoint after testing to avoid unnecessary charges.")
    logger.info("=" * 60)


if __name__ == "__main__":
    deploy()
