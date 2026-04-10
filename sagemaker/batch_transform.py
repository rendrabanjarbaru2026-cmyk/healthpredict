"""
batch_transform.py
SageMaker Batch Transform — HealthPredict AI
Runs bulk inference on validation dataset stored in S3.

AWS Academy Notes:
- Uses ml.m5.large (no persistent endpoint cost)
- Job terminates automatically after completion
- Output: one score per line in S3 predictions/batch/

Usage:
  python batch_transform.py
"""

import boto3
import sagemaker
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────
STUDENT_NAME  = "yourname"   # ← CHANGE THIS
ACCOUNT_ID    = boto3.client('sts').get_caller_identity()['Account']
REGION        = "us-east-1"
ROLE_ARN      = f"arn:aws:iam::{ACCOUNT_ID}:role/LabRole"

PROJECT       = "healthpredict"
MODEL_NAME    = f"{PROJECT}-model-diabetes"
DATASET_BUCKET = f"{PROJECT}-dataset-{STUDENT_NAME}-2026"
PRED_BUCKET   = f"{PROJECT}-predictions-{STUDENT_NAME}-2026"
JOB_NAME      = f"{PROJECT}-batch-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

sm = boto3.client('sagemaker', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)


def run_batch_transform():
    logger.info("=" * 55)
    logger.info("BATCH TRANSFORM JOB — HEALTHPREDICT AI")
    logger.info(f"Job name: {JOB_NAME}")
    logger.info("=" * 55)

    sm.create_transform_job(
        TransformJobName=JOB_NAME,
        ModelName=MODEL_NAME,
        MaxConcurrentTransforms=1,
        MaxPayloadInMB=6,
        BatchStrategy="MultiRecord",
        TransformInput={
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{DATASET_BUCKET}/validation/",
                }
            },
            "ContentType":  "text/csv",
            "SplitType":    "Line",
        },
        TransformOutput={
            "S3OutputPath":  f"s3://{PRED_BUCKET}/batch/",
            "AssembleWith":  "Line",
            "Accept":        "text/csv",
        },
        TransformResources={
            "InstanceType":  "ml.m5.large",
            "InstanceCount": 1,
        },
        Tags=[{"Key": "Project", "Value": PROJECT}]
    )

    logger.info("Batch Transform job started. Waiting for completion...")

    # Poll every 30 seconds
    while True:
        desc   = sm.describe_transform_job(TransformJobName=JOB_NAME)
        status = desc['TransformJobStatus']
        logger.info(f"  Status: {status}")

        if status == "Completed":
            logger.info("Batch Transform completed successfully!")
            break
        elif status in ("Failed", "Stopped"):
            reason = desc.get('FailureReason', 'Unknown')
            raise RuntimeError(f"Batch Transform failed: {reason}")

        time.sleep(30)

    # Verify output exists
    resp = s3.list_objects_v2(Bucket=PRED_BUCKET, Prefix="batch/", MaxKeys=5)
    files = resp.get('Contents', [])
    logger.info(f"\nOutput files in s3://{PRED_BUCKET}/batch/:")
    for f in files:
        logger.info(f"  {f['Key']} ({f['Size']} bytes)")

    # Preview first 5 scores
    if files:
        obj  = s3.get_object(Bucket=PRED_BUCKET, Key=files[0]['Key'])
        body = obj['Body'].read().decode('utf-8')
        scores = [float(x) for x in body.strip().split('\n') if x.strip()][:5]
        logger.info(f"\nFirst 5 prediction scores: {[round(s, 4) for s in scores]}")
        high  = sum(1 for s in scores if s >= 0.7)
        low   = sum(1 for s in scores if s  < 0.3)
        logger.info(f"Sample — HIGH: {high}, LOW: {low}, MEDIUM: {len(scores)-high-low}")


if __name__ == "__main__":
    run_batch_transform()
