"""
predict.py
Lambda Function — healthpredict-lambda-predict
POST /predict → validate → normalize → SageMaker → DynamoDB + Redshift → SNS

Environment Variables (configured manually in Lambda Console):
  SAGEMAKER_ENDPOINT_NAME  name of the SageMaker real-time endpoint
  DYNAMODB_TABLE_NAME       name of the DynamoDB predictions table
  SNS_TOPIC_ARN             ARN of the SNS alerts topic
  REDSHIFT_SECRET_ARN       ARN of the Secrets Manager secret
  RISK_THRESHOLD_HIGH       float, default 0.7
  RISK_THRESHOLD_LOW        float, default 0.3
  PREDICTIONS_BUCKET        S3 bucket for Parquet result files
"""

import json
import os
import time
import uuid
import logging
import boto3
import botocore
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── AWS Clients (initialized once per Lambda container) ────
sm_runtime     = boto3.client('sagemaker-runtime')
dynamodb       = boto3.resource('dynamodb')
sns_client     = boto3.client('sns')
secrets_client = boto3.client('secretsmanager')
s3_client      = boto3.client('s3')
redshift_data  = boto3.client('redshift-data')

# ── Environment Variables ──────────────────────────────────
ENDPOINT_NAME    = os.environ['SAGEMAKER_ENDPOINT_NAME']
TABLE_NAME       = os.environ['DYNAMODB_TABLE_NAME']
SNS_TOPIC_ARN    = os.environ['SNS_TOPIC_ARN']
SECRET_ARN       = os.environ.get('REDSHIFT_SECRET_ARN', '')
REDSHIFT_CLUSTER = os.environ.get('REDSHIFT_CLUSTER_ID', 'healthpredict-redshift')
REDSHIFT_DB      = os.environ.get('REDSHIFT_DB', 'healthdb')
REDSHIFT_USER    = os.environ.get('REDSHIFT_USER', 'adminuser')
THRESHOLD_HIGH  = float(os.environ.get('RISK_THRESHOLD_HIGH', '0.7'))
THRESHOLD_LOW   = float(os.environ.get('RISK_THRESHOLD_LOW',  '0.3'))
PRED_BUCKET     = os.environ.get('PREDICTIONS_BUCKET', '')

# ── Normalization constants (from Glue ETL output) ────────
# These are the population statistics from the Pima dataset.
# Lambda reads updated stats from S3 on cold start if available.
DEFAULT_NORM_STATS = {
    'pregnancies':      {'mean': 3.845,   'std': 3.370},
    'glucose':          {'mean': 121.69,  'std': 30.44},
    'blood_pressure':   {'mean': 72.40,   'std': 12.10},
    'skin_thickness':   {'mean': 29.15,   'std': 10.48},
    'insulin':          {'mean': 155.55,  'std': 118.78},
    'bmi':              {'mean': 32.46,   'std': 6.92},
    'diabetes_pedigree':{'mean': 0.4719,  'std': 0.3313},
    'age':              {'mean': 33.24,   'std': 11.76},
}

_norm_stats_cache = None


def get_norm_stats() -> dict:
    """
    Load normalization stats from S3 (updated by Glue ETL job).
    Falls back to DEFAULT_NORM_STATS if unavailable.
    Cached per Lambda container lifetime.
    """
    global _norm_stats_cache
    if _norm_stats_cache:
        return _norm_stats_cache

    if PRED_BUCKET:
        try:
            dataset_bucket = PRED_BUCKET.replace('predictions', 'dataset')
            obj  = s3_client.get_object(Bucket=dataset_bucket,
                                        Key='config/normalization_stats.json')
            stats = json.loads(obj['Body'].read().decode('utf-8'))
            _norm_stats_cache = stats
            logger.info("Loaded normalization stats from S3")
            return stats
        except Exception as e:
            logger.warning(f"Could not load norm stats from S3: {e}. Using defaults.")

    _norm_stats_cache = DEFAULT_NORM_STATS
    return DEFAULT_NORM_STATS


# ── Validation Rules ───────────────────────────────────────
VALIDATION_RULES = {
    'pregnancies':      (0,   20,   int),
    'glucose':          (50,  300,  (int, float)),
    'blood_pressure':   (40,  140,  (int, float)),
    'skin_thickness':   (5,   99,   (int, float)),
    'insulin':          (0,   900,  (int, float)),
    'bmi':              (10,  70,   (int, float)),
    'diabetes_pedigree':(0.0, 2.5,  float),
    'age':              (1,   120,  int),
}

FEATURE_ORDER = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'diabetes_pedigree', 'age'
]

RECOMMENDATIONS = {
    'HIGH':   ('Immediate clinical consultation recommended. '
               'Patient shows multiple high-risk indicators for diabetes.'),
    'MEDIUM': ('Lifestyle intervention recommended. '
               'Schedule follow-up glucose testing within 3 months.'),
    'LOW':    ('Routine annual screening recommended. '
               'Maintain healthy lifestyle habits.'),
}


def validate_input(body: dict) -> list:
    """Return list of validation error messages (empty = valid)."""
    errors = []
    if 'patient_id' not in body:
        errors.append("Missing required field: patient_id")

    for field, (lo, hi, dtype) in VALIDATION_RULES.items():
        if field not in body:
            errors.append(f"Missing required field: {field}")
            continue
        val = body[field]
        if not isinstance(val, dtype):
            try:
                val = float(val) if isinstance(dtype, type) and dtype == float else int(val)
            except (ValueError, TypeError):
                errors.append(f"{field}: expected {dtype.__name__}, got {type(val).__name__}")
                continue
        if not (lo <= val <= hi):
            errors.append(f"{field}: value {val} out of range [{lo}, {hi}]")

    return errors


def normalize_features(body: dict, stats: dict) -> str:
    """
    Z-score normalize features and return as comma-separated string
    in the order expected by the XGBoost model.
    """
    scaled = []
    for feature in FEATURE_ORDER:
        raw  = float(body[feature])
        mean = stats[feature]['mean']
        std  = stats[feature]['std'] or 1.0
        scaled.append((raw - mean) / std)
    return ','.join(f'{v:.6f}' for v in scaled)


def invoke_endpoint_with_retry(payload: str, max_retries: int = 3) -> float:
    """Invoke SageMaker endpoint with exponential backoff retry."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp  = sm_runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='text/csv',
                Body=payload
            )
            score = float(resp['Body'].read().decode('utf-8').strip())
            return score
        except botocore.exceptions.ClientError as e:
            code = e.response['Error']['Code']
            if code in ('ThrottlingException', 'ServiceUnavailable') and attempt < max_retries - 1:
                logger.warning(f"Endpoint attempt {attempt+1} failed ({code}). "
                               f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise RuntimeError("SageMaker endpoint invocation failed after all retries")


def write_dynamodb(record: dict) -> None:
    table = dynamodb.Table(TABLE_NAME)
    table.put_item(Item=record)


def write_redshift(record: dict) -> None:
    """
    Write prediction to Redshift via Redshift Data API.
    Non-critical — failure does not block API response.
    """
    import time as _time
    try:
        def _v(val):
            """Format value for SQL — handle None and Decimal."""
            if val is None:
                return 'NULL'
            return str(float(val))

        sql = f"""
            INSERT INTO healthpredict.prediction_log
                (patient_id, pred_timestamp, pregnancies, glucose, blood_pressure,
                 skin_thickness, insulin, bmi, diabetes_pedigree, age,
                 risk_score, risk_level, model_version)
            VALUES (
                '{record['patient_id']}',
                '{record['prediction_timestamp']}',
                {_v(record.get('pregnancies'))},
                {_v(record.get('glucose'))},
                {_v(record.get('blood_pressure'))},
                {_v(record.get('skin_thickness'))},
                {_v(record.get('insulin'))},
                {_v(record.get('bmi'))},
                {_v(record.get('diabetes_pedigree'))},
                {_v(record.get('age'))},
                {_v(record.get('risk_score'))},
                '{record['risk_level']}',
                '{record.get('model_version', 'v1')}'
            )
        """
        resp    = redshift_data.execute_statement(
            ClusterIdentifier=REDSHIFT_CLUSTER,
            Database=REDSHIFT_DB,
            DbUser=REDSHIFT_USER,
            Sql=sql,
        )
        stmt_id = resp['Id']
        # Fire-and-forget: poll briefly then move on
        for _ in range(5):
            _time.sleep(0.5)
            st = redshift_data.describe_statement(Id=stmt_id)['Status']
            if st in ('FINISHED', 'FAILED', 'ABORTED'):
                break
        logger.info(f"Redshift Data API write submitted: {stmt_id}")
    except Exception as e:
        logger.error(f"Redshift write failed (non-critical): {e}")


def send_sns_alert(record: dict) -> None:
    """Publish HIGH-risk alert to SNS topic (with retry)."""
    message = {
        "alert_type":  "HIGH_RISK_PREDICTION",
        "patient_id":  record['patient_id'],
        "risk_score":  float(record['risk_score']),
        "risk_level":  record['risk_level'],
        "timestamp":   record['prediction_timestamp'],
        "glucose":     record.get('glucose'),
        "bmi":         record.get('bmi'),
        "age":         record.get('age'),
        "recommendation": RECOMMENDATIONS['HIGH'],
    }
    for attempt in range(3):
        try:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject='⚠️ HealthPredict AI — HIGH RISK PATIENT DETECTED',
                Message=json.dumps(message, indent=2),
            )
            logger.info(f"SNS alert sent for patient {record['patient_id']}")
            return
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                logger.error(f"SNS publish failed after 3 attempts: {e}")


def lambda_handler(event, context):
    start_time = time.time()

    # ── Parse Request Body ─────────────────────────────────
    try:
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        elif isinstance(event.get('body'), dict):
            body = event['body']
        else:
            body = event
    except (json.JSONDecodeError, TypeError) as e:
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': f'Invalid JSON: {e}'})
        }

    # ── Validate Input ─────────────────────────────────────
    errors = validate_input(body)
    if errors:
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': 'Validation failed', 'details': errors})
        }

    # ── Normalize & Invoke SageMaker ───────────────────────
    try:
        stats   = get_norm_stats()
        payload = normalize_features(body, stats)
        score   = invoke_endpoint_with_retry(payload)
    except Exception as e:
        logger.error(f"SageMaker invocation error: {e}")
        return {
            'statusCode': 503,
            'headers': {'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': 'Prediction service unavailable', 'detail': str(e)})
        }

    # ── Classify Risk ──────────────────────────────────────
    if score >= THRESHOLD_HIGH:
        risk_level = 'HIGH'
    elif score >= THRESHOLD_LOW:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'

    # ── Build Record ───────────────────────────────────────
    now = datetime.now(timezone.utc)
    ttl = int(now.timestamp()) + (365 * 2 * 24 * 3600)   # 2 years TTL

    def _d(v):
        """Convert numeric value to Decimal for DynamoDB compatibility."""
        if v is None:
            return None
        return Decimal(str(round(float(v), 6)))

    record = {
        'patient_id':             body['patient_id'],
        'prediction_timestamp':   now.isoformat(),
        'risk_score':             _d(score),
        'risk_level':             risk_level,
        'recommendation':         RECOMMENDATIONS[risk_level],
        'model_version':          'v1',
        'model_endpoint':         ENDPOINT_NAME,
        'pregnancies':            _d(body.get('pregnancies')),
        'glucose':                _d(body.get('glucose')),
        'blood_pressure':         _d(body.get('blood_pressure')),
        'skin_thickness':         _d(body.get('skin_thickness')),
        'insulin':                _d(body.get('insulin')),
        'bmi':                    _d(body.get('bmi')),
        'diabetes_pedigree':      _d(body.get('diabetes_pedigree')),
        'age':                    _d(body.get('age')),
        'expiry_time':            ttl,
        'request_id':             context.aws_request_id if context else str(uuid.uuid4()),
    }

    # ── Write DynamoDB (primary store) ─────────────────────
    try:
        write_dynamodb(record)
        logger.info(f"DynamoDB write: OK — patient={body['patient_id']}")
    except Exception as e:
        logger.error(f"DynamoDB write failed: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': 'Failed to persist prediction', 'detail': str(e)})
        }

    # ── Write Redshift (analytics store, non-critical) ─────
    write_redshift(record)

    # ── SNS Alert for HIGH risk ────────────────────────────
    if risk_level == 'HIGH':
        send_sns_alert(record)

    # ── Build Response ─────────────────────────────────────
    latency_ms = round((time.time() - start_time) * 1000, 1)

    response_body = {
        'patient_id':          record['patient_id'],
        'prediction_timestamp': record['prediction_timestamp'],
        'risk_score':           round(score, 4),
        'risk_level':           risk_level,
        'recommendation':       record['recommendation'],
        'model_version':        record['model_version'],
        'latency_ms':           latency_ms,
    }

    logger.info(
        f"Prediction: patient={body['patient_id']} "
        f"score={score:.4f} level={risk_level} "
        f"latency={latency_ms}ms"
    )

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,x-api-key,X-Amz-Date,Authorization',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type,x-api-key',
        },
        'body': json.dumps(response_body)
    }
