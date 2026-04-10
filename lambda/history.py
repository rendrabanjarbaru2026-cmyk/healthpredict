"""
history.py
Lambda Function — healthpredict-lambda-history
GET /predict/{patient_id}  →  query DynamoDB by patient
GET /history               →  scan DynamoDB (with optional ?analytics=true for Redshift stats)
GET /history?analytics=true → DynamoDB records + Redshift aggregated analytics
"""

import json
import os
import logging
import time
import boto3
from boto3.dynamodb.conditions import Key, Attr
from decimal import Decimal

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb       = boto3.resource('dynamodb')
redshift_data  = boto3.client('redshift-data')

TABLE_NAME       = os.environ['DYNAMODB_TABLE_NAME']
SECRET_ARN       = os.environ.get('REDSHIFT_SECRET_ARN', '')
REDSHIFT_CLUSTER = os.environ.get('REDSHIFT_CLUSTER_ID', 'healthpredict-redshift')
REDSHIFT_DB      = os.environ.get('REDSHIFT_DB', 'healthdb')
REDSHIFT_USER    = os.environ.get('REDSHIFT_USER', 'adminuser')


# ── Decimal serializer for DynamoDB ───────────────────────
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def query_dynamodb_by_patient(patient_id: str, limit: int = 20) -> list:
    table  = dynamodb.Table(TABLE_NAME)
    resp   = table.query(
        KeyConditionExpression=Key('patient_id').eq(patient_id),
        ScanIndexForward=False,   # newest first
        Limit=limit,
    )
    return resp.get('Items', [])


def scan_dynamodb(risk_filter: str = None, limit: int = 50) -> list:
    table  = dynamodb.Table(TABLE_NAME)
    kwargs = {'Limit': limit}
    if risk_filter:
        kwargs['FilterExpression'] = Attr('risk_level').eq(risk_filter.upper())
    resp = table.scan(**kwargs)
    items = resp.get('Items', [])
    # Sort newest first
    items.sort(key=lambda x: x.get('prediction_timestamp', ''), reverse=True)
    return items


def _run_redshift_query(sql: str) -> list:
    """Execute SQL via Redshift Data API (boto3, no psycopg2 needed)."""
    resp = redshift_data.execute_statement(
        ClusterIdentifier=REDSHIFT_CLUSTER,
        Database=REDSHIFT_DB,
        DbUser=REDSHIFT_USER,
        Sql=sql,
    )
    stmt_id = resp['Id']
    # Poll until done
    for _ in range(30):
        time.sleep(1)
        status = redshift_data.describe_statement(Id=stmt_id)
        if status['Status'] == 'FINISHED':
            if status.get('HasResultSet'):
                result = redshift_data.get_statement_result(Id=stmt_id)
                return result.get('Records', [])
            return []
        if status['Status'] in ('FAILED', 'ABORTED'):
            raise RuntimeError(f"Redshift query failed: {status.get('Error','')}")
    raise TimeoutError("Redshift query timed out")


def query_redshift_analytics() -> dict:
    """
    Query Redshift for aggregated statistics using Redshift Data API.
    Non-critical — returns empty dict on failure.
    """
    try:
        analytics = {}

        # Risk level distribution
        rows = _run_redshift_query("""
            SELECT risk_level,
                   COUNT(*) AS total,
                   ROUND(AVG(risk_score)::numeric, 4) AS avg_score,
                   ROUND(MIN(risk_score)::numeric, 4) AS min_score,
                   ROUND(MAX(risk_score)::numeric, 4) AS max_score
            FROM healthpredict.prediction_log
            GROUP BY risk_level
            ORDER BY risk_level
        """)
        analytics['risk_distribution'] = [
            {
                'risk_level': r[0].get('stringValue',''),
                'total':      int(r[1].get('longValue', r[1].get('stringValue',0))),
                'avg_score':  float(r[2].get('stringValue', 0)),
                'min_score':  float(r[3].get('stringValue', 0)),
                'max_score':  float(r[4].get('stringValue', 0)),
            }
            for r in rows
        ]

        # 7-day daily trend
        rows = _run_redshift_query("""
            SELECT DATE(pred_timestamp) AS day,
                   COUNT(*) AS predictions,
                   SUM(CASE WHEN risk_level='HIGH' THEN 1 ELSE 0 END) AS high_risk,
                   ROUND(AVG(risk_score)::numeric, 4) AS avg_score
            FROM healthpredict.prediction_log
            WHERE pred_timestamp >= DATEADD(day, -7, CURRENT_DATE)
            GROUP BY DATE(pred_timestamp)
            ORDER BY day DESC
        """)
        analytics['daily_trend_7d'] = [
            {
                'day':         r[0].get('stringValue',''),
                'predictions': int(r[1].get('longValue', r[1].get('stringValue',0))),
                'high_risk':   int(r[2].get('longValue', r[2].get('stringValue',0))),
                'avg_score':   float(r[3].get('stringValue', 0)),
            }
            for r in rows
        ]

        # Overall stats
        rows = _run_redshift_query("""
            SELECT COUNT(*) AS total_predictions,
                   ROUND(AVG(risk_score)::numeric, 4) AS avg_risk_score,
                   ROUND(AVG(glucose)::numeric, 1)    AS avg_glucose,
                   ROUND(AVG(bmi)::numeric, 1)        AS avg_bmi,
                   ROUND(AVG(age)::numeric, 1)        AS avg_age
            FROM healthpredict.prediction_log
        """)
        if rows:
            r = rows[0]
            analytics['overall_stats'] = {
                'total_predictions': int(r[0].get('longValue', r[0].get('stringValue',0))),
                'avg_risk_score':    float(r[1].get('stringValue', 0)),
                'avg_glucose':       float(r[2].get('stringValue', 0)),
                'avg_bmi':           float(r[3].get('stringValue', 0)),
                'avg_age':           float(r[4].get('stringValue', 0)),
            }

        logger.info("Redshift analytics via Data API: OK")
        return analytics

    except Exception as e:
        logger.error(f"Redshift analytics failed (non-critical): {e}")
        return {}


def lambda_handler(event, context):
    path_params  = event.get('pathParameters') or {}
    query_params = event.get('queryStringParameters') or {}

    patient_id   = path_params.get('patient_id')
    analytics    = query_params.get('analytics', 'false').lower() == 'true'
    risk_filter  = query_params.get('risk_level')
    limit        = min(int(query_params.get('limit', 20)), 100)

    try:
        if patient_id:
            # GET /predict/{patient_id}
            items = query_dynamodb_by_patient(patient_id, limit)
            response_body = {
                'patient_id': patient_id,
                'count':      len(items),
                'predictions': items,
            }
        else:
            # GET /history
            items = scan_dynamodb(risk_filter, limit)
            response_body = {
                'count':       len(items),
                'predictions': items,
            }

        # Optionally enrich with Redshift analytics
        if analytics:
            response_body['analytics'] = query_redshift_analytics()

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,x-api-key,X-Amz-Date,Authorization',
                'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
            },
            'body': json.dumps(response_body, cls=DecimalEncoder)
        }

    except Exception as e:
        logger.error(f"History query error: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': str(e)})
        }
