"""
health.py
Lambda Function — healthpredict-lambda-health
GET /health  →  check SageMaker endpoint + DynamoDB status
No API Key required. Returns 200 (healthy) or 503 (degraded).
"""

import json
import os
import logging
import time
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm_client = boto3.client('sagemaker')
dynamo    = boto3.client('dynamodb')

ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']
TABLE_NAME    = os.environ['DYNAMODB_TABLE_NAME']


def check_sagemaker() -> dict:
    start = time.time()
    try:
        resp   = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp['EndpointStatus']
        return {
            'service':  'SageMaker Endpoint',
            'name':     ENDPOINT_NAME,
            'status':   'healthy' if status == 'InService' else 'degraded',
            'detail':   status,
            'latency_ms': round((time.time() - start) * 1000, 1),
        }
    except Exception as e:
        return {
            'service': 'SageMaker Endpoint',
            'name':    ENDPOINT_NAME,
            'status':  'unhealthy',
            'detail':  str(e),
            'latency_ms': round((time.time() - start) * 1000, 1),
        }


def check_dynamodb() -> dict:
    start = time.time()
    try:
        resp   = dynamo.describe_table(TableName=TABLE_NAME)
        status = resp['Table']['TableStatus']
        return {
            'service': 'DynamoDB',
            'name':    TABLE_NAME,
            'status':  'healthy' if status == 'ACTIVE' else 'degraded',
            'detail':  status,
            'latency_ms': round((time.time() - start) * 1000, 1),
        }
    except Exception as e:
        return {
            'service': 'DynamoDB',
            'name':    TABLE_NAME,
            'status':  'unhealthy',
            'detail':  str(e),
            'latency_ms': round((time.time() - start) * 1000, 1),
        }


def lambda_handler(event, context):
    sm_check    = check_sagemaker()
    dynamo_check = check_dynamodb()

    checks      = [sm_check, dynamo_check]
    all_healthy = all(c['status'] == 'healthy' for c in checks)
    overall     = 'healthy' if all_healthy else 'degraded'

    body = {
        'status':    overall,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'checks':    {c['service']: c for c in checks},
        'version':   '1.0.0',
    }

    logger.info(f"Health check: {overall}")
    return {
        'statusCode': 200 if all_healthy else 503,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,x-api-key,X-Amz-Date,Authorization',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        },
        'body': json.dumps(body)
    }
