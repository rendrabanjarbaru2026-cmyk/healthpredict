# HealthPredict AI — Student Code Package
# LKS Cloud Computing - Tingkat Provinsi 2026

Code package for the HealthPredict AI competition.
Please read the module carefully before using these files.

---

## File Structure

```
healthpredict-student/
├── glue/
│   └── healthpredict_etl.py        PySpark ETL — 9 Steps for Data Transformation
├── sagemaker/
│   ├── sagemaker_pipeline.py       Register a 3-step SageMaker Pipeline
│   ├── processing_script.py        Script Processing Step (SKLearnProcessor)
│   ├── deploy_endpoint.py          Deploy model Approved to real-time endpoint
│   └── batch_transform.py          Submit Batch Transform job to S3
├── lambda/
│   ├── predict.py                  POST /predict — validasi, normalize, invoke, save
│   ├── history.py                  GET /history and GET /predict/{patient_id}
│   └── health.py                   GET /health — check status endpoint and DynamoDB
├── athena/
│   └── athena_queries.sql          5 dataset exploration queries
├── scripts/
│   └── redshift_schema.sql         DDL: 4 tabel, 3 view, 1 stored procedure
└── frontend/
    └── index.html                  Single-page web app (deploy to AWS Amplify)
```

---
---

## Required Configuration

### sagemaker_pipeline.py and deploy_endpoint.py and batch_transform.py
Replace the following values before running:
```python
STUDENT_NAME = "yourname"   # ← Replace with your name (lowercase, no spaces)
```

### Lambda Environment Variables

| Variable | Value |
|---|---|
| SAGEMAKER_ENDPOINT_NAME | healthpredict-endpoint-diabetes |
| DYNAMODB_TABLE_NAME | healthpredict-predictions |
| SNS_TOPIC_ARN | Your ARN SNS topic |
| REDSHIFT_SECRET_ARN | Your ARN Secrets Manager secret |
| RISK_THRESHOLD_HIGH | 0.7 |
| RISK_THRESHOLD_LOW | 0.3 |
