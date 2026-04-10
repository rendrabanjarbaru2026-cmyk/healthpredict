"""
healthpredict_etl.py
AWS Glue ETL Job — HealthPredict AI
PySpark data pipeline: Raw CSV → Clean → Normalize → S3 Parquet + Redshift

AWS Academy Learner Lab Notes:
  - Uses LabRole for all AWS SDK calls
  - Retrieves Redshift password from Secrets Manager (no hardcoded credentials)
  - Compatible with Glue 4.0 (Spark 3.3, Python 3.10)

Job Parameters Required:
  --SOURCE_BUCKET       S3 bucket containing raw CSV
  --SOURCE_KEY          S3 key of the CSV file (e.g., raw/diabetes.csv)
  --DEST_BUCKET         S3 bucket for processed output (usually same as SOURCE_BUCKET)
  --SECRETS_ARN         ARN of Secrets Manager secret for Redshift credentials
  --GLUE_DATABASE       Glue Data Catalog database name
  --TEMP_DIR            S3 path for Glue Spark temp files
"""

import sys
import json
import boto3
import logging
from datetime import datetime

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ── Job Parameters ───────────────────────────────────────────
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'SOURCE_BUCKET',
    'SOURCE_KEY',
    'DEST_BUCKET',
    'SECRETS_ARN',
    'GLUE_DATABASE',
    'TEMP_DIR',
])

# ── Glue / Spark Context ──────────────────────────────────────
sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info("=" * 60)
logger.info("HEALTHPREDICT AI — GLUE ETL JOB STARTED")
logger.info(f"  Source: s3://{args['SOURCE_BUCKET']}/{args['SOURCE_KEY']}")
logger.info(f"  Dest:   s3://{args['DEST_BUCKET']}/processed/")
logger.info("=" * 60)


# ════════════════════════════════════════════════════════════
# HELPER: Retrieve Redshift credentials from Secrets Manager
# ════════════════════════════════════════════════════════════

def get_redshift_credentials(secret_arn: str) -> dict:
    """
    Retrieve Redshift credentials from AWS Secrets Manager.
    AWS Academy: LabRole has secretsmanager:GetSecretValue permission.
    """
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_arn)
    return json.loads(response['SecretString'])


# ════════════════════════════════════════════════════════════
# STEP 1 — READ RAW DATA FROM S3
# ════════════════════════════════════════════════════════════

logger.info("STEP 1: Reading raw CSV from S3...")

raw_dyf = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={
        "paths": [f"s3://{args['SOURCE_BUCKET']}/{args['SOURCE_KEY']}"],
        "recurse": True,
    },
    format="csv",
    format_options={
        "withHeader": True,
        "separator": ",",
        "optimizePerformance": False,
    }
)

df = raw_dyf.toDF()

# Normalize column names to lowercase
column_renames = {
    'Pregnancies':              'pregnancies',
    'Glucose':                  'glucose',
    'BloodPressure':            'blood_pressure',
    'SkinThickness':            'skin_thickness',
    'Insulin':                  'insulin',
    'BMI':                      'bmi',
    'DiabetesPedigreeFunction': 'diabetes_pedigree',
    'Age':                      'age',
    'Outcome':                  'outcome',
}
for old, new in column_renames.items():
    if old in df.columns:
        df = df.withColumnRenamed(old, new)

# Cast to correct types
for col in ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'age', 'outcome']:
    df = df.withColumn(col, df[col].cast(IntegerType()))
for col in ['bmi', 'diabetes_pedigree']:
    df = df.withColumn(col, df[col].cast(DoubleType()))

total_raw = df.count()
logger.info(f"  Loaded {total_raw} records. Schema:")
df.printSchema()


# ════════════════════════════════════════════════════════════
# STEP 2 — DATA QUALITY AUDIT
# ════════════════════════════════════════════════════════════

logger.info("STEP 2: Data quality audit...")

medical_cols = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
for col in medical_cols:
    n_null  = df.filter(F.col(col).isNull()).count()
    n_zero  = df.filter(F.col(col) == 0).count()
    logger.info(f"  {col:25s}: nulls={n_null}, zeros={n_zero}")


# ════════════════════════════════════════════════════════════
# STEP 3 — DATA CLEANING
# ════════════════════════════════════════════════════════════

logger.info("STEP 3: Replacing invalid zeros with column medians...")

for col in medical_cols:
    median_val = (
        df.filter(F.col(col) > 0)
          .approxQuantile(col, [0.5], 0.001)[0]
    )
    logger.info(f"  {col:25s}: median = {median_val:.3f}")
    df = df.withColumn(
        col,
        F.when(F.col(col) == 0, median_val).otherwise(F.col(col))
    )

df = df.dropna()
df = df.dropDuplicates()
logger.info(f"  Records after cleaning: {df.count()}")


# ════════════════════════════════════════════════════════════
# STEP 4 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════

logger.info("STEP 4: Feature engineering...")

# BMI category (WHO standard)
df = df.withColumn(
    'bmi_category',
    F.when(F.col('bmi') < 18.5,  'Underweight')
     .when(F.col('bmi') < 25.0,  'Normal')
     .when(F.col('bmi') < 30.0,  'Overweight')
     .otherwise('Obese')
)

# Age group
df = df.withColumn(
    'age_group',
    F.when(F.col('age') < 30, 'Young')
     .when(F.col('age') < 45, 'Middle')
     .when(F.col('age') < 60, 'Senior')
     .otherwise('Elderly')
)

# Glucose risk classification (ADA thresholds)
df = df.withColumn(
    'glucose_risk',
    F.when(F.col('glucose') < 100, 'Normal')
     .when(F.col('glucose') < 126, 'Prediabetes')
     .otherwise('Diabetes_Range')
)

# Interaction feature
df = df.withColumn(
    'glucose_bmi_interaction',
    (F.col('glucose') * F.col('bmi') / 1000.0)
)

logger.info("  Feature engineering complete. Sample:")
df.select('bmi_category', 'age_group', 'glucose_risk',
          'glucose_bmi_interaction', 'outcome').show(3, truncate=False)


# ════════════════════════════════════════════════════════════
# STEP 5 — NORMALIZATION (StandardScaler)
# ════════════════════════════════════════════════════════════

logger.info("STEP 5: Applying StandardScaler normalization...")

numerical_features = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'diabetes_pedigree', 'age'
]

# Compute mean + std in a single pass
stats_row = df.select(
    [F.mean(c).alias(f'{c}_mean') for c in numerical_features] +
    [F.stddev(c).alias(f'{c}_std')  for c in numerical_features]
).collect()[0]

# Store normalization stats as JSON in S3 for Lambda to use at inference time
norm_stats = {}
for col in numerical_features:
    mean_val = stats_row[f'{col}_mean']
    std_val  = stats_row[f'{col}_std'] or 1.0
    norm_stats[col] = {'mean': round(mean_val, 6), 'std': round(std_val, 6)}
    logger.info(f"  {col:25s}: mean={mean_val:.4f}, std={std_val:.4f}")
    df = df.withColumn(
        f'{col}_scaled',
        (F.col(col) - mean_val) / std_val
    )

# Save normalization stats to S3 (used by Lambda for inference normalization)
s3 = boto3.client('s3', region_name='us-east-1')
s3.put_object(
    Bucket=args['DEST_BUCKET'],
    Key='config/normalization_stats.json',
    Body=json.dumps(norm_stats, indent=2).encode('utf-8'),
    ContentType='application/json'
)
logger.info("  Normalization stats saved to s3://DEST_BUCKET/config/normalization_stats.json")


# ════════════════════════════════════════════════════════════
# STEP 6 — TRAIN / VALIDATION SPLIT (80 / 20)
# ════════════════════════════════════════════════════════════

logger.info("STEP 6: Splitting dataset 80/20...")

scaled_cols  = [f'{c}_scaled' for c in numerical_features]
training_df  = df.select(scaled_cols + ['outcome'])

train_df, val_df = training_df.randomSplit([0.8, 0.2], seed=42)
logger.info(f"  Training:   {train_df.count()} records")
logger.info(f"  Validation: {val_df.count()} records")


# ════════════════════════════════════════════════════════════
# STEP 7 — WRITE PROCESSED PARQUET TO S3
# ════════════════════════════════════════════════════════════

logger.info("STEP 7: Writing processed Parquet to S3...")

processed_dyf = DynamicFrame.fromDF(df, glueContext, "processed")
glueContext.write_dynamic_frame.from_options(
    frame=processed_dyf,
    connection_type="s3",
    connection_options={
        "path": f"s3://{args['DEST_BUCKET']}/processed/",
        "partitionKeys": [],
    },
    format="parquet"
)
logger.info(f"  Parquet saved: s3://{args['DEST_BUCKET']}/processed/")


# ════════════════════════════════════════════════════════════
# STEP 8 — WRITE TRAIN / VALIDATION CSV FOR SAGEMAKER
# ════════════════════════════════════════════════════════════

logger.info("STEP 8: Writing SageMaker-format CSV files...")

# XGBoost built-in requires: no header, label in first column
train_pd = train_df.toPandas()
val_pd   = val_df.toPandas()

# Ensure outcome (label) is first column
cols_ordered = ['outcome'] + [c for c in train_pd.columns if c != 'outcome']
train_pd = train_pd[cols_ordered]
val_pd   = val_pd[cols_ordered]

s3.put_object(
    Bucket=args['DEST_BUCKET'],
    Key='train/train.csv',
    Body=train_pd.to_csv(index=False, header=False).encode('utf-8'),
)
logger.info(f"  Train CSV:      s3://{args['DEST_BUCKET']}/train/train.csv")

s3.put_object(
    Bucket=args['DEST_BUCKET'],
    Key='validation/validation.csv',
    Body=val_pd.to_csv(index=False, header=False).encode('utf-8'),
)
logger.info(f"  Validation CSV: s3://{args['DEST_BUCKET']}/validation/validation.csv")


# ════════════════════════════════════════════════════════════
# STEP 9 — LOAD TO AMAZON REDSHIFT
# ════════════════════════════════════════════════════════════

logger.info("STEP 9: Loading processed data to Amazon Redshift...")

try:
    creds = get_redshift_credentials(args['SECRETS_ARN'])

    jdbc_url = (
        f"jdbc:redshift://{creds['host']}:{creds['port']}/{creds['dbname']}"
        f"?user={creds['username']}&password={creds['password']}"
    )

    # Write to Redshift via Glue JDBC
    glueContext.write_dynamic_frame.from_options(
        frame=DynamicFrame.fromDF(df, glueContext, "redshift_frame"),
        connection_type="jdbc",
        connection_options={
            "url":          jdbc_url,
            "dbtable":      "healthpredict.patient_data_processed",
            "redshiftTmpDir": args['TEMP_DIR'],
        }
    )
    logger.info("  Data loaded to healthpredict.patient_data_processed in Redshift")

except Exception as e:
    logger.error(f"  Redshift load failed (non-critical): {e}")
    logger.warning("  Continuing — Parquet and CSV outputs are already saved.")


# ════════════════════════════════════════════════════════════
# STEP 9b — UPDATE GLUE DATA CATALOG STATISTICS
# ════════════════════════════════════════════════════════════

logger.info("STEP 9b: Updating Glue Data Catalog statistics...")

try:
    glue_client = boto3.client('glue', region_name='us-east-1')
    glue_client.update_table(
        DatabaseName=args['GLUE_DATABASE'],
        TableInput={
            'Name': 'processed_patient_data',
            'Parameters': {
                'recordCount':      str(df.count()),
                'averageRecordSize': '512',
                'last_etl_run':     datetime.utcnow().isoformat(),
            }
        }
    )
    logger.info("  Glue Data Catalog updated")
except Exception as e:
    logger.warning(f"  Catalog update skipped: {e}")


# ════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════

logger.info("=" * 60)
logger.info("ETL JOB COMPLETED SUCCESSFULLY")
logger.info(f"  Raw records:        {total_raw}")
logger.info(f"  Processed records:  {df.count()}")
logger.info(f"  Training set:       {train_df.count()}")
logger.info(f"  Validation set:     {val_df.count()}")
logger.info(f"  Norm stats:         s3://{args['DEST_BUCKET']}/config/normalization_stats.json")
logger.info("=" * 60)

job.commit()
