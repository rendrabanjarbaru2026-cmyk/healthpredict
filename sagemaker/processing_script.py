"""
processing_script.py
SageMaker Processing Step - HealthPredict AI
Reads train/validation CSV from Glue ETL output,
validates format, and writes to output channels for XGBoost.

Input paths (from pipeline definition):
  /opt/ml/processing/input/train/       <- train CSV from Glue
  /opt/ml/processing/input/validation/  <- validation CSV from Glue
Output paths:
  /opt/ml/processing/output/train/      -> to XGBoost train channel
  /opt/ml/processing/output/validation/ -> to XGBoost validation channel
"""

import os
import glob
import shutil
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

INPUT_TRAIN  = "/opt/ml/processing/input/train"
INPUT_VAL    = "/opt/ml/processing/input/validation"
OUTPUT_TRAIN = "/opt/ml/processing/output/train"
OUTPUT_VAL   = "/opt/ml/processing/output/validation"

os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_VAL,   exist_ok=True)


def process_split(input_dir, output_dir, split_name):
    """Copy and validate CSV files from input to output."""
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files in {input_dir}. "
            f"Contents: {os.listdir(input_dir) if os.path.exists(input_dir) else 'dir not found'}"
        )

    total_rows = 0
    for src in csv_files:
        fname = os.path.basename(src)
        dst   = os.path.join(output_dir, fname)
        shutil.copy2(src, dst)

        # Count rows and validate
        with open(src) as f:
            rows = f.readlines()

        # Detect and skip header if present
        first = rows[0].strip().split(",")
        has_header = not all(
            v.replace(".", "").replace("-", "").isdigit()
            for v in first[:3]
        )
        data_rows = rows[1:] if has_header else rows
        n_rows    = len([r for r in data_rows if r.strip()])
        n_cols    = len(rows[0].strip().split(","))
        total_rows += n_rows

        logger.info(f"  {split_name}/{fname}: {n_rows} rows x {n_cols} cols")
        if n_cols < 9:
            raise ValueError(
                f"{split_name}/{fname}: expected >=9 columns, got {n_cols}. "
                "Check Glue ETL output format."
            )

    logger.info(f"  {split_name}: total {total_rows} records written to {output_dir}")
    return total_rows


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("SAGEMAKER PROCESSING STEP — HEALTHPREDICT AI")
    logger.info("=" * 50)
    logger.info(f"Input train dir:      {os.listdir(INPUT_TRAIN) if os.path.exists(INPUT_TRAIN) else 'MISSING'}")
    logger.info(f"Input validation dir: {os.listdir(INPUT_VAL) if os.path.exists(INPUT_VAL) else 'MISSING'}")

    n_train = process_split(INPUT_TRAIN, OUTPUT_TRAIN, "train")
    n_val   = process_split(INPUT_VAL,   OUTPUT_VAL,   "validation")

    logger.info("=" * 50)
    logger.info(f"Processing complete: {n_train} train, {n_val} validation records")
    logger.info("=" * 50)
