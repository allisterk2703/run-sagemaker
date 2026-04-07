from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from callbacks.notify import notify_task_failure

from airflow import DAG
from airflow.models import Variable
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from pendulum import timezone


AWS_REGION = Variable.get("AWS_REGION")
AWS_ACCOUNT_ID = Variable.get("AWS_ACCOUNT_ID")


def promote_model(**context):
    job_name = context["params"]["job_name"]
    project  = job_name.split("-train-")[0]
    bucket   = f"{project}-{AWS_REGION}-{AWS_ACCOUNT_ID}"

    src_key = f"training/output/{job_name}/output/model.tar.gz"
    dst_key = "current_model/model.tar.gz"

    s3 = boto3.client("s3", region_name=AWS_REGION)

    # Verify source exists before copying
    try:
        s3.head_object(Bucket=bucket, Key=src_key)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        raise FileNotFoundError(
            f"Source artifact not found at s3://{bucket}/{src_key} (error: {code})"
        ) from e

    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
    )

    print(f"Promoted s3://{bucket}/{src_key} → s3://{bucket}/{dst_key}")


DAG_DOC_MD = """
## Promote Model DAG

Copies a trained model artifact from `training/output/` to `current_model/`,
making it available for the SageMaker Inference DAG.

### Required Airflow Variables
- `AWS_REGION`: AWS region

### Parameters
- **job_name** : Name of the SageMaker Training job to promote — the bucket is dérivé automatiquement (`{project}-{AWS_REGION}-{AWS_ACCOUNT_ID}`) (https://eu-west-1.console.aws.amazon.com/sagemaker/home?region=eu-west-1#/jobs)

### S3 Copy
```
training/output/{job_name}/output/model.tar.gz  →  current_model/model.tar.gz
```

### Expected S3 Structure
```
<bucket_name>/
├── current_model/
│   └── model.tar.gz          ← overwritten by this DAG
├── inference/
│   ├── input/
│   │   └── data.jsonl
│   └── output/
└── training/
    ├── input/
    │   └── data.csv
    └── output/
        └── {job_name}/
            └── output/
                └── model.tar.gz
```
"""

default_args = {
    "owner": "Allister",
    "depends_on_past": False,
    "retries": 0,
    "on_failure_callback": notify_task_failure,
}

dag = DAG(
    dag_id="dag_promote_model",
    description="Promote a trained model artifact to current_model/",
    default_args=default_args,
    start_date=datetime(2025, 6, 1, tzinfo=timezone("Europe/Paris")),
    schedule_interval=None,
    catchup=False,
    max_active_tasks=1,
    tags=["sagemaker", "model"],
    doc_md=DAG_DOC_MD,
    params={
        "job_name": Param(
            "",
            type="string",
            description="Name of the SageMaker Training job to promote (🔗 https://eu-west-1.console.aws.amazon.com/sagemaker/home?region=eu-west-1#/jobs)",
        ),
    },
)

promote = PythonOperator(
    task_id="promote_model",
    python_callable=promote_model,
    dag=dag,
)

promote
