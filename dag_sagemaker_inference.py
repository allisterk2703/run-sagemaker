from datetime import datetime

import boto3
import time
from botocore.exceptions import ClientError
from callbacks.notify import notify_task_failure
from utils.sagemaker_logs import LOG_GROUPS, make_cw_client, stream_cw_logs

from airflow import DAG
from airflow.models import Variable
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from pendulum import timezone


AWS_REGION = Variable.get("AWS_REGION")
AWS_ACCOUNT_ID = Variable.get("AWS_ACCOUNT_ID")

MAX_WAIT_SECONDS = 3600
POLL_INTERVAL = 10


def _get_or_create_model(sm, image_uri: str, role_arn: str, bucket: str, project: str) -> str:
    """
    Recreates a SageMaker Model pointing to the inference image URI in ECR and to
    current_model/model.tar.gz in S3, then returns its name.
    """
    model_name   = f"{project}-current-model"
    model_s3_uri = f"s3://{bucket}/current_model/model.tar.gz"

    try:
        sm.describe_model(ModelName=model_name)
        sm.delete_model(ModelName=model_name)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        message = e.response["Error"].get("Message", "")
        if code != "ValidationException" or "Could not find model" not in message:
            raise

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_s3_uri,
            "Environment": {
                "SM_MODEL_DIR": "/opt/ml/model",
            },
        },
        ExecutionRoleArn=role_arn,
    )

    return model_name


def run_sagemaker_inference(**context):
    project       = context["params"]["project_name"]
    bucket        = context["params"]["bucket_name"]
    ecr_repo      = context["params"]["ecr_repo"]
    image_tag     = context["params"]["image_tag"]
    instance_type = context["params"]["instance_type"]

    image_uri = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ecr_repo}:{image_tag}"
    role_arn  = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/SageMakerExecutionRole"

    timestamp  = datetime.now(tz=timezone("UTC")).strftime("%Y%m%d-%H%M")
    job_name   = f"{project}-batch-transform-{timestamp}"
    sm         = boto3.client("sagemaker", region_name=AWS_REGION)
    model_name = _get_or_create_model(sm, image_uri, role_arn, bucket, project)

    sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        MaxConcurrentTransforms=1,
        MaxPayloadInMB=6,
        TransformInput={
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{bucket}/inference/input/",
                }
            },
            "ContentType": "application/jsonlines",
            "SplitType": "Line",
        },
        TransformOutput={
            "S3OutputPath": f"s3://{bucket}/inference/output/",
            "AssembleWith": "Line",
        },
        TransformResources={
            "InstanceType": instance_type,
            "InstanceCount": 1,
        },
    )

    cw = make_cw_client(AWS_REGION)
    log_group = LOG_GROUPS["transform"]
    log_stream = f"{job_name}/algo-1"
    cw_token = None

    waited = 0
    while waited < MAX_WAIT_SECONDS:
        response = sm.describe_transform_job(TransformJobName=job_name)
        status = response["TransformJobStatus"]
        cw_token = stream_cw_logs(cw, log_group, log_stream, cw_token)
        if status == "Completed":
            stream_cw_logs(cw, log_group, log_stream, cw_token)
            return job_name
        if status in ("Failed", "Stopped"):
            stream_cw_logs(cw, log_group, log_stream, cw_token)
            reason = response.get("FailureReason", "unknown")
            raise RuntimeError(f"Transform job {job_name} ended with status {status}: {reason}")

        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL

    raise TimeoutError(f"Transform job {job_name} did not complete within {MAX_WAIT_SECONDS}s")


DAG_DOC_MD = """
## SageMaker Inference DAG

Runs a SageMaker Batch Transform job.

### Required Airflow Variables
- `AWS_REGION`: AWS region
- `AWS_ACCOUNT_ID`: AWS account ID

### Parameters
- **project_name** : Identifier for the project; used as a prefix for naming SageMaker Batch Transform jobs (https://eu-west-1.console.aws.amazon.com/sagemaker/home?region=eu-west-1#/transform-jobs)
- **bucket_name** : Target S3 bucket for storing input data, source code, and model artifacts (https://eu-west-1.console.aws.amazon.com/s3/buckets?region=eu-west-1)
- **ecr_repo** : Name of the Amazon ECR repository hosting the inference Docker image (https://eu-west-1.console.aws.amazon.com/ecr/private-registry/repositories?region=eu-west-1)
- **image_tag** : Specific tag or version of the inference Docker image to pull from ECR
- **instance_type** : EC2 instance type utilized for the inference node (https://aws.amazon.com/sagemaker/ai/pricing/)

### Expected S3 Structure
```
<bucket_name>/
├── current_model/
│   └── model.tar.gz
├── inference/
│   ├── input/
│   │   └── data.jsonl
│   └── output/
└── training/
    ├── input/
    │   └── data.csv
    └── output/
```
"""

default_args = {
    "owner": "Allister",
    "depends_on_past": False,
    "retries": 0,
    "on_failure_callback": notify_task_failure
}

dag = DAG(
    dag_id="dag_sagemaker_inference",
    description="Run SageMaker Batch Transform",
    default_args=default_args,
    start_date=datetime(2025, 6, 1, tzinfo=timezone("Europe/Paris")),
    schedule_interval=None,
    catchup=False,
    max_active_tasks=1,
    tags=["sagemaker", "inference"],
    doc_md=DAG_DOC_MD,
    params={
        "project_name": Param("titanic", type="string", description="Identifier for the project; used as a prefix for naming SageMaker Batch Transform jobs (🔗 https://eu-west-1.console.aws.amazon.com/sagemaker/home?region=eu-west-1#/transform-jobs)"),
        "bucket_name": Param("titanic-<aws_region>-<aws_account_id>", type="string", description="Target S3 bucket for storing input data, source code, and model artifacts (🔗 https://eu-west-1.console.aws.amazon.com/s3/buckets?region=eu-west-1)"),
        "ecr_repo": Param("titanic-inference", type="string", description="Name of the Amazon ECR repository hosting the inference Docker image (🔗 https://eu-west-1.console.aws.amazon.com/ecr/private-registry/repositories?region=eu-west-1)"),
        "image_tag": Param("latest", type="string", description="Specific tag or version of the inference Docker image to pull from ECR"),
        "instance_type": Param("ml.m5.large", type="string", description="EC2 instance type utilized for the inference node (🔗 https://aws.amazon.com/sagemaker/ai/pricing/)"),
    },
)

run_inference = PythonOperator(
    task_id="run_sagemaker_inference",
    python_callable=run_sagemaker_inference,
    dag=dag,
)

run_inference