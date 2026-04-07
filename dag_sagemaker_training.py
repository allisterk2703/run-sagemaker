from datetime import datetime

import boto3
import time
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
POLL_INTERVAL = 30


def run_sagemaker_training(**context):
    project = context["params"]["project_name"]
    bucket = context["params"]["bucket_name"]
    ecr_repo = context["params"]["ecr_repo"]
    image_tag = context["params"].get("image_tag", "latest")
    instance_type = context["params"]["instance_type"]
    volume_size = context["params"]["volume_size_gb"]

    image_uri = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ecr_repo}:{image_tag}"
    role_arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/SageMakerExecutionRole"

    timestamp = datetime.now(tz=timezone("UTC")).strftime("%Y%m%d-%H%M")
    job_name = f"{project}-train-{timestamp}"
    sm = boto3.client("sagemaker", region_name=AWS_REGION)

    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        RoleArn=role_arn,
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/training/input/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/training/output/"},
        ResourceConfig={
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": volume_size,
        },
        StoppingCondition={"MaxRuntimeInSeconds": MAX_WAIT_SECONDS},
        Environment={
            "SM_CHANNEL_TRAINING": "/opt/ml/input/data/training",
            "SM_MODEL_DIR": "/opt/ml/model",
            "SM_OUTPUT_DIR": "/opt/ml/output",
        },
    )

    cw = make_cw_client(AWS_REGION)
    log_group = LOG_GROUPS["training"]
    log_stream = f"{job_name}/algo-1"
    cw_token = None

    waited = 0
    while waited < MAX_WAIT_SECONDS:
        response = sm.describe_training_job(TrainingJobName=job_name)
        status = response["TrainingJobStatus"]
        cw_token = stream_cw_logs(cw, log_group, log_stream, cw_token)
        if status == "Completed":
            stream_cw_logs(cw, log_group, log_stream, cw_token)
            return job_name
        if status in ("Failed", "Stopped"):
            stream_cw_logs(cw, log_group, log_stream, cw_token)
            reason = response.get("FailureReason", "unknown")
            raise RuntimeError(f"SageMaker job {job_name} ended with status {status}: {reason}")
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL

    raise TimeoutError(f"SageMaker job {job_name} did not complete within {MAX_WAIT_SECONDS}s")


DAG_DOC_MD = """
## SageMaker Training DAG

Runs a SageMaker Training job.

### Required Airflow Variables
- `AWS_REGION`: AWS region
- `AWS_ACCOUNT_ID`: AWS account ID

### Parameters
- **project_name** : Identifier for the project; used as a prefix for naming SageMaker Training jobs (https://eu-west-1.console.aws.amazon.com/sagemaker/home?region=eu-west-1#/jobs)
- **bucket_name** : Target S3 bucket for storing input data, source code, and model artifacts (https://eu-west-1.console.aws.amazon.com/s3/buckets?region=eu-west-1)
- **ecr_repo** : Name of the Amazon ECR repository hosting the training Docker image (https://eu-west-1.console.aws.amazon.com/ecr/private-registry/repositories?region=eu-west-1)
- **image_tag** : Specific tag or version of the training Docker image to pull from ECR
- **instance_type** : EC2 instance type utilized for the training compute node (https://aws.amazon.com/sagemaker/ai/pricing/)
- **volume_size_gb** : EBS storage capacity (in GB) attached to the instance for local data processing

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
    "on_failure_callback": notify_task_failure,
}

dag = DAG(
    dag_id="dag_sagemaker_training",
    description="Run SageMaker Training",
    default_args=default_args,
    start_date=datetime(2025, 6, 1, tzinfo=timezone("Europe/Paris")),
    schedule_interval=None,
    catchup=False,
    max_active_tasks=1,
    tags=["sagemaker", "training"],
    doc_md=DAG_DOC_MD,
    params={
        "project_name": Param(
            "titanic", 
            type="string", 
            description="Identifier for the project; used as a prefix for naming SageMaker Training jobs (🔗 https://eu-west-1.console.aws.amazon.com/sagemaker/home?region=eu-west-1#/jobs)"
        ),
        "bucket_name": Param(
            "titanic-<aws_region>-<aws_account_id>", 
            type="string", 
            description="Target S3 bucket for storing input data, source code, and model artifacts (🔗 https://eu-west-1.console.aws.amazon.com/s3/buckets?region=eu-west-1)"
        ),
        "ecr_repo": Param(
            "titanic-training", 
            type="string", 
            description="Name of the Amazon ECR repository hosting the training Docker image (🔗 https://eu-west-1.console.aws.amazon.com/ecr/private-registry/repositories?region=eu-west-1)"
        ),
        "image_tag": Param(
            "latest", 
            type="string", 
            description="Specific tag or version of the training Docker image to pull from ECR"
        ),
        "instance_type": Param(
            "ml.m5.large", 
            type="string", 
            description="EC2 instance type utilized for the training compute node (🔗 https://aws.amazon.com/sagemaker/ai/pricing/)"
        ),
        "volume_size_gb": Param(
            1,
            type="integer",
            minimum=1,
            maximum=30,
            description="EBS storage capacity (in GB) attached to the instance for local data processing",
        ),
    },
)

run_training = PythonOperator(
    task_id="run_sagemaker_training",
    python_callable=run_sagemaker_training,
    dag=dag,
)

run_training