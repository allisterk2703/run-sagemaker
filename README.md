# run-sagemaker

Airflow DAGs and Docker images to run SageMaker Training and Batch Transform jobs on AWS.

## Structure

```
├── dag_sagemaker_training.py   # Airflow DAG: SageMaker Training job
├── dag_sagemaker_inference.py  # Airflow DAG: SageMaker Batch Transform job
├── dag_promote_model.py        # Airflow DAG: Promote a training artifact to current_model/
└── test-docker-image/          # Docker images + local test tooling (Makefile)
    ├── Dockerfile.training     # Trains a RandomForest, saves model.joblib
    ├── Dockerfile.inference    # Flask server on port 8080 (/ping, /invocations)
    └── src/
        ├── train.py
        ├── predict.py
        └── config.py
```

## Prerequisites

- An IAM role named `SageMakerExecutionRole` with SageMaker, S3, and ECR permissions.
- S3 bucket structured as:

```
<bucket_name>/
├── current_model/model.tar.gz
├── training/input/data.csv
└── inference/input/data.jsonl
```

## Airflow variables

Set in `Admin → Variables`:

| Variable | Description |
|---|---|
| `AWS_REGION` | AWS region (e.g. `eu-west-1`) |
| `AWS_ACCOUNT_ID` | 12-digit AWS account ID |

## DAG parameters

All DAGs are triggered manually. The typical workflow is:
1. **Training** → produces `training/output/{job_name}/output/model.tar.gz`
2. **Promote** → copies the artifact to `current_model/model.tar.gz`
3. **Inference** → runs Batch Transform using `current_model/model.tar.gz`

### Training & Inference

| Parameter | Training default | Inference default | Description |
|---|---|---|---|
| `project_name` | `titanic` | `titanic` | Prefix for job names |
| `bucket_name` | `titanic-<region>-<account>` | `titanic-<region>-<account>` | S3 bucket |
| `ecr_repo` | `titanic-training` | `titanic-inference` | ECR repository |
| `image_tag` | `latest` | `latest` | Docker image tag |
| `instance_type` | `ml.m5.large` | `ml.m5.large` | SageMaker instance type |
| `volume_size_gb` | `1` | — | EBS volume size in GB (training only) |

### Promote Model

| Parameter | Description |
|---|---|
| `job_name` | Name of the training job to promote (e.g. `titanic-train-20260326-1050`) |

The bucket is derived automatically from the job name: `{project}-{AWS_REGION}-{AWS_ACCOUNT_ID}`.

## Local testing (`test-docker-image/`)

### `.env`

```dotenv
AWS_REGION=<your-aws-region>
AWS_ACCOUNT_ID=<your-account-id>
```

Used by the `Makefile` to build ECR URIs. Never commit this file.

### `secrets/` directory

Mounted into containers locally in place of S3:

```
secrets/
├── training/input/    # data.csv
├── training/output/   # metrics.json (written by container)
├── training/model/    # model.joblib (written by container)
└── inference/input/   # data.jsonl
```

### Key `make` commands

```bash
make build              # Build AMD64 images
make build-arm64        # Build ARM64 images
make run-training-arm64 # Run training container locally
make run-inference-arm64# Run inference container locally
make deploy             # ECR setup + build + push (AMD64)
make help               # List all commands
```
