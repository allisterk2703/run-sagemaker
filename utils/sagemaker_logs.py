import logging

import boto3

log = logging.getLogger("airflow.task")

LOG_GROUPS = {
    "training": "/aws/sagemaker/TrainingJobs",
    "transform": "/aws/sagemaker/TransformJobs",
}


def stream_cw_logs(cw, log_group: str, log_stream: str, next_token: str | None) -> str | None:
    """
    Fetch new CloudWatch log events for a SageMaker job and print them to the Airflow task log.
    Returns the updated forward token to pass on the next call.

    The log stream may not exist yet right after job submission — ResourceNotFoundException
    is silently ignored so the caller can retry on the next polling iteration.
    """
    kwargs = {
        "logGroupName": log_group,
        "logStreamName": log_stream,
        "startFromHead": True,
    }
    if next_token:
        kwargs["nextToken"] = next_token

    try:
        resp = cw.get_log_events(**kwargs)
    except cw.exceptions.ResourceNotFoundException:
        return next_token

    for event in resp.get("events", []):
        log.info("[SageMaker] %s", event["message"].rstrip())

    new_token = resp.get("nextForwardToken")
    return new_token if new_token != next_token else next_token


def make_cw_client(region: str):
    return boto3.client("logs", region_name=region)
