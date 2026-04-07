import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SECRETS_DIR = BASE_DIR / "secrets"


def get_training_input_dir() -> Path:
    return Path(os.environ.get("SM_CHANNEL_TRAINING") or SECRETS_DIR / "training/input")


def get_training_output_dir() -> Path:
    return Path(os.environ.get("SM_OUTPUT_DIR") or SECRETS_DIR / "training/output")


def get_model_dir() -> Path:
    return Path(os.environ.get("SM_MODEL_DIR") or SECRETS_DIR / "training/model")
