import json
import logging
import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from config import get_model_dir, get_training_input_dir, get_training_output_dir
from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, engineer_features


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


INPUT_DIR = get_training_input_dir()
MODEL_DIR = get_model_dir()
OUTPUT_DIR = get_training_output_dir()

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(input_dir: Path) -> pd.DataFrame:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {input_dir}")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    logger.info(f"[load] {len(df)} rows loaded from {[f.name for f in csv_files]}")
    return df



numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,  NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])


def train(df: pd.DataFrame) -> dict:
    df = engineer_features(df)

    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    TARGET   = "Survived"

    X = df[FEATURES]
    y = df[TARGET]

    logger.info(f"[train] Features: {FEATURES}")
    logger.info(f"[train] X shape: {X.shape}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    logger.info(f"[train] CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X, y)

    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    logger.info(f"[train] Train accuracy: {report['accuracy']:.4f}")

    metrics = {
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std":  round(float(cv_scores.std()),  4),
        "train_accuracy":   round(float(report["accuracy"]), 4),
    }
    return metrics


def save_artifacts(metrics: dict) -> None:
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"[save] Model saved to {model_path}")

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[save] Metrics saved to {metrics_path}")
    logger.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    df      = load_data(INPUT_DIR)
    metrics = train(df)
    save_artifacts(metrics)
    logger.info("[done] Training completed.")