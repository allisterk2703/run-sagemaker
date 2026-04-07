import json
import logging
import joblib
import pandas as pd

from flask import Flask, request, Response
from waitress import serve
from config import get_model_dir
from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, engineer_features


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


app = Flask(__name__)

MODEL_DIR = get_model_dir()
model = None


def load_model():
    global model
    try:
        model_path = MODEL_DIR / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {type(e).__name__}: {e}")
        raise


@app.route("/ping", methods=["GET"])
def ping():
    logger.info("/ping: Health check called")
    return Response("OK", status=200)


@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        logger.info("/invocations: Request received")
        content_type = request.content_type or ""

        try:
            if "application/jsonlines" in content_type:
                lines = request.data.decode("utf-8").strip().splitlines()
                records = [json.loads(line) for line in lines if line.strip()]
                df = pd.DataFrame(records)

            elif "application/json" in content_type:
                body = json.loads(request.data.decode("utf-8"))
                if "instances" in body:
                    df = pd.DataFrame(body["instances"])
                else:
                    df = pd.DataFrame(body if isinstance(body, list) else [body])

            elif "text/plain" in content_type:
                # Parse CSV or raw text as JSON lines
                lines = request.data.decode("utf-8").strip().splitlines()
                records = []
                for line in lines:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipped invalid JSON line: {str(e)}")
                if not records:
                    raise ValueError("No valid JSON lines found in text/plain data")
                df = pd.DataFrame(records)

            else:
                logger.error(f"Unsupported content type: {content_type}")
                return Response(f"Unsupported content type: {content_type}. Use application/json, application/jsonlines, or text/plain.", status=415)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return Response(f"Invalid JSON: {e}", status=400)
        except Exception as e:
            logger.error(f"Data parsing failed: {type(e).__name__}: {e}")
            return Response(f"Data parsing error: {e}", status=400)

        logger.info(f"/invocations: Received {len(df)} rows")

        try:
            df = engineer_features(df)
            X  = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

            probabilities = model.predict_proba(X)

            prob_positive = [p[1] for p in probabilities]
            result = "\n".join(str(p) for p in prob_positive)

            logger.info(f"/invocations: Generated {len(prob_positive)} predictions")

            return Response(result, status=200, mimetype="text/csv")
        except Exception as e:
            logger.error(f"Inference failed: {type(e).__name__}: {e}")
            return Response(f"Inference error: {e}", status=500)

    except Exception as e:
        logger.error(f"Unexpected error in invocations: {type(e).__name__}: {e}")
        return Response(f"Internal server error: {e}", status=500)


if __name__ == "__main__":
    load_model()
    serve(app, host="0.0.0.0", port=8080)