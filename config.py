import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


MODEL_PATH = _abs(os.getenv("MODEL_PATH", "models/heart_model.joblib"))
METRICS_PATH = _abs(os.getenv("METRICS_PATH", "models/metrics.json"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
