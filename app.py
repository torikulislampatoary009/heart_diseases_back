import logging
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

import config
import db
from predictor import Predictor, ValidationError, validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": config.CORS_ORIGINS}})

predictor, model_error = None, None


def load_model():
    global predictor, model_error
    if not os.path.exists(config.MODEL_PATH):
        model_error = f"Model file not found: {config.MODEL_PATH}. Put heart_model.joblib in backend/models/."
        log.error(model_error)
        return
    try:
        predictor = Predictor(config.MODEL_PATH, config.METRICS_PATH)
        log.info("Model loaded (trained with scikit-learn %s)", predictor.trained_with)
    except Exception as exc:  # noqa: BLE001
        model_error = f"Could not load model: {exc}"
        log.exception(model_error)


load_model()
db.init_db()


@app.get("/")
def index():
    return jsonify(service="Heart Disease Diagnostic Tool API", health="/api/health")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model_loaded=predictor is not None, model_error=model_error,
                   db_configured=db.is_configured(),
                   db_connected=db.ping() if db.is_configured() else False)


@app.get("/api/model-info")
def model_info():
    if predictor is None:
        return jsonify(error=model_error or "Model not loaded"), 503
    return jsonify(target_recall=predictor.target_recall, rows=predictor.rows,
                   metrics=predictor.metrics,
                   rule="A patient is flagged if either model is at or above its safety threshold.")


@app.post("/api/predict")
def predict():
    if predictor is None:
        return jsonify(error=model_error or "Model not loaded"), 503
    payload = request.get_json(silent=True) or {}
    try:
        values = validate(payload)
    except ValidationError as exc:
        return jsonify(error=str(exc)), 400

    result = predictor.predict(values)
    saved, record_id, save_error = False, None, None
    if payload.get("save", True) not in (False, "false", "0", 0):
        try:
            record_id = save_prediction(values, result)
            saved = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save prediction: %s", exc)
            save_error = "Result could not be saved to the database."
    return jsonify(inputs=values, saved=saved, record_id=record_id, save_error=save_error, **result)


def save_prediction(v, result):
    m = result["models"]
    with db.cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO predictions (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, "
            "slope, ca, thal, prob_logreg, prob_svm, flagged) VALUES "
            "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (v["age"], v["sex"], v["cp"], v["trestbps"], v["chol"], v["fbs"], v["restecg"], v["thalach"],
             v["exang"], v["oldpeak"], v["slope"], v["ca"], v["thal"],
             m["logistic_regression"]["probability"], m["svm"]["probability"], int(result["flagged"])))
        return cur.lastrowid


def _db_error(exc):
    log.warning("DB error: %s", exc)
    return jsonify(error="Database is unavailable. Check the DB_* environment variables."), 503


@app.get("/api/history")
def history():
    try:
        page = max(int(request.args.get("page", 1)), 1)
        limit = min(max(int(request.args.get("limit", 10)), 1), 50)
    except ValueError:
        return jsonify(error="page and limit must be numbers"), 400
    try:
        with db.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM predictions")
            total = cur.fetchone()["n"]
            cur.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT %s OFFSET %s",
                        (limit, (page - 1) * limit))
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        return _db_error(exc)
    for r in rows:
        r["created_at"] = r["created_at"].isoformat() + "Z"
    return jsonify(items=rows, total=total, page=page, limit=limit)


@app.delete("/api/history/<int:rid>")
def history_delete(rid):
    try:
        with db.cursor(commit=True) as cur:
            cur.execute("DELETE FROM predictions WHERE id=%s", (rid,))
            deleted = cur.rowcount
    except Exception as exc:  # noqa: BLE001
        return _db_error(exc)
    return (jsonify(deleted=True), 200) if deleted else (jsonify(error="Not found"), 404)


@app.get("/api/stats")
def stats():
    try:
        with db.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total, COALESCE(SUM(flagged),0) AS flagged FROM predictions")
            row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        return _db_error(exc)
    return jsonify(total=int(row["total"]), flagged=int(row["flagged"]))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
