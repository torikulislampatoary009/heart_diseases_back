"""Loads the trained Logistic Regression + SVM and applies the safety-first rule."""
import json
import logging
import os

import joblib
import numpy as np

log = logging.getLogger("predictor")

# field: (min, max, is_integer, allowed values or None)
SPEC = {
    "age":      (1, 120, True, None),
    "sex":      (0, 1, True, {0, 1}),
    "cp":       (1, 4, True, {1, 2, 3, 4}),
    "trestbps": (60, 260, True, None),
    "chol":     (80, 700, True, None),
    "fbs":      (0, 1, True, {0, 1}),
    "restecg":  (0, 2, True, {0, 1, 2}),
    "thalach":  (50, 250, True, None),
    "exang":    (0, 1, True, {0, 1}),
    "oldpeak":  (0, 10, False, None),
    "slope":    (1, 3, True, {1, 2, 3}),
    "ca":       (0, 3, True, {0, 1, 2, 3}),
    "thal":     (3, 7, True, {3, 6, 7}),
}


class ValidationError(ValueError):
    pass


def validate(payload):
    """Return a clean dict of numbers, or raise ValidationError with a friendly message."""
    clean, problems = {}, []
    for name, (lo, hi, is_int, allowed) in SPEC.items():
        raw = payload.get(name)
        if raw is None or str(raw).strip() == "":
            problems.append(f"{name} is required")
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            problems.append(f"{name} must be a number")
            continue
        if is_int and val != int(val):
            problems.append(f"{name} must be a whole number")
            continue
        if not lo <= val <= hi:
            problems.append(f"{name} must be between {lo} and {hi}")
            continue
        if allowed is not None and int(val) not in allowed:
            problems.append(f"{name} must be one of {sorted(allowed)}")
            continue
        clean[name] = int(val) if is_int else round(val, 2)
    if problems:
        raise ValidationError("; ".join(problems))
    return clean


class Predictor:
    def __init__(self, model_path, metrics_path=None):
        bundle = joblib.load(model_path)
        self.models = bundle["models"]
        self.features = bundle["features"]
        self.target_recall = bundle.get("target_recall")
        self.trained_with = bundle.get("sklearn_version")
        self.rows = bundle.get("rows")
        self.metrics = None
        if metrics_path and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)

    def predict(self, values):
        x = np.array([[values[f] for f in self.features]], dtype=float)
        out = {}
        for name, m in self.models.items():
            p = float(m["pipeline"].predict_proba(x)[0, 1])
            out[name] = {"probability": round(p * 100, 1),
                         "threshold": round(m["threshold"] * 100, 1),
                         "flagged": p >= m["threshold"]}
        flagged = any(v["flagged"] for v in out.values())   # either model -> flag
        return {"models": out, "flagged": flagged}
