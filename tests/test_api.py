"""Quick API check with a stub for flask_cors / DB. Run: python tests/test_api.py"""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import flask_cors  # noqa: F401
except ImportError:
    m = types.ModuleType("flask_cors"); m.CORS = lambda *a, **k: None; sys.modules["flask_cors"] = m
try:
    import pymysql  # noqa: F401
except ImportError:
    m = types.ModuleType("pymysql"); m.connect = lambda **k: None
    c = types.ModuleType("pymysql.cursors"); c.DictCursor = object
    m.cursors = c; sys.modules["pymysql"] = m; sys.modules["pymysql.cursors"] = c
import app as A

good = dict(age=58, sex=1, cp=4, trestbps=140, chol=240, fbs=0, restecg=2, thalach=120,
            exang=1, oldpeak=2.5, slope=2, ca=2, thal=7, save=False)
c = A.app.test_client()
h = c.get("/api/health").get_json(); print("health:", h)
assert h["model_loaded"] and not h["db_configured"]
r = c.post("/api/predict", json=good); print(r.status_code, r.get_json()["models"], r.get_json()["flagged"])
assert r.status_code == 200
bad = dict(good, age=500, cp=9, chol="abc")
r = c.post("/api/predict", json=bad); print(r.status_code, r.get_json()); assert r.status_code == 400
r = c.post("/api/predict", json={}); assert r.status_code == 400
assert c.get("/api/model-info").status_code == 200
assert c.get("/api/history").status_code == 503      # no DB configured -> clean error
print("ALL OK")
