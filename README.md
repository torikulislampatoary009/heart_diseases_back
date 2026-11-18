# Heart Disease Diagnostic Tool - backend (Flask + MySQL)

| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | model + database status |
| POST | /api/predict | JSON with the 13 UCI fields (+ optional `"save": false`) |
| GET | /api/model-info | test-set results of both models |
| GET | /api/history?page=1&limit=10 | saved checks |
| DELETE | /api/history/<id> | delete a saved check |
| GET | /api/stats | totals |

## Environment variables (set on Render, never in GitHub)
`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` - copy from the Aiven console.
`DB_SSL_MODE` - `REQUIRED` (default, works with Aiven) or `VERIFY` + `DB_SSL_CA` (paste ca.pem text).
Optional: `CORS_ORIGINS`, `MODEL_PATH`.

## Deploy on Render (free)
1. Put `models/heart_model.joblib`, `metrics.json`, `sklearn_version.txt` in this folder's `models/` (see model/README.md).
2. Push to GitHub. New Web Service -> pick the repo -> Root Directory `backend`, Runtime Python 3.
3. Build command:
   `pip install -r requirements.txt && pip install "scikit-learn==$(cat models/sklearn_version.txt)"`
   Start command:
   `gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT`
   Add env var `PYTHON_VERSION=3.11.9`.
4. Add the DB_* variables under Environment, deploy, open `/api/health`.
   The `predictions` table is created automatically.

The build command installs the exact scikit-learn version the model was trained with,
because a saved model can fail to load on a different version.

## Local run
    pip install -r requirements.txt scikit-learn
    copy .env.example .env      (fill in)
    python app.py
Without DB variables the API still predicts; it just skips saving history.
