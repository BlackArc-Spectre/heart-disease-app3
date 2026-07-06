"""
app.py
------
Flask web application that serves the heart-disease ensemble model
(Random Forest + Logistic Regression + SVM, soft voting) for new patient
intake and prediction.

Run locally:
    python app.py
Then open http://127.0.0.1:5000
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "ensemble_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

app = Flask(__name__)

# ---- Load model + metadata once at startup ----
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "Model artifact not found. Run `python train_model.py` first to train "
        "and save the model before starting the server."
    )

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

FEATURE_COLUMNS = METRICS["feature_columns"]
FEATURE_META = METRICS["feature_meta"]

# Reasonable physiological bounds used for basic server-side validation
FIELD_BOUNDS = {
    "age": (0, 120),
    "gender": (0, 1),
    "impluse": (0, 300),
    "pressurehight": (40, 300),
    "pressurelow": (20, 200),
    "glucose": (20, 900),
    "kcm": (0, 400),
    "troponin": (0, 50),
}


def validate_and_parse(payload):
    """Validate incoming JSON payload, return (values_dict, errors_list)."""
    values = {}
    errors = []

    for col in FEATURE_COLUMNS:
        raw = payload.get(col, None)
        if raw is None or raw == "":
            errors.append(f"Missing value for '{FEATURE_META[col]['label']}'.")
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{FEATURE_META[col]['label']}' must be numeric.")
            continue

        lo, hi = FIELD_BOUNDS[col]
        if val < lo or val > hi:
            errors.append(
                f"'{FEATURE_META[col]['label']}' looks out of range "
                f"(expected {lo}-{hi})."
            )
            continue

        values[col] = val

    return values, errors


def interpret_result(probability_positive, input_values):
    """Build a human-readable risk breakdown alongside the raw prediction."""
    if probability_positive >= 0.75:
        risk_level = "High"
    elif probability_positive >= 0.4:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # Simple, transparent flags based on common clinical reference ranges.
    # These are informational callouts, not part of the model's decision.
    flags = []
    if input_values["troponin"] > 0.04:
        flags.append("Troponin is above the typical reference range (>0.04 ng/mL).")
    if input_values["kcm"] > 5:
        flags.append("CK-MB is elevated above the typical reference range (>5 ng/mL).")
    if input_values["pressurehight"] >= 140 or input_values["pressurelow"] >= 90:
        flags.append("Blood pressure reading falls in a hypertensive range.")
    if input_values["glucose"] >= 126:
        flags.append("Blood glucose is above the typical fasting threshold (>=126 mg/dL).")
    if input_values["impluse"] > 100 or input_values["impluse"] < 60:
        flags.append("Heart rate is outside the typical resting range (60-100 bpm).")

    return risk_level, flags


@app.route("/")
def index():
    return render_template(
        "index.html",
        feature_columns=FEATURE_COLUMNS,
        feature_meta=FEATURE_META,
        accuracy=round(METRICS["accuracy"] * 100, 2),
        auc=round(METRICS["roc_auc"] * 100, 2),
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        metrics=METRICS,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()

    values, errors = validate_and_parse(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    x = pd.DataFrame([[values[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    proba = model.predict_proba(x)[0]
    prediction = int(model.predict(x)[0])
    probability_positive = float(proba[1])
    probability_negative = float(proba[0])

    risk_level, flags = interpret_result(probability_positive, values)

    result = {
        "ok": True,
        "prediction": "positive" if prediction == 1 else "negative",
        "prediction_label": (
            "Indicators consistent with heart disease risk"
            if prediction == 1
            else "Indicators not consistent with heart disease risk"
        ),
        "probability_positive": round(probability_positive * 100, 2),
        "probability_negative": round(probability_negative * 100, 2),
        "risk_level": risk_level,
        "flags": flags,
        "input": values,
        "model_accuracy": round(METRICS["accuracy"] * 100, 2),
    }
    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
