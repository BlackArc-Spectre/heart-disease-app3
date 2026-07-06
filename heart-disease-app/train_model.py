"""
train_model.py
----------------
Reproduces the modeling pipeline from the original project notebook
(LogesticREgression(Heart_Desise_DataSet).ipynb) and saves a deployable
artifact for the web app.

Pipeline (matches the notebook):
1. Load Heart Attack.csv
2. Encode target: 'negative' -> 0, 'positive' -> 1
3. Train/test split (80/20, random_state=42)
4. Train three base models on the RAW features (as in the notebook):
     - Random Forest (n_estimators=100, random_state=42)
     - Logistic Regression
     - SVM (linear kernel, probability=True)
5. Combine them into a soft-voting VotingClassifier ensemble
   (this was the best-performing model in the notebook: ~94% accuracy)
6. Persist the fitted ensemble + metadata needed by the Flask app.

Run:
    python train_model.py
"""

import json
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "heart_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
FEATURE_COLUMNS = [
    "age",
    "gender",
    "impluse",
    "pressurehight",
    "pressurelow",
    "glucose",
    "kcm",
    "troponin",
]

# Human-friendly labels / units shown in the UI and used in API responses
FEATURE_META = {
    "age": {"label": "Age", "unit": "years"},
    "gender": {"label": "Gender", "unit": "0 = Female, 1 = Male"},
    "impluse": {"label": "Heart Rate (Impulse)", "unit": "bpm"},
    "pressurehight": {"label": "Systolic Blood Pressure", "unit": "mmHg"},
    "pressurelow": {"label": "Diastolic Blood Pressure", "unit": "mmHg"},
    "glucose": {"label": "Blood Glucose", "unit": "mg/dL"},
    "kcm": {"label": "CK-MB", "unit": "ng/mL"},
    "troponin": {"label": "Troponin", "unit": "ng/mL"},
}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["class"] = df["class"].map({"negative": 0, "positive": 1})

    X = df[FEATURE_COLUMNS]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    logistic_model = LogisticRegression(max_iter=1000)
    svm_model = SVC(kernel="linear", probability=True, random_state=42)

    rf_model.fit(X_train, y_train)
    logistic_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    voting_classifier = VotingClassifier(
        estimators=[
            ("Random Forest", rf_model),
            ("Logistic Regression", logistic_model),
            ("SVM", svm_model),
        ],
        voting="soft",
    )
    voting_classifier.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = voting_classifier.predict(X_test)
    y_proba = voting_classifier.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Ensemble Accuracy: {accuracy:.4f}")
    print(f"Ensemble ROC AUC:  {auc:.4f}")
    print("Confusion Matrix:", cm)

    # Feature importance from the Random Forest branch (for explainability in the UI)
    importances = dict(zip(FEATURE_COLUMNS, rf_model.feature_importances_.tolist()))

    metrics = {
        "accuracy": accuracy,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importances": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "feature_meta": FEATURE_META,
    }

    joblib.dump(voting_classifier, os.path.join(MODEL_DIR, "ensemble_model.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model to model/ensemble_model.joblib")
    print("Saved metrics to model/metrics.json")


if __name__ == "__main__":
    main()
