# CardioSignal — Project Guide (for Williams)

This is the "explain it to yourself, then explain it to Prof. Ogunleye" version of the docs.
The `README.md` covers setup and deployment; this file covers **how the project fits together
and how to talk about it** — as the person who trained the model *and* the person who built
the app around it.

---

## 1. Two lenses on the same project

You're presenting this as a final-year CS dissertation, but you built it like a fullstack
engineer would. Both framings are true at once, and examiners tend to respond well when you can
switch between them:

| As a **CS / ML student** | As a **fullstack developer** |
|---|---|
| Ensemble learning (Random Forest + Logistic Regression + SVM, soft voting) reduces variance vs. any single model | The ML pipeline is isolated in `train_model.py` — a build step, not runtime logic |
| 92.8% accuracy, 0.97 ROC AUC on a held-out 20% test split (`random_state=42` for reproducibility) | `app.py` loads the *already-trained* model artifact (`model/ensemble_model.joblib`) at startup — the server never retrains on request, which keeps `/api/predict` fast |
| Feature importance shows troponin (0.57) and CK-MB (0.26) dominate the decision — consistent with real cardiac biomarker literature | Feature importances are persisted to `model/metrics.json` and rendered on `/about`, so the "why" is inspectable, not just the "what" |
| Validated only on 1,319 records from one dataset — a real limitation you should state up front | Input validation (`validate_and_parse`) and physiological bounds (`FIELD_BOUNDS`) happen server-side, not just in the HTML form — so the API is safe to call directly, not just from the UI |

If a question feels like a "the model" question, answer from the left column. If it feels like a
"the app" question, answer from the right. Most defense panels ask a mix of both.

---

## 2. Architecture, in one picture

```
 Browser (templates/index.html + static/js/script.js)
        │  fetch('/api/predict', {method:'POST', body: JSON})
        ▼
 Flask app (app.py)
        │  validate_and_parse()  → catches bad/missing/out-of-range input
        │  model.predict_proba() → loads model ONCE at startup, not per-request
        ▼
 model/ensemble_model.joblib  (produced offline by train_model.py)
        │
        ▼
 data/heart_data.csv  (1,319 rows: age, gender, vitals, biomarkers → class)
```

Key design decision worth mentioning in your defense: **training and serving are separate
processes.** `train_model.py` is a one-off/offline script; `app.py` is the always-on web server.
This is exactly how production ML systems are structured (train offline, serve a frozen
artifact) — it's not just a Flask convenience, it's the textbook-correct architecture, and
it's worth saying that explicitly since it shows you understand *why*, not just *how*.

---

## 3. Code walkthrough (what to point at, in order)

1. **`data/heart_data.csv`** — the raw dataset. Know the column names cold:
   `age, gender, impluse, pressurehight, pressurelow, glucose, kcm, troponin, class`.
   (Yes, `impluse` and `pressurehight`/`pressurelow` are misspellings in the original dataset —
   if asked, say so plainly rather than pretending they're intentional.)

2. **`train_model.py`** — the ML pipeline:
   - Loads the CSV, encodes `class` (`negative`→0, `positive`→1)
   - 80/20 train/test split, `random_state=42` (reproducibility — a favorite exam-style question)
   - Trains three base learners on the raw features, combines them with
     `VotingClassifier(voting='soft')` — soft voting averages predicted *probabilities* across
     models rather than just majority-voting labels, which is why you get a probability score
     in the UI, not just a yes/no.
   - Saves the fitted ensemble and a `metrics.json` (accuracy, ROC AUC, confusion matrix,
     feature importances) — this file is what powers the `/about` page and the API's
     `model_accuracy` field.

3. **`app.py`** — the Flask server:
   - Loads the model + metrics once at import time (`model = joblib.load(...)`)
   - `validate_and_parse()` — server-side validation with physiological bounds, independent of
     whatever the HTML form does. This matters because `/api/predict` is a public JSON endpoint;
     anyone (or any script) can call it directly, so validation can't live only in the browser.
   - `interpret_result()` — turns the raw probability into a `risk_level` (Low/Moderate/High) and
     a list of plain-English clinical reference flags. These flags are explicitly *not* part of
     the model's decision — they're threshold callouts for explainability. Say this clearly if
     asked; it shows you understand the difference between a model output and a rule-based
     explanation layered on top.
   - Three routes matter: `/` (form), `/api/predict` (JSON API), `/about` (methodology + metrics).

4. **`templates/index.html` + `static/js/script.js`** — the frontend:
   - Plain `fetch()` call, no framework — appropriate for the scope of the project, and easy to
     explain line-by-line if asked to walk through it live.
   - The gauge, bars, and EKG line in the hero are all inline SVG animated with CSS/JS — no
     external chart library, which keeps the app dependency-light and fast to load.

---

## 4. Running it in VS Code

This repo now ships with a `.vscode/` folder so you don't have to remember commands during a
live demo.

**First time setup:**
1. Open the `heart-disease-app` folder in VS Code (`File → Open Folder…`).
2. Install the recommended extensions if prompted (Python + Python Debugger).
3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) → `Tasks: Run Task` → run, in order:
   - **1. Create venv + install requirements**
   - **2. Train the ensemble model**
4. Go to the **Run and Debug** panel (left sidebar, the ▷🐞 icon) and choose
   **"Flask: Run CardioSignal (app.py)"**, then press `F5`.
5. Open **http://127.0.0.1:5000** in your browser.

Running with `F5` (instead of just `python app.py` in a terminal) means you get real breakpoints —
useful if a lecturer asks you to demonstrate what happens inside `validate_and_parse()` or
`interpret_result()` for a specific input.

**Every time after that:** just re-run the "Flask: Run CardioSignal" debug config (`F5`). You
only need to re-run "Train the ensemble model" if you change `data/heart_data.csv`.

---

## 5. Likely questions and how to answer them

- **"Why an ensemble instead of one model?"** — Different models make different kinds of
  mistakes; soft voting averages their probability estimates, which tends to be more stable than
  any single model, especially on a modest-sized dataset (1,319 rows).
- **"Why is troponin weighted so heavily?"** — It lines up with clinical practice: troponin is a
  standard biomarker for myocardial injury, so a model that leans on it is learning something
  clinically sensible, not just a statistical artifact.
- **"How do you handle bad input?"** — Twice: HTML5 `min`/`max`/`required` in the form for
  immediate UX feedback, and `validate_and_parse()` server-side with the same bounds, because the
  API endpoint is reachable independently of the form.
- **"Is this safe to use as a real diagnostic tool?"** — No, and the app says so explicitly (the
  disclaimer banner). It's a trained pattern-matcher on one dataset of 1,319 records, not a
  validated clinical instrument. Saying this proactively, before being asked, tends to land well
  with examiners.
- **"What would you improve with more time?"** — Cross-validation instead of a single train/test
  split, a larger/external validation dataset, calibration curves for the probability outputs,
  and feature engineering (e.g. pulse pressure = systolic − diastolic) instead of raw vitals.

---

## 6. If you want to extend it further

- Add `GET /api/metrics` (already exists) to a small chart on the `/about` page comparing base
  models vs. the ensemble.
- Log each prediction (input + output, no PII) to a local SQLite file for a "recent screenings"
  view — a natural fullstack extension if you want to show database work too.
- Containerize with a `Dockerfile` if you want to demonstrate deployment beyond Render/Railway.

None of this is required for what you've already built — it's a complete, working, well-tested
project as-is. This section is only here in case you want to go further for the final
presentation.
