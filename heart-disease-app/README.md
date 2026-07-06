# CardioSignal — Heart Disease Risk Screening

A deployable web app built on top of the original `DataScience---Heart-Disease-main` project. It takes new
patient vitals through a web form, runs them through the same ensemble model from the dissertation notebook
(Random Forest + Logistic Regression + SVM, soft voting), and returns a risk readout with probabilities.

> ⚠️ **Not a diagnostic device.** This is an academic implementation of a final-year ML project. It should
> never be used to make real clinical decisions. See the in-app disclaimer.

## Quick start in VS Code

1. Open this folder in VS Code (`File → Open Folder…`).
2. `Ctrl+Shift+P` → `Tasks: Run Task` → run **"1. Create venv + install requirements"**, then
   **"2. Train the ensemble model"**.
3. Open the **Run and Debug** panel → select **"Flask: Run CardioSignal (app.py)"** → press `F5`.
4. Visit **http://127.0.0.1:5000**.

See **`PROJECT_GUIDE.md`** for a full walkthrough of how the code fits together, what to say if
Prof. Ogunleye asks you to explain a specific part, and ideas for extending the project further.

## What's inside

```
heart-disease-app/
├── .vscode/                 # One-click VS Code setup (tasks, debug configs)
│   ├── launch.json
│   ├── tasks.json
│   ├── settings.json
│   └── extensions.json
├── app.py                  # Flask server (UI + JSON API)
├── train_model.py          # Reproduces the notebook pipeline, saves the model
├── data/
│   └── heart_data.csv      # Original training data (from Heart Attack.csv)
├── model/
│   ├── ensemble_model.joblib   # Trained VotingClassifier (generated)
│   └── metrics.json            # Accuracy / confusion matrix / feature importances (generated)
├── templates/
│   ├── index.html          # Patient intake + prediction UI
│   └── about.html           # Model methodology & metrics page
├── static/
│   ├── css/style.css
│   └── js/script.js
├── requirements.txt
├── Procfile                 # For Render / Railway / Heroku-style platforms
├── runtime.txt
├── PROJECT_GUIDE.md          # How the code fits together + defense talking points
└── README.md
```

## How the model works

This reproduces the **best-performing** model from the original notebook
(`LogesticREgression(Heart_Desise_DataSet).ipynb`), which the README reported at ~94% accuracy:

1. Features: `age, gender, impluse (heart rate), pressurehight (systolic BP), pressurelow (diastolic BP),
   glucose, kcm (CK-MB), troponin`
2. Target: `class` → `negative` = 0, `positive` = 1
3. 80/20 train/test split, `random_state=42`
4. Three base models trained on the same features:
   - `RandomForestClassifier(n_estimators=100, random_state=42)`
   - `LogisticRegression()`
   - `SVC(kernel='linear', probability=True)`
5. Combined into a `VotingClassifier(voting='soft')` — final prediction is the averaged class probability
   across all three models.

`train_model.py` re-runs this pipeline against `data/heart_data.csv` and saves:
- `model/ensemble_model.joblib` — the fitted ensemble, used by the Flask app to score new patients
- `model/metrics.json` — accuracy, ROC AUC, confusion matrix, and Random Forest feature importances,
  shown on the `/about` page

## Run it locally

Requires Python 3.10+.

```bash
cd heart-disease-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Train the model (only needed once, or whenever you change the data)
python train_model.py

# Start the web app
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Click "Load sample case" to try it instantly, or enter your
own values and click "Run diagnosis".

## API

The frontend calls a small JSON API you can also use directly (e.g. for the "new intakes of data" requirement,
or to hook up another client):

```
POST /api/predict
Content-Type: application/json

{
  "age": 63,
  "gender": 1,
  "impluse": 92,
  "pressurehight": 152,
  "pressurelow": 95,
  "glucose": 168,
  "kcm": 8.2,
  "troponin": 0.45
}
```

Response:

```json
{
  "ok": true,
  "prediction": "positive",
  "prediction_label": "Indicators consistent with heart disease risk",
  "probability_positive": 91.4,
  "probability_negative": 8.6,
  "risk_level": "High",
  "flags": ["Troponin is above the typical reference range (>0.04 ng/mL)."],
  "input": { "...": "echoed back" },
  "model_accuracy": 92.8
}
```

`GET /api/metrics` returns the full metrics.json (accuracy, confusion matrix, feature importances) —
useful if you want to build another dashboard on top of this model later.

`GET /healthz` is a simple uptime check for deployment platforms.

## Deploying it

Any platform that runs a Python web service works. Two common free/cheap options:

### Render.com
1. Push this folder to a GitHub repo.
2. In Render: **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt && python train_model.py`
4. Start command: `gunicorn app:app`
5. Deploy. Render will give you a public URL.

### Railway.app
1. Push this folder to GitHub, then **New Project → Deploy from GitHub repo** in Railway.
2. Railway auto-detects the `Procfile`. Add a build step / one-off command to run
   `python train_model.py` before first deploy (or just commit the generated `model/` folder so it's
   already trained — see note below).
3. Deploy — Railway gives you a public URL.

### PythonAnywhere / any VPS
1. Upload the folder, `pip install -r requirements.txt` in a virtualenv.
2. Run `python train_model.py` once.
3. Point your WSGI config at `app:app` (PythonAnywhere) or run `gunicorn app:app` behind nginx (VPS).

**Note on the model file:** `model/ensemble_model.joblib` is a build artifact. You can either (a) commit it
to your repo so no training step is needed at deploy time, or (b) run `python train_model.py` as part of
your build command, as shown above. Either works — the model trains in a couple of seconds since the
dataset is small (1,319 rows).

## Retraining with more data

To retrain on an updated or larger dataset, replace `data/heart_data.csv` with a CSV that has the same
columns (`age, gender, impluse, pressurehight, pressurelow, glucose, kcm, troponin, class`) and re-run:

```bash
python train_model.py
```

Restart the Flask app (or redeploy) afterward so it picks up the new `model/ensemble_model.joblib`.

## Limitations to keep in mind

- Trained on 1,319 records from a single dataset — not validated on an external population.
- The "clinical reference flags" shown alongside the prediction are simple threshold callouts for context,
  not part of the model's decision — they're there to make the result explainable, not to replace clinical
  judgment.
- This tool does not store or transmit patient data anywhere; predictions happen in-memory on the server for
  each request.
