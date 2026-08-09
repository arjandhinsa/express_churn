import json
import os
from pathlib import Path

from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from xgboost import XGBClassifier

from src.model import prepare_features
from src.value import K, PRIOR


ROOT = Path(__file__).resolve().parent.parent

SYNTHETIC = bool(os.getenv("DATA_DIR"))
FLAVOUR = "synthetic" if SYNTHETIC else "real"

WORKING_DIR = Path(os.getenv("DATA_DIR", ROOT / "data_working"))
MODELS_DIR = ROOT / "models" / FLAVOUR
OUTPUTS_DIR = ROOT / "outputs" / FLAVOUR

app = FastAPI(title="Recall Engine")

# load ONCE at startup, not per request
model = XGBClassifier()
model.load_model(MODELS_DIR / "churn_model.json")


class PatientFeatures(BaseModel):
    visit_count: int
    tenure_days: float
    ReaMonths: int
    avg_gap_days: float | None
    total_spend: float
    order_count: int
    age: int
    sex: str


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(features: PatientFeatures):
    row = pd.DataFrame([features.model_dump()])
    row = prepare_features(row)                            # exact training column order
    churn_prob = float(model.predict_proba(row)[:, 1][0])
    value = (features.total_spend + K * PRIOR) / (features.visit_count + K)
    return {"churn_prob": churn_prob, "value": value, "priority": churn_prob * value}


@app.get("/recall-list")
def recall_list(top_n: int = 20):
    path = OUTPUTS_DIR / "recall_list.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No recall list generated yet")
    generated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return {
        "generated_at": generated_at,
        "flavour": FLAVOUR,
        "count": top_n,
        "patients": pd.read_csv(path).head(top_n).to_dict("records"),
    }


@app.get("/monitoring")
def monitoring():
    """Latest data and prediction drift status."""
    path = OUTPUTS_DIR / "monitoring" / "drift_status.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No drift check has run yet")
    return json.loads(path.read_text())


@app.get("/monitoring/report", include_in_schema=False)
def monitoring_report():
    """Full Evidently drift report (HTML)."""
    path = OUTPUTS_DIR / "monitoring" / "drift_latest.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No drift report generated yet")
    return FileResponse(path, media_type="text/html")