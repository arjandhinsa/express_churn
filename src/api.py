from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier

from src.model import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Recall Engine")

# load ONCE at startup, not per request
model = XGBClassifier()
model.load_model(ROOT / "models" / "churn_model.json")


class PatientFeatures(BaseModel):
    visit_count: int
    tenure_days: float
    ReaMonths: int
    avg_gap_days: float | None
    total_spend: float
    order_count: int
    age: int
    sex: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(features: PatientFeatures):
    row = pd.DataFrame([features.model_dump()])
    row["sex"] = (row["sex"] == "M").astype(int)        # same encoding as training
    row = row[FEATURE_COLS]                              # exact training column order
    churn_prob = float(model.predict_proba(row)[:, 1][0])
    value = (features.total_spend + 157.0) / (features.visit_count + 1)
    return {"churn_prob": churn_prob, "value": value, "priority": churn_prob * value}


@app.get("/recall-list")
def recall_list(top_n: int = 20):
    path = ROOT / "outputs" / "recall_list.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No recall list generated yet")
    return pd.read_csv(path).head(top_n).to_dict("records")