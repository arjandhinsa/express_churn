import os

from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from src.dataset import load_tables, build_model_inputs
from src.features import build_features
from src.model import prepare_features
from src.priority import build_priority


ROOT = Path(__file__).resolve().parent.parent


SYNTHETIC = bool(os.getenv("DATA_DIR"))
FLAVOUR = "synthetic" if SYNTHETIC else "real"

WORKING_DIR = Path(os.getenv("DATA_DIR", ROOT / "data_working"))
MODELS_DIR = ROOT / "models" / FLAVOUR
OUTPUTS_DIR = ROOT / "outputs" / FLAVOUR



def build_scoring_cohort(completed, interval):
    """ 
    Building the live-scoring cohort, which consists of established patients (2+ completed visits)
    with a valid recall interval.

    Anchored at each patient's last completed visit so live features are 
    computed identically to training features.

    At scoring time the recall window is still open; predicting it is the model's job.
    """
    g = completed.groupby("patient_key")
    cohort = pd.DataFrame({
        "index_visit": g["AppointDate"].max(),   # last completed visit
        "n_visits": g.size(),
    })
    cohort = cohort[cohort["n_visits"] >= 2]          # established patients only
    cohort["ReaMonths"] = cohort.index.map(interval)
    cohort = cohort.dropna(subset=["ReaMonths"])
    return cohort[["index_visit", "ReaMonths"]] 


def main():
    # 1. load
    pat, app, rx, orders = load_tables(WORKING_DIR)
    completed, interval, data_end = build_model_inputs(pat, app, rx, orders)

    # 2. cohort + features (reusing tested modules end to end)
    cohort = build_scoring_cohort(completed, interval)
    features = build_features(cohort, completed, orders, pat)

    # 3. score
    model = XGBClassifier()
    model.load_model(MODELS_DIR / "churn_model.json")
    X = prepare_features(features)
    churn_probs = pd.Series(model.predict_proba(X)[:, 1], index=features.index)

    # 4. value + priority
    values = (features["total_spend"] + 157.0) / (features["visit_count"] + 1)
    ranked = build_priority(churn_probs, values)

    # 5. write: timestamped history + the file the API serves
    stamp = pd.Timestamp.today().strftime("%Y-%m-%d")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(OUTPUTS_DIR / f"recall_list_{stamp}.csv")
    ranked.to_csv(OUTPUTS_DIR / "recall_list.csv")

    print(f"Scored {len(ranked)} patients")
    print(f"Top-20% expected recovery: £{ranked['priority'].head(int(len(ranked)*0.2)).sum() * 0.2:,.0f}")



if __name__ == "__main__":
    main()