import os

import pandas as pd
from pathlib import Path
from src.dataset import load_tables, build_model_inputs
from src.labels import build_labels
from src.features import build_features
from src.model import build_model

# --- config: the one place these live ---
ROOT = Path(__file__).resolve().parent.parent


SYNTHETIC = bool(os.getenv("DATA_DIR"))
FLAVOUR = "synthetic" if SYNTHETIC else "real"

WORKING_DIR = Path(os.getenv("DATA_DIR", ROOT / "data_working"))
MODELS_DIR = ROOT / "models" / FLAVOUR


def run_pipeline(working_dir=WORKING_DIR):
    """Load → derive → label → features → train. Returns (model, metrics)."""
    pat, app, rx, orders = load_tables(working_dir)
    completed, interval, data_end = build_model_inputs(pat, app, rx, orders)
    labels = build_labels(completed, interval, data_end)
    features = build_features(labels, completed, orders, pat)
    model, metrics, test_scores, reference = build_model(features)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODELS_DIR / "churn_model.json")
    reference.to_parquet(MODELS_DIR / "reference.parquet")

    return model, metrics, test_scores



if __name__ == "__main__":
    model, metrics, test_scores = run_pipeline()
    print(metrics)