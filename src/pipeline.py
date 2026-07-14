from pathlib import Path
from src.dataset import load_tables, build_model_inputs
from src.labels import build_labels
from src.features import build_features
from src.model import build_model

# --- config: the one place these live ---
WORKING_DIR = Path(__file__).parent.parent / "data_working"

def run_pipeline(working_dir=WORKING_DIR):
    """Load → derive → label → features → train. Returns (model, metrics)."""
    pat, app, rx, orders = load_tables(working_dir)
    completed, interval, data_end = build_model_inputs(pat, app, rx, orders)
    labels = build_labels(completed, interval, data_end)
    features = build_features(labels, completed, orders, pat)
    model, metrics = build_model(features)
    return model, metrics

if __name__ == "__main__":
    model, metrics = run_pipeline()
    print(metrics)