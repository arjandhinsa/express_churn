"""Data drift monitoring: compare the live scoring cohort against the
feature distribution the model was trained on.

Caveat: the reference (training cohort, anchored at the second-to-last
visit) and the scoring cohort (anchored at the last visit) are built by
different rules, so some drift is structural rather than real. Track the
trend in drifted_share across runs, not any single value. Prediction
drift is less affected and is reported separately.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# share of drifted columns above which we call it dataset drift
DRIFT_THRESHOLD = 0.5


def run_drift_check(reference: pd.DataFrame, current: pd.DataFrame,
                    out_dir: Path,
                    reference_probs=None, current_probs=None,
                    threshold: float = DRIFT_THRESHOLD) -> dict:
    """Compare current features (and optionally predictions) to the training reference.

    Writes a human-readable HTML report and a machine-readable status file.
    Patient identity is dropped before anything reaches the report — drift
    only needs distributions.
    """
    ref = reference.reset_index(drop=True)
    cur = current[reference.columns].reset_index(drop=True)   # same columns, same order

    if reference_probs is not None and current_probs is not None:
        ref["churn_prob"] = pd.Series(reference_probs).reset_index(drop=True)
        cur["churn_prob"] = pd.Series(current_probs).reset_index(drop=True)

    report = Report([DataDriftPreset()])
    result = report.run(cur, ref)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result.save_html(str(out_dir / f"drift_{stamp}.html"))
    result.save_html(str(out_dir / "drift_latest.html"))

    metrics = result.dict()["metrics"]

    summary = next(m for m in metrics
                   if m["metric_name"].startswith("DriftedColumnsCount"))

    prediction_drift = next(
        (m["value"] for m in metrics if "column=churn_prob" in m["metric_name"]),
        None,
    )

    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_columns": len(ref.columns),
        "n_drifted": int(summary["value"]["count"]),
        "drifted_share": float(summary["value"]["share"]),
        "prediction_drift": float(prediction_drift) if prediction_drift is not None else None,
        "threshold": threshold,
        "drift_detected": float(summary["value"]["share"]) >= threshold,
        "reference_rows": len(ref),
        "current_rows": len(cur),
    }

    (out_dir / "drift_status.json").write_text(json.dumps(status, indent=2))
    return status