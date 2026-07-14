import pandas as pd

def build_labels(completed, interval, data_end, grace_days=90) -> pd.DataFrame:
    """Per-patient churn labels, indexed by patient_key.
    Returns observable established patients only (~5,029 rows);
    columns include churned and index_visit."""

    appts_sorted = completed.sort_values(["patient_key", "AppointDate"])
    g = appts_sorted.groupby("patient_key")
    index_visit = g.nth(-2).set_index("patient_key")["AppointDate"]
    last_visit  = g.nth(-1).set_index("patient_key")["AppointDate"]

    cohort2 = pd.DataFrame({"index_visit": index_visit, "last_visit": last_visit})
    cohort2["ReaMonths"] = cohort2.index.map(interval)
    cohort2 = cohort2.dropna(subset=["ReaMonths", "index_visit"])

    cohort2["window_close"] = cohort2["index_visit"] + \
        pd.to_timedelta(cohort2["ReaMonths"] * 30 + grace_days, unit="D")
    cohort2["observable"] = cohort2["window_close"] <= data_end

    # filter to observable, then test each patient's appointments against their window
    obs = cohort2[cohort2["observable"]].copy()

    cc = completed[completed["patient_key"].isin(obs.index)].copy()
    cc["index_visit"]  = cc["patient_key"].map(obs["index_visit"])
    cc["window_close"] = cc["patient_key"].map(obs["window_close"])

    in_window = ((cc["AppointDate"] > cc["index_visit"]) &
                 (cc["AppointDate"] <= cc["window_close"]))
    returned = in_window.groupby(cc["patient_key"]).any()

    obs["returned"] = obs.index.map(returned).fillna(False)
    obs["churned"]  = (~obs["returned"]).astype(int)

    return obs[["index_visit", "churned", "ReaMonths"]]