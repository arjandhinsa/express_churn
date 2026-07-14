import pandas as pd
import numpy as np

def build_features(labels, completed, orders, pat) -> pd.DataFrame:
    # the prediction point for every patient
    obs = labels.copy()
    index_map = obs["index_visit"]

    # appointments strictly BEFORE or AT the index visit (the leakage wall)
    appts = completed[completed["patient_key"].isin(obs.index)].copy()
    appts["index_visit"] = appts["patient_key"].map(index_map)
    appts_upto = appts[appts["AppointDate"] <= appts["index_visit"]]

    # FEATURE 1: visit_count (visits up to and including index)
    obs["visit_count"] = obs.index.map(appts_upto.groupby("patient_key").size())
    
    
    # FEATURE 2: tenure_days — first visit to index visit
    first_upto = appts_upto.groupby("patient_key")["AppointDate"].min()
    obs["first_visit"] = obs.index.map(first_upto)
    obs["tenure_days"] = (obs["index_visit"] - obs["first_visit"]).dt.days

    # FEATURE 3:  avg gap = tenure / (number of gaps); gaps = visits - 1
    # for 1-visit patients, visits-1 = 0 -> NaN (undefined, XGBoost handles it)
    gaps = obs["visit_count"] - 1
    obs["avg_gap_days"] = obs["tenure_days"] / gaps.where(gaps > 0)


    # orders up to the index visit (leakage wall)
    ord_c = orders[orders["patient_key"].isin(obs.index)].copy()
    ord_c["index_visit"] = ord_c["patient_key"].map(index_map)
    ord_upto = ord_c[ord_c["OrderDate"] <= ord_c["index_visit"]]

    # FEATURES 4 & 5
    obs["total_spend"] = obs.index.map(ord_upto.groupby("patient_key")["Amount"].sum()).fillna(0)
    obs["order_count"] = obs.index.map(ord_upto.groupby("patient_key").size()).fillna(0)

    by  = pat.set_index("patient_key")["birth_year"]
    sex = pat.set_index("patient_key")["Sex"]
    obs["birth_year"] = obs.index.map(by)
    obs["sex"]        = obs.index.map(sex)

    # age as of the INDEX visit (prediction point), not today
    obs["age"] = obs["index_visit"].dt.year - obs["birth_year"]

    return obs