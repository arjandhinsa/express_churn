
import pandas as pd
import pytest

from src.batch_score import build_scoring_cohort

@pytest.fixture
def completed():
    # one ROW PER VISIT, like the real appointments table
    return pd.DataFrame({
        "patient_key": ["P1", "P1", "P1",   # 3 visits → in
                        "P2",               # 1 visit  → excluded (not established)
                        "P3", "P3",         # 2 visits, no ReaMonths → excluded
                        "P4", "P4"],        # 2 visits → in
        "AppointDate": pd.to_datetime(["2023-01-10", "2022-01-05", "2022-08-10", #shuffled
                                       "2023-03-01",
                                       "2023-01-15", "2023-08-01",
                                       "2023-01-15", "2023-07-20", ]),   
    })

@pytest.fixture
def interval():
    # Series keyed by patient with P3 missing 
    return pd.Series({"P1": 24, "P2": 12, "P4": 12})


def test_build_scoring_cohort(completed, interval):
    cohort = build_scoring_cohort(completed, interval)

    assert set(cohort.index) == {"P1", "P4"}                      
    assert cohort.loc["P1", "index_visit"] == pd.Timestamp("2023-01-10")  # P1's LAST visit date
    assert cohort.loc["P1", "ReaMonths"] == 24
    assert cohort.loc["P4", "ReaMonths"] == 12