

"""
Test for src/priority.py

Three fake patients

The aim of the test is to prove that the priority ranking disagrees with the churn ranking

Formula: Priority = P(churn) * value = expected £ recovered per call

"""


import pandas as pd
import pytest


from src.priority import build_priority


@pytest.fixture
def churn_probs():
    return pd.Series({"A": 0.9, "B": 0.6, "C": 0.3})

@pytest.fixture
def values():
    return pd.Series({"A": 40.0, "B": 600.0, "C": 200.0})

def test_build_priority(churn_probs, values):
    result = build_priority(churn_probs, values)

    # Ranked by priority, not by churn — the point of the module
    assert list(result.index) == ["B", "C", "A"]

    # Hand-calculated priorities
    assert result.loc["B", "priority"] == pytest.approx(360.0)
    assert result.loc["C", "priority"] == pytest.approx(60.0)
    assert result.loc["A", "priority"] == pytest.approx(36.0)

    # Running total down the ranked list
    assert list(result["cumulative_expected"]) == pytest.approx([360.0, 420.0, 456.0])

    # Inputs carried along with each row
    assert result.loc["B", "churn_prob"] == pytest.approx(0.6)
    assert result.loc["B", "value"] == pytest.approx(600.0)

    assert len(result) == 3


