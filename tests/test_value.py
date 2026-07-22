"""Tests for src/value.py.

Five fake patients, each exercising one rule from docs/design.md:
  P1  big spender      — own history dominates the prior
  P2  modest regular   — also has a DNA (completed=False) that must NOT count
  P3  zero-dispense    — informative missingness: prior dragged down by visits
  P4  single visit     — weak evidence, prior tempers it
  P5  leakage trap     — has a post-index order that must be ignored

Formula: (spend_before_index + k*prior) / (completed_visits_before_index + k)
Convention: order_date <= index_date counts (mirrors the features rule:
"at or before the prediction point").
"""

import pandas as pd
import pytest

from src.value import avg_spend_per_visit, assign_tiers


@pytest.fixture
def visits():
    return pd.DataFrame({
        "patient_key": ["P1", "P1", "P1", "P1",
                        "P2", "P2", "P2",
                        "P3", "P3", "P3",
                        "P4",
                        "P5", "P5"],
        "visit_date": pd.to_datetime([
            "2022-01-05", "2022-08-10", "2023-01-10", "2023-06-01",
            "2023-01-15", "2023-07-20", "2023-09-01",
            "2022-03-01", "2022-11-01", "2023-05-01",
            "2023-03-01",
            "2023-01-15", "2023-08-01",
        ]),
        "completed": [True, True, True, True,
                      True, True, False,      # P2's third visit is a DNA
                      True, True, True,
                      True,
                      True, True],
    })


@pytest.fixture
def orders():
    return pd.DataFrame({
        "patient_key": ["P1", "P1", "P2", "P4", "P5", "P5"],
        "order_date": pd.to_datetime([
            "2023-01-10", "2023-06-01",   # P1: 500 + 300 = 800
            "2023-02-01",                 # P2: 150
            "2023-03-01",                 # P4: 100
            "2023-01-15", "2024-05-01",   # P5: 200 before index, 500 AFTER
        ]),
        "value": [500.0, 300.0, 150.0, 100.0, 200.0, 500.0],
    })
    # NOTE: P3 has no order rows at all — the function must handle that.


@pytest.fixture
def index_dates():
    return pd.DataFrame({
        "patient_key": ["P1", "P2", "P3", "P4", "P5"],
        "index_date": pd.to_datetime([
            "2023-12-01", "2023-12-01", "2023-12-01", "2023-12-01",
            "2024-01-01",                 # P5's 500 order (2024-05-01) is after this
        ]),
    })


def test_avg_spend_per_visit(orders, visits, index_dates):
    result = avg_spend_per_visit(orders, visits, index_dates, k=1, prior=157.0)

    # Hand-calculated: (spend + 1*157) / (completed_visits + 1)
    assert result["P1"] == pytest.approx((800 + 157) / 5)    # 191.40
    assert result["P2"] == pytest.approx((150 + 157) / 3)    # 102.33 — DNA not counted
    assert result["P3"] == pytest.approx((0 + 157) / 4)      # 39.25  — no orders at all
    assert result["P4"] == pytest.approx((100 + 157) / 2)    # 128.50
    assert result["P5"] == pytest.approx((200 + 157) / 3)    # 119.00 — post-index 500 ignored

    # One row per cohort patient, no strays
    assert len(result) == 5


def test_assign_tiers():
    values = pd.Series(
        [39.25, 102.33, 128.50, 191.40, 119.00],
        index=["P3", "P2", "P4", "P1", "P5"],
    )
    # Frozen train-period cutoffs: Bronze < 50 <= Silver < 110 <= Gold < 180 <= Platinum
    cutoffs = [50.0, 110.0, 180.0]

    tiers = assign_tiers(values, cutoffs)

    assert tiers["P3"] == "Bronze"
    assert tiers["P2"] == "Silver"
    assert tiers["P4"] == "Gold"
    assert tiers["P5"] == "Gold"
    assert tiers["P1"] == "Platinum"