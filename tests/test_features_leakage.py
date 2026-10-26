"""Properties that make the historical-rate features honest. Each test
would have failed on an earlier version of features.py."""
import numpy as np
import pandas as pd
import pytest

from features import add_historical_rates, build_feature_table, required_joined_columns

T0 = pd.Timestamp("2018-01-01")


def _orders(spec):
    """spec: list of (group, purchase_day, delivered_day, delayed)"""
    return pd.DataFrame({
        "g": [s[0] for s in spec],
        "order_purchase_timestamp": [T0 + pd.Timedelta(days=s[1]) for s in spec],
        "order_delivered_customer_date": [T0 + pd.Timedelta(days=s[2]) for s in spec],
        "is_delayed": [s[3] for s in spec],
    })


def _rate(df):
    return add_historical_rates(df, "g", "r")["r"]


def test_an_orders_own_outcome_never_affects_its_own_feature():
    base = _orders([("A", 0, 5, 1), ("A", 10, 15, 0), ("A", 20, 25, 1)])
    flipped = base.copy()
    flipped.loc[2, "is_delayed"] = 0
    assert _rate(base)[2] == _rate(flipped)[2]


def test_later_orders_never_affect_an_earlier_orders_feature():
    base = _orders([("A", 0, 5, 1), ("A", 10, 15, 0), ("A", 30, 35, 1)])
    flipped = base.copy()
    flipped.loc[2, "is_delayed"] = 0
    assert _rate(base)[0] == _rate(flipped)[0]
    assert _rate(base)[1] == _rate(flipped)[1]


def test_outcome_is_only_visible_after_delivery_not_after_purchase():
    """A was bought day 0 but delivered day 5. B (bought day 2) cannot know
    A's outcome yet; C (bought day 6) can."""
    df = _orders([("A", 0, 5, 1), ("A", 2, 7, 0), ("A", 6, 11, 0)])
    r = _rate(df)
    # B (index 1): A not delivered yet -> no group history -> fallback, not A's 1.0
    assert r[1] != 1.0
    # C (index 2): A delivered on day 5 (< 6) -> sees A only (rate 1.0); B delivered day 7, not yet
    assert r[2] == 1.0


def test_history_uses_only_delivered_orders_and_averages_them():
    df = _orders([("A", 0, 2, 1), ("A", 1, 3, 0), ("A", 10, 12, 0)])
    assert _rate(df)[2] == pytest.approx(0.5)        # A0 (late) and A1 (on time), both delivered by day 10


def test_fallback_does_not_look_at_the_future():
    """Changing a far-future outcome must not change an early order's
    fallback rate (the first version used the all-time mean)."""
    spec = [("A", 0, 2, 1), ("B", 10, 12, 0), ("B", 20, 22, 0), ("C", 100, 102, 1)]
    base = _orders(spec)
    flipped = base.copy()
    flipped.loc[3, "is_delayed"] = 0                 # far future
    rb, rf = _rate(base), _rate(flipped)
    assert rb[1] == rf[1]                            # B's first order: fallback from A only
    assert rb[1] == pytest.approx(1.0)               # platform-wide rate known on day 10 = A's outcome


def test_orders_before_anything_was_delivered_get_zero():
    df = _orders([("A", 0, 5, 1), ("B", 1, 6, 1)])
    assert list(_rate(df)) == [0.0, 0.0]


def test_groups_are_independent():
    df = _orders([("A", 0, 2, 1), ("B", 0, 2, 0), ("A", 10, 12, 0), ("B", 10, 12, 0)])
    r = _rate(df)
    assert r[2] == 1.0 and r[3] == 0.0


def test_feature_table_names_every_missing_column():
    with pytest.raises(ValueError) as e:
        build_feature_table(pd.DataFrame({"a": [1]}))
    assert "missing" in str(e.value) and required_joined_columns()[0] in str(e.value)
