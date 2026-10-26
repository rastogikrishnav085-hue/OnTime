"""The per-order guardrail: reject the malformed, warn on the unusual,
and never punish a genuinely risky order for being risky."""
import numpy as np
import pytest

from inference_validation import validate_order, is_missing

NUM = ["total_price", "distance", "seller_rate"]
CAT = ["payment_type"]
FEATS = NUM + CAT
STATS = {
    "numeric": {
        "total_price": {"min": 1.0, "max": 500.0, "p01": 5.0, "p99": 300.0},
        "distance": {"min": 0.0, "max": 3000.0, "p01": 1.0, "p99": 2000.0},
        "seller_rate": {"min": 0.0, "max": 1.0, "p01": 0.0, "p99": 0.4},
    },
    "categorical": {"payment_type": ["credit_card", "boleto"]},
    "rate_cols": ["seller_rate"],
}
GOOD = {"total_price": 50.0, "distance": 400.0, "seller_rate": 0.1, "payment_type": "boleto"}


def check(**over):
    rec = {**GOOD, **over}
    return validate_order(rec, FEATS, NUM, CAT, STATS)


def test_good_order_passes_cleanly():
    r = check()
    assert r["ok"] and not r["warnings"]


@pytest.mark.parametrize("blank", [None, np.nan, "", "   "])
def test_every_kind_of_blank_categorical_is_rejected(blank):
    """Regression: a blank CSV cell is NaN, not None, and used to slip through."""
    r = check(payment_type=blank)
    assert not r["ok"] and "Missing" in r["reject_reasons"][0]


@pytest.mark.parametrize("blank", [None, np.nan, ""])
def test_blank_numeric_is_rejected(blank):
    assert not check(total_price=blank)["ok"]


def test_missing_key_is_rejected():
    rec = dict(GOOD)
    del rec["distance"]
    assert not validate_order(rec, FEATS, NUM, CAT, STATS)["ok"]


@pytest.mark.parametrize("bad", ["12", "abc", True, [1], {"a": 1}])
def test_non_numeric_values_are_rejected(bad):
    assert not check(total_price=bad)["ok"]


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinite_is_rejected(bad):
    assert not check(distance=bad)["ok"]


def test_negative_where_training_never_negative_is_rejected():
    assert not check(total_price=-5.0)["ok"]


def test_rate_above_one_is_rejected():
    assert not check(seller_rate=1.4)["ok"]


def test_high_rate_is_not_penalised():
    """A seller with an unusually high delay rate is the signal, not bad input."""
    r = check(seller_rate=0.9)
    assert r["ok"] and not r["warnings"]


def test_value_inside_training_range_is_not_warned_even_if_in_the_tail():
    assert not check(total_price=450.0)["warnings"]      # above p99 but within seen range


def test_value_beyond_training_range_is_warned_not_rejected():
    r = check(distance=50000.0)
    assert r["ok"] and any("extrapolating" in w for w in r["warnings"])


def test_unseen_category_is_warned_not_rejected():
    r = check(payment_type="crypto")
    assert r["ok"] and any("not seen" in w for w in r["warnings"])


def test_numpy_numbers_are_accepted():
    assert check(total_price=np.float64(10.0), distance=np.int64(5))["ok"]


def test_is_missing_helper():
    assert is_missing(None) and is_missing(np.nan) and is_missing(" ")
    assert not is_missing(0) and not is_missing(0.0) and not is_missing("a")
