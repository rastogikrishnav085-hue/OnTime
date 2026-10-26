import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from enriched_features import EXTRA_COLUMNS, GROUPS, LABELS
from ensemble import shift_odds


def test_extra_columns_are_37_unique_and_labelled():
    assert len(EXTRA_COLUMNS) == 37 == len(set(EXTRA_COLUMNS))
    assert sorted(sum(GROUPS.values(), [])) == sorted(EXTRA_COLUMNS)
    assert set(EXTRA_COLUMNS) <= set(LABELS)


def test_no_outcome_column_among_features():
    bad = ("delivered_customer", "delay_days", "review", "is_delayed")
    assert not [c for c in EXTRA_COLUMNS if any(b in c for b in bad)]


def test_shift_odds_is_monotone_and_bounded():
    p = np.linspace(0.001, 0.999, 50)
    s = shift_odds(p, 11.0)
    assert (np.diff(s) > 0).all() and s.min() > 0 and s.max() < 1 and (s > p).all()
