"""
audit_leakage.py
-----------------
Adversarial, empirical leakage check for every checkout feature (core 33 + enriched 37).

Idea: a checkout feature of an order bought at time t may depend only on what was known at t.
So for a cut-off date T, if we SCRAMBLE every outcome that happens at or after T (labels flipped,
carrier / delivery dates pushed to the year 2100), the features of every order purchased BEFORE T
must not change at all. If any of them moves, that feature leaks future information.

    python audit_leakage.py                (about 1 minute; reads data/interim/olist_joined.csv)

Writes data/processed/leakage_audit.json and exits non-zero if any feature moved.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from enriched_features import EXTRA_COLUMNS, build_extra_features       # noqa: E402
from features import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_table  # noqa: E402

DATES = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
         "order_delivered_customer_date", "order_estimated_delivery_date", "shipping_limit_min"]
FAR = pd.Timestamp("2100-01-01")
CUTS = ["2017-06-01", "2017-11-01", "2018-03-01", "2018-06-15"]
# These two use dataset-wide MEDIANS of checkout-time quantities to fill blanks (not outcomes); they are
# allowed to differ at the 1e-9 level and are listed separately so nobody mistakes them for outcome leakage.
MEDIAN_FILLED = {"approval_delay_hours", "total_weight_g", "avg_weight_g", "total_volume_cm3", "seller_customer_distance_km",
                 "total_payment_value", "n_payment_installments", "n_payment_methods", "avg_photos_qty", "avg_name_length",
                 "avg_description_length"}


def scramble(joined, T):
    j = joined.copy()
    carrier_after = j["order_delivered_carrier_date"] >= T
    deliv_after = j["order_delivered_customer_date"] >= T
    j.loc[carrier_after, "order_delivered_carrier_date"] = FAR
    j.loc[deliv_after, "order_delivered_customer_date"] = FAR
    flip = deliv_after
    j.loc[flip, TARGET_COLUMN] = 1 - j.loc[flip, TARGET_COLUMN]
    if "delay_days" in j:
        j.loc[flip, "delay_days"] = np.where(j.loc[flip, "delay_days"] > 0, -5.0, 5.0)
    return j


def features(j, holidays):
    core = build_feature_table(j).reset_index(drop=True)
    extra = build_extra_features(core, holidays)
    return pd.concat([core[FEATURE_COLUMNS], extra], axis=1)


def main():
    path = os.path.join(ROOT, "data", "interim", "olist_joined.csv")
    joined = pd.read_csv(path, parse_dates=DATES + ["order_delivered_carrier_date"])
    holidays = os.path.join(ROOT, "data", "external", "holidays_brazil.csv")
    base = features(joined, holidays)
    base_t = joined["order_purchase_timestamp"].reset_index(drop=True)
    cols = FEATURE_COLUMNS + EXTRA_COLUMNS
    report = {"cuts": {}, "n_features_checked": len(cols), "leaking_features": [], "median_filled_features_that_moved": []}
    for T in CUTS:
        T = pd.Timestamp(T)
        alt = features(scramble(joined, T), holidays)
        before = (base_t < T).to_numpy()
        moved, med_moved = [], []
        for c in cols:
            a, b = base.loc[before, c], alt.loc[before, c]
            if a.dtype == object or str(a.dtype).startswith("str"):
                same = (a.astype(str).values == b.astype(str).values).all()
                delta = 0.0 if same else 1.0
            else:
                a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
                both_nan = np.isnan(a) & np.isnan(b)
                diff = np.where(both_nan, 0.0, np.abs(a - b))
                delta = float(np.nanmax(diff)) if len(diff) else 0.0
                same = bool(np.isnan(diff).sum() == 0 and delta <= 1e-9)
            if not same:
                (med_moved if c in MEDIAN_FILLED else moved).append({"feature": c, "max_abs_change": delta})
        report["cuts"][str(T.date())] = {"orders_checked": int(before.sum()), "features_that_moved": moved,
                                         "median_filled_that_moved": med_moved}
        report["leaking_features"] += [m["feature"] for m in moved]
        report["median_filled_features_that_moved"] += [m["feature"] for m in med_moved]
        print(f"cut {T.date()}: {int(before.sum()):,} earlier orders, leaking features: {[m['feature'] for m in moved] or 'none'}"
              f"  (median-filled that moved: {[m['feature'] for m in med_moved] or 'none'})", flush=True)
    report["leaking_features"] = sorted(set(report["leaking_features"]))
    report["median_filled_features_that_moved"] = sorted(set(report["median_filled_features_that_moved"]))
    report["verdict"] = "PASS: no feature depends on outcomes after the order's purchase time" if not report["leaking_features"] else "FAIL"
    out = os.path.join(ROOT, "data", "processed", "leakage_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(report["verdict"])
    return 0 if not report["leaking_features"] else 1


if __name__ == "__main__":
    sys.exit(main())
