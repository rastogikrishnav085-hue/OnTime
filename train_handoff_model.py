"""
train_handoff_model.py - Stage 2 model: "after the carrier picks the order up".

Why it exists
  The checkout model only knows what is known when the order is placed (ROC-AUC about 0.80 on a random split,
  about 0.67 when tested month by month). Once the carrier collects the parcel, three more things are known:
  how long the seller took to hand it over, how many days of the promise are left, and how many km per day that
  leaves. Using them is NOT leakage, because they exist before the delivery happens. This model is a SEPARATE,
  second score. It must never be mixed with, or reported as, the checkout score.

What it does not use (that would be leakage): delivery date, review score, order status after handoff,
anything from the delivered order's own outcome.

Run (Windows):   .venv\\Scripts\\activate   then   python train_handoff_model.py
Needs olist_orders_dataset.csv in data/raw (or pass --raw PATH). data/processed/ontime_model_ready.csv is used for the
checkout features. Writes models/handoff_model.json and models/handoff_meta.json.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
HANDOFF_FEATURES = ["handling_hours", "days_left_at_handoff", "frac_promise_left", "km_per_day_needed",
                    "handoff_dow", "handoff_hour"]
PARAMS = dict(learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
              tree_method="hist", enable_categorical=True, n_jobs=4)


def handoff_features(df, handed_over_at):
    """Features known at the moment of handoff. `df` needs the checkout columns plus purchased_at/approved_at."""
    out = pd.DataFrame(index=df.index)
    approved = df["approved_at"]
    out["handling_hours"] = ((handed_over_at - approved).dt.total_seconds() / 3600).clip(-24, 24 * 30)
    left = (df["promised_by"] - handed_over_at).dt.total_seconds() / 86400
    out["days_left_at_handoff"] = left
    out["frac_promise_left"] = left / df["promised_delivery_days"].clip(lower=1)
    out["km_per_day_needed"] = df["seller_customer_distance_km"] / left.clip(lower=0.5)
    out["handoff_dow"] = handed_over_at.dt.dayofweek
    out["handoff_hour"] = handed_over_at.dt.hour
    return out


def to_frame(df, cols, categories):
    X = df[cols].copy()
    for c, cats in categories.items():
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    return X


def main(raw):
    meta = json.load(open(os.path.join(ROOT, "models", "model_metadata.json")))
    base, cat = meta["feature_columns"], meta["categorical"]
    d = pd.read_csv(os.path.join(ROOT, "data", "processed", "ontime_model_ready.csv"),
                    parse_dates=["order_purchase_timestamp"])
    o = pd.read_csv(os.path.join(raw, "olist_orders_dataset.csv"),
                    parse_dates=["order_approved_at", "order_delivered_carrier_date", "order_estimated_delivery_date"],
                    usecols=["order_id", "order_approved_at", "order_delivered_carrier_date",
                             "order_estimated_delivery_date"])
    d = d.merge(o, on="order_id", how="left").dropna(subset=["order_delivered_carrier_date", "order_approved_at"])
    d = d.rename(columns={"order_approved_at": "approved_at", "order_estimated_delivery_date": "promised_by"})
    # Impossible timestamps (a known Olist quirk): the carrier cannot collect an order before it was placed.
    # The app rejects these too, so training and serving follow the same rule.
    gap = d["order_delivered_carrier_date"] - d["order_purchase_timestamp"]
    n0 = len(d)
    d = d[(gap >= pd.Timedelta(0)) & (gap <= pd.Timedelta(days=120))]
    print(f"dropped {n0 - len(d)} orders with impossible carrier timestamps")
    d = d.sort_values("order_delivered_carrier_date").reset_index(drop=True)
    h = handoff_features(d, d["order_delivered_carrier_date"])
    d = pd.concat([d, h], axis=1)
    y = d["is_delayed"].astype(int).values
    cols = base + HANDOFF_FEATURES
    categories = {c: sorted(d[c].astype(str).unique().tolist()) for c in cat}
    X = to_frame(d, cols, categories)
    print(f"{len(d):,} orders with a carrier handoff, {y.mean():.1%} late")

    def fit(Xa, ya, Xv=None, yv=None, n=None, use_cols=None):
        m = xgb.XGBClassifier(n_estimators=n or 900, eval_metric="auc",
                              early_stopping_rounds=60 if Xv is not None else None, **PARAMS)
        if Xv is not None:
            m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        else:
            m.fit(Xa, ya)
        return m

    def report(model_cols, name):
        Xtr, Xte, ytr, yte = train_test_split(X[model_cols], y, test_size=0.2, stratify=y, random_state=42)
        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=1)
        m = fit(Xa, ya, Xv, yv)
        pv, pt = m.predict_proba(Xv)[:, 1], m.predict_proba(Xte)[:, 1]
        ts = np.linspace(0.02, 0.9, 89)
        thr = float(ts[np.argmax([f1_score(yv, pv >= t) for t in ts])])     # chosen on validation, never on test
        a = pt >= thr
        r = dict(name=name, roc_auc=float(roc_auc_score(yte, pt)), pr_auc=float(average_precision_score(yte, pt)),
                 threshold=thr, precision=float(precision_score(yte, a)), recall=float(recall_score(yte, a)),
                 f1=float(f1_score(yte, a)), accuracy=float(accuracy_score(yte, a)),
                 n_train=int(len(ytr)), n_test=int(len(yte)), best_iteration=int(m.best_iteration))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        return r, m

    r_check, _ = report(base, "Checkout features only (same orders)")
    r_hand, model = report(cols, "Checkout + carrier handoff")

    # Strict check: train on all earlier months, test on the next month only.
    d["ym"] = d["order_delivered_carrier_date"].dt.to_period("M")
    folds = {"checkout": [], "handoff": []}
    for mth in sorted(d["ym"].unique()):
        if not ("2017-04" <= str(mth) <= "2018-08"):
            continue
        tr, te = (d["ym"] < mth).values, (d["ym"] == mth).values
        if te.sum() < 300 or y[te].sum() < 15:
            continue
        for key, cs in (("checkout", base), ("handoff", cols)):
            m = fit(X.loc[tr, cs], y[tr], n=250)
            m.set_params(learning_rate=0.05, max_depth=5)
            folds[key].append((str(mth), float(roc_auc_score(y[te], m.predict_proba(X.loc[te, cs])[:, 1]))))
    bt = {k: {"mean_auc": float(np.mean([a for _, a in v])), "months": len(v), "by_month": dict(v)} for k, v in folds.items()}
    print("monthly backtest mean AUC:", {k: round(v["mean_auc"], 3) for k, v in bt.items()})

    # Final model: refit on all orders with the iteration count found above.
    final = fit(X, y, n=int(r_hand["best_iteration"] * 1.15) + 1)
    final.save_model(os.path.join(ROOT, "models", "handoff_model.json"))
    imp = pd.Series(final.feature_importances_, index=cols).sort_values(ascending=False)
    json.dump({
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": "after carrier handoff", "feature_columns": cols, "handoff_features": HANDOFF_FEATURES,
        "categorical": categories, "threshold": r_hand["threshold"],
        "random_split": r_hand, "checkout_same_orders": r_check, "monthly_backtest": bt,
        "top_features": {k: float(v) for k, v in imp.head(10).items()},
        "n_orders": int(len(d)), "late_rate": float(y.mean()),
        "note": ("Second score, used only once the carrier has collected the parcel. The random-split AUC is optimistic "
                 "because late rates swing with the season; the monthly backtest is the honest figure."),
    }, open(os.path.join(ROOT, "models", "handoff_meta.json"), "w"), indent=1)
    print("saved models/handoff_model.json and models/handoff_meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    main(ap.parse_args().raw)
