"""
tracking.py - third score: an order that is in transit, re-scored from its carrier scans, failed delivery attempts,
weather and strikes known at a chosen moment.

IMPORTANT: the tracking data is SIMULATED (make_tracking_dataset.py). Olist publishes none of it. Everything this module
returns is labelled simulated = True, and the score is shown next to, never instead of, the checkout score.
Uses the same feature builder as training (src/tracking_features.py), so serving cannot drift from training.
"""
import json
import os
import sys
import threading

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Same convention as engine.py: put src/ on the path and import modules by their plain names. (A top-level
# "src" package installed elsewhere on a machine can shadow our src folder, which has no __init__.py.)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from tracking_features import LABELS, build_features, load_context  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "tracking_model.json")
META_PATH = os.path.join(ROOT, "models", "tracking_meta.json")
DATA_DIR = os.path.join(ROOT, "data", "synthetic")
_lock = threading.Lock()
_cache = {}


class TrackingUnavailable(RuntimeError):
    pass


def _load():
    with _lock:
        if "model" in _cache:
            return _cache
        need = [MODEL_PATH, META_PATH] + [os.path.join(DATA_DIR, f) for f in
                                          ("tracking_events_simulated.csv", "weather_daily_simulated.csv", "strikes_simulated.csv",
                                           "hub_congestion_daily_simulated.csv", "road_disruptions_simulated.csv",
                                           "holidays_brazil.csv")]
        if not all(os.path.exists(p) for p in need):
            raise TrackingUnavailable("The tracking model or data is missing. Run make_tracking_dataset.py and "
                                      "train_tracking_model.py.")
        m = xgb.XGBClassifier()
        m.load_model(MODEL_PATH)
        ev = pd.read_csv(os.path.join(DATA_DIR, "tracking_events_simulated.csv"), parse_dates=["event_time"])
        ctx = load_context(DATA_DIR)
        _cache.update(model=m, meta=json.load(open(META_PATH)), events=ev, ctx=ctx)
        return _cache


def info() -> dict:
    try:
        meta = _load()["meta"]
    except TrackingUnavailable as e:
        return {"available": False, "detail": str(e), "simulated": True}
    return {"available": True, "simulated": True, **{k: meta[k] for k in (
        "stage", "with_tracking", "without_tracking", "monthly_backtest", "top_features", "note", "n_orders",
        "n_snapshots", "threshold")}}


def resolve_time(row, as_of=None, days_after_purchase=None) -> pd.Timestamp:
    purchased = pd.Timestamp(row["purchased_at"])
    if as_of not in (None, ""):
        try:
            t = pd.Timestamp(as_of)
        except (ValueError, TypeError):
            raise ValueError("Use a date and time such as 2018-04-02T14:30.")
        if pd.isna(t):
            raise ValueError("Use a date and time such as 2018-04-02T14:30.")
    elif days_after_purchase is not None:
        try:
            t = purchased + pd.Timedelta(days=float(days_after_purchase))
        except (ValueError, TypeError):
            raise ValueError("days_after_purchase must be a number.")
    else:
        raise ValueError("Give as_of or days_after_purchase.")
    if t < purchased:
        raise ValueError(f"That time is before the order was placed ({purchased:%d %b %Y %H:%M}).")
    if t > purchased + pd.Timedelta(days=120):
        raise ValueError("That time is more than 120 days after the order. Check the date.")
    return t


def _frame(row):
    keys = ["order_id", "purchased_at", "promised_by", "approval_delay_hours", "promised_delivery_days",
            "seller_customer_distance_km", "customer_state", "seller_state"]
    return pd.DataFrame([{k: row[k] for k in keys}])


def timeline(row, as_of: pd.Timestamp) -> dict:
    """Scans known at `as_of` plus weather and strike context. Never includes the DELIVERED event."""
    c = _load()
    f = build_features(_frame(row), c["events"], c["ctx"], [as_of]).iloc[0]
    e = c["events"]
    e = e[(e["order_id"] == row["order_id"]) & (e["event_time"] <= as_of) & (e["event_type"] != "DELIVERED")]
    return {"simulated": True, "as_of": str(as_of),
            "events": [{"time": str(r.event_time), "type": r.event_type, "state": r.hub_state}
                       for r in e.sort_values("event_time").itertuples()],
            "context": {k: (None if pd.isna(f[k]) else float(f[k])) for k in (
                "n_scans", "hours_since_last_scan", "max_gap_hours", "n_failed_attempts", "out_for_delivery",
                "dest_weather_3d_max", "origin_weather_3d_max", "dest_strike_now", "origin_strike_now")}}


def score(row, as_of: pd.Timestamp) -> dict:
    c = _load()
    model, meta, ev, ctx = c["model"], c["meta"], c["events"], c["ctx"]
    if pd.notna(row.get("delivered_at")) and as_of >= pd.Timestamp(row["delivered_at"]):
        raise ValueError("This order was already delivered by then, so there is nothing left to forecast.")
    if as_of >= pd.Timestamp(row["promised_by"]):
        raise ValueError("This order is already past its promised date, so it is late already. Pick an earlier time.")
    o = _frame(row)
    f = build_features(o, ev, ctx, [as_of])
    X = pd.DataFrame([{k: row[k] for k in meta["feature_columns"] if k not in f.columns}])
    X = pd.concat([X.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    for col, cats in meta["categorical"].items():
        X[col] = pd.Categorical(X[col].astype(str), categories=cats)
    X = X[meta["feature_columns"]]
    p = float(model.predict_proba(X)[:, 1][0])
    contrib = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[0][:-1]
    reasons = []
    for i in np.argsort(-contrib)[:6]:
        name = meta["feature_columns"][i]
        if contrib[i] > 0.05 and name in LABELS and (name in meta["tracking_features"] + meta["weather_features"]):
            v = X.iloc[0][name]
            reasons.append({"feature": name, "label": LABELS[name], "push": float(contrib[i]),
                            "value": None if pd.isna(v) else float(v)})
    thr = float(meta["threshold"])
    band = "CRITICAL" if p >= 0.5 else "HIGH" if p >= max(thr * 1.5, 0.3) else "MEDIUM" if p >= thr else "LOW"
    return {"simulated": True, "stage": "in transit (simulated tracking)", "probability": p, "risk_level": band,
            "flagged": p >= thr, "threshold": thr, "as_of": str(as_of), "days_left": round(float(f["days_left"].iloc[0]), 1),
            "days_since_purchase": round(float(f["days_since_purchase"].iloc[0]), 1),
            "failed_attempts": int(f["n_failed_attempts"].iloc[0]), "scans": int(f["n_scans"].iloc[0]),
            "longest_silence_hours": None if pd.isna(f["max_gap_hours"].iloc[0]) else round(float(f["max_gap_hours"].iloc[0]), 1),
            "reasons": reasons}
