"""
handoff.py - Stage 2 score: re-score an order once the carrier has collected it.

Uses models/handoff_model.json (trained by train_handoff_model.py). Everything it needs is known at that moment:
the checkout features already in the queue, plus when the carrier collected the parcel.
It is a second score and is always shown next to, never instead of, the checkout score.
"""
import json
import os
import threading
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "models", "handoff_model.json")
META_PATH = os.path.join(ROOT, "models", "handoff_meta.json")

LABELS = {
    "handling_hours": "Time the seller took to hand the parcel over",
    "days_left_at_handoff": "Days of the promise left at handoff",
    "frac_promise_left": "Share of the promised window left at handoff",
    "km_per_day_needed": "Distance per day the carrier must cover",
    "handoff_dow": "Weekday of the handoff",
    "handoff_hour": "Hour of the handoff",
}
_lock = threading.Lock()
_cache = {}


class HandoffUnavailable(RuntimeError):
    pass


def _load():
    with _lock:
        if "model" in _cache:
            return _cache["model"], _cache["meta"]
        if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
            raise HandoffUnavailable("The handoff model is not trained yet. Run train_handoff_model.py.")
        meta = json.load(open(META_PATH))
        m = xgb.XGBClassifier()
        m.load_model(MODEL_PATH)
        _cache.update(model=m, meta=meta)
        return m, meta


def info() -> dict:
    """Metrics for the Model page. Returns {'available': False} when the model file is missing."""
    try:
        _, meta = _load()
    except HandoffUnavailable as e:
        return {"available": False, "detail": str(e)}
    return {"available": True, **{k: meta[k] for k in (
        "stage", "random_split", "checkout_same_orders", "monthly_backtest", "top_features", "note", "n_orders",
        "threshold")}}


def _band(p: float, thr: float) -> str:
    return "CRITICAL" if p >= 0.5 else "HIGH" if p >= max(thr * 1.5, 0.3) else "MEDIUM" if p >= thr else "LOW"


def score(row: pd.Series, handed_over_at: pd.Timestamp) -> dict:
    """`row` is one queue row (checkout features, purchased_at, approval_delay_hours, promised_by)."""
    model, meta = _load()
    purchased = pd.Timestamp(row["purchased_at"])
    promised = pd.Timestamp(row["promised_by"])
    handed = pd.Timestamp(handed_over_at)
    if handed < purchased:
        raise ValueError("The carrier cannot collect an order before it was placed "
                         f"({purchased:%d %b %Y %H:%M}).")
    if handed > purchased + pd.Timedelta(days=120):
        raise ValueError("That handoff time is more than 120 days after the order. Check the date.")
    approved = purchased + pd.Timedelta(hours=float(row.get("approval_delay_hours") or 0))
    left = (promised - handed).total_seconds() / 86400
    feats = {
        "handling_hours": float(np.clip((handed - approved).total_seconds() / 3600, -24, 24 * 30)),
        "days_left_at_handoff": left,
        "frac_promise_left": left / max(float(row["promised_delivery_days"]), 1.0),
        "km_per_day_needed": float(row["seller_customer_distance_km"]) / max(left, 0.5),
        "handoff_dow": handed.dayofweek, "handoff_hour": handed.hour,
    }
    X = pd.DataFrame([{c: row[c] for c in meta["feature_columns"] if c not in feats}])
    for c, v in feats.items():
        X[c] = v
    for c, cats in meta["categorical"].items():
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    X = X[meta["feature_columns"]]
    p = float(model.predict_proba(X)[:, 1][0])
    contrib = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[0][:-1]
    reasons = []
    for i in np.argsort(-contrib)[:3]:
        name = meta["feature_columns"][i]
        if contrib[i] > 0.05 and name in LABELS:
            reasons.append({"feature": name, "label": LABELS[name], "push": float(contrib[i]), "value": feats[name]})
    thr = float(meta["threshold"])
    return {"stage": "after carrier handoff", "probability": p, "risk_level": _band(p, thr), "flagged": p >= thr,
            "threshold": thr, "days_left": round(left, 1), "handling_hours": round(feats["handling_hours"], 1),
            "km_per_day_needed": round(feats["km_per_day_needed"], 1), "reasons": reasons}
