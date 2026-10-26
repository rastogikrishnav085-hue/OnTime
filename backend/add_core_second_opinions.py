"""Add original XGBoost and core-ensemble comparison scores without changing the served queue model.

The queue may be served by the enriched 70-feature checkout model. This script preserves its
probability/risk/action columns and only refreshes p_xgboost and p_ensemble so the UI's second-opinion
and disagreement views work for the same held-out queue orders.

Run after model artifacts and backend/app_data/queue.csv are present:
    python backend/add_core_second_opinions.py
"""
from __future__ import annotations
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from artifacts import load_model
from ensemble import EnsembleError, load_ensemble
from features import FEATURE_COLUMNS
from modeling import NUMERIC

APP_DATA = os.path.join(ROOT, "backend", "app_data")
MODELS = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data", "processed", "ontime_model_ready.csv")


def main():
    queue_path = os.path.join(APP_DATA, "queue.csv")
    q = pd.read_csv(queue_path, dtype={"order_id": str, "seller_id": str})
    data = pd.read_csv(DATA, dtype={"order_id": str})
    # Guard against scoring a stale or accidentally modified queue.
    data["order_purchase_timestamp"] = pd.to_datetime(data["order_purchase_timestamp"], errors="coerce")
    ordered = data.sort_values("order_purchase_timestamp", kind="stable")
    test = ordered.iloc[int(len(ordered) * 0.8):]
    if set(q.order_id) != set(test.order_id):
        raise SystemExit("Queue IDs do not match the newest chronological test block. Rebuild app data first.")
    X = q[FEATURE_COLUMNS].copy()
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    prod, _ = load_model(MODELS, FEATURE_COLUMNS, data_path=DATA)
    q["p_xgboost"] = np.round(prod.predict_proba(X)[:, 1], 6)
    try:
        ensemble, _ = load_ensemble(os.path.join(MODELS, "ensemble"), FEATURE_COLUMNS)
        q["p_ensemble"] = np.round(ensemble.predict_proba(X)[:, 1], 6)
    except EnsembleError as exc:
        print(f"Core ensemble unavailable; p_ensemble will be null: {exc}")
        q["p_ensemble"] = np.nan
    q.to_csv(queue_path, index=False, float_format="%.7g")
    print(f"Added core-model comparison scores to {len(q):,} held-out queue orders; preserved serving model={__import__('json').load(open(os.path.join(APP_DATA, 'meta.json')))['serving']}.")


if __name__ == "__main__":
    main()
