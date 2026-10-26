"""
add_second_opinion.py
----------------------
Scores the app's order queue with BOTH models (the production XGBoost and the
ensemble from train_ensemble.py) and stores both probabilities in
backend/app_data/queue.csv (columns p_xgboost, p_ensemble).

It also decides which model the queue itself is shown with (probability,
risk band, alert flag, reasons, suggested action, priority):

    .venv\\Scripts\\activate
    python backend/add_second_opinion.py                 # follow the ensemble report's decision
    python backend/add_second_opinion.py --serve xgboost # force the original model
    python backend/add_second_opinion.py --serve ensemble

Needs only the queue's own feature columns, so it does not need the raw Olist
files. Run it after train_ensemble.py (rebuild_app_data.bat does this).
"""
import argparse
import json
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
warnings.filterwarnings("ignore")

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from artifacts import load_model                                 # noqa: E402
from categories import weight_of                                 # noqa: E402
from ensemble import EnsembleError, load_ensemble                # noqa: E402
from features import FEATURE_COLUMNS                             # noqa: E402
from modeling import NUMERIC, risk_bucket                        # noqa: E402
from prediction import explain_orders                            # noqa: E402
from scoring import suggested_action                             # noqa: E402

APP_DATA = os.path.join(ROOT, "backend", "app_data")
MODELS = os.path.join(ROOT, "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", choices=["auto", "xgboost", "ensemble"], default="auto")
    args = ap.parse_args()

    q = pd.read_csv(os.path.join(APP_DATA, "queue.csv"), dtype={"order_id": str, "seller_id": str})
    prod, pmeta = load_model(MODELS, FEATURE_COLUMNS, data_path=os.path.join(ROOT, "data", "processed", "ontime_model_ready.csv"))
    models = {"xgboost": (prod, float(pmeta["threshold"]))}
    ens_meta = None
    try:
        blend, ens_meta = load_ensemble(os.path.join(MODELS, "ensemble"), FEATURE_COLUMNS)
        models["ensemble"] = (blend, float(ens_meta["threshold"]))
    except EnsembleError as ex:
        print(f"No ensemble available ({ex}). Only the XGBoost probability is stored.")

    X = q[FEATURE_COLUMNS].copy()
    for c in NUMERIC:
        X[c] = X[c].astype(float)
    for key, (m, _thr) in models.items():
        q[f"p_{key}"] = np.round(m.predict_proba(X)[:, 1], 4)
        print(f"  scored the queue with {key}")

    serve = args.serve
    if serve == "auto":
        serve = "ensemble" if (ens_meta and ens_meta.get("serve_by_default")) else "xgboost"
    if serve not in models:
        raise SystemExit("The ensemble was requested but is not available. Run train_ensemble.py first.")
    model, thr = models[serve]
    p = q[f"p_{serve}"].to_numpy()
    reasons = explain_orders(model, X, top_n=3)
    q["probability"] = p
    q["risk_level"] = [risk_bucket(v) for v in p]
    q["flagged"] = p >= thr
    q["reasons_json"] = [json.dumps([{"feature": f, "label": lab, "contribution": c} for f, c, lab in r]) for r in reasons]
    q["main_driver"] = [r[0][2] if r else "" for r in reasons]
    q["suggested_action"] = [suggested_action(lvl, bool(fl), r[0][0] if r else None)
                             for lvl, fl, r in zip(q["risk_level"], q["flagged"], reasons)]
    q["priority"] = (q["probability"] * q["product_category"].map(weight_of)).round(4)
    q = q.sort_values("probability", ascending=False).reset_index(drop=True)
    q.to_csv(os.path.join(APP_DATA, "queue.csv"), index=False, float_format="%.6g")

    meta_path = os.path.join(APP_DATA, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8"))
    meta["serving"], meta["threshold"] = serve, thr
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=1)

    # the Model page and the Overview read these summaries
    dash = os.path.join(ROOT, "dashboard", "risk_summary.json")
    if serve != "xgboost" and os.path.exists(dash):
        print("Note: dashboard/risk_summary.json still describes the XGBoost bands; the Overview recomputes from the queue.")
    print(f"queue.csv updated: {len(q):,} orders, served by {serve} ({int(q.flagged.sum()):,} alerts at threshold {thr:.3f}).")


if __name__ == "__main__":
    main()
