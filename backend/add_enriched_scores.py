"""
add_enriched_scores.py
-----------------------
Scores the app's order queue (the 19,294 unseen test orders) with the ENRICHED checkout model and stores
the result in backend/app_data/queue.csv:

    p_enriched            the enriched blend's probability (balanced scale, same bands as the other models)
    p_calibrated          the estimated REAL chance of being late for the serving model (isotonic, fitted on
                          out-of-fold TRAINING predictions only)
    + the 37 extra feature columns (the exact values computed at each order's own purchase time), so the
      queue can be re-scored later without the raw Olist files.

    .venv\\Scripts\\activate
    python backend/add_enriched_scores.py                  # serve whatever the honest evaluation decided
    python backend/add_enriched_scores.py --serve enriched
    python backend/add_enriched_scores.py --serve xgboost  # or: ensemble

The decision comes from dashboard/chronological_report.json (written by evaluate_chronological.py):
the enriched model is served by default only if it wins on the untouched, strictly time-ordered test set.
Actual outcomes (is_delayed, delivered_at, ...) stay in the file for evaluation only; they are never part of
the model input.

Needs: models/enriched/ (train_enriched.py), data/processed/ontime_model_ready.csv (build_dataset.py).
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

from categories import weight_of                                 # noqa: E402
from enriched_features import EXTRA_COLUMNS                      # noqa: E402
from enriched_model import ENRICHED_COLUMNS, EnrichedModel, action_for, as_frame, explain_rows   # noqa: E402
from ensemble import apply_calibration, load_ensemble            # noqa: E402
from features import FEATURE_COLUMNS                             # noqa: E402
from modeling import risk_bucket, stratified_split               # noqa: E402

APP_DATA = os.path.join(ROOT, "backend", "app_data")
MODELS = os.path.join(ROOT, "models")
DASH = os.path.join(ROOT, "dashboard")
DATA = os.path.join(ROOT, "data", "processed", "ontime_model_ready.csv")


def decided_serving():
    """What the honest (chronological) evaluation says to serve; the original XGBoost if there is no verdict."""
    path = os.path.join(DASH, "chronological_report.json")
    if not os.path.exists(path):
        return "xgboost", "no chronological_report.json yet (run evaluate_chronological.py)"
    d = json.load(open(path, encoding="utf-8")).get("decision", {})
    return d.get("serve", "xgboost"), d.get("reason", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", choices=["auto", "xgboost", "ensemble", "enriched"], default="auto")
    args = ap.parse_args()

    q = pd.read_csv(os.path.join(APP_DATA, "queue.csv"), dtype={"order_id": str, "seller_id": str})
    table = pd.read_csv(DATA, dtype={"order_id": str}, parse_dates=["order_purchase_timestamp"])

    # The queue must be exactly the newest chronological 20% held-out block.
    table = table.sort_values("order_purchase_timestamp", kind="stable").reset_index(drop=True)
    test = table.iloc[int(len(table) * 0.80):]
    if set(q["order_id"]) != set(test["order_id"]):
        raise SystemExit("The queue is not the newest chronological 20% of data/processed/ontime_model_ready.csv. "
                         "Rebuild it first: python backend/prepare_app_data.py")
    ex = table[["order_id"] + EXTRA_COLUMNS]
    q = q.drop(columns=[c for c in EXTRA_COLUMNS if c in q.columns]).merge(ex, on="order_id", how="left", validate="1:1")
    core_check = table.set_index("order_id").loc[q["order_id"], FEATURE_COLUMNS].reset_index(drop=True)
    for c in FEATURE_COLUMNS:
        if q[c].dtype.kind in "fi":
            assert np.allclose(q[c].astype(float), core_check[c].astype(float), rtol=1e-4, atol=1e-5, equal_nan=True), \
                f"queue column {c} differs from the model-ready table"

    blend, meta = load_ensemble(os.path.join(MODELS, "enriched"), ENRICHED_COLUMNS)
    model = EnrichedModel(blend, os.path.join(MODELS, "enriched", "context.json"))
    thr = float(meta["threshold"])
    X = as_frame(q)                                   # only feature columns: no outcome column can reach the model
    p = np.asarray(model.predict_proba(X)[:, 1])
    q["p_enriched"] = np.round(p, 4)
    chance = np.round(apply_calibration(meta["calibration"], p), 4)
    # A probability calibration is only exposed if it passes the predeclared unseen-test checks.
    # The current Olist holdout shows temporal calibration drift, so keep the calibrated chance hidden.
    cal_report = meta.get("calibration_report", {})
    test_base = float(meta.get("split", {}).get("test_late_rate", 0.0))
    base_brier = test_base * (1.0 - test_base)
    calibration_acceptable = (float(cal_report.get("test_ece", 1.0)) <= 0.03 and
                              float(cal_report.get("test_brier", 1.0)) <= base_brier)
    if not calibration_acceptable:
        chance[:] = np.nan
        app_meta_path = os.path.join(APP_DATA, "meta.json")
        # The setting is updated below after loading app_meta; retain this status for the UI/API.
    print(f"scored {len(q):,} queue orders with the enriched model (threshold {thr:.3f}, {int((p >= thr).sum()):,} alerts)")

    serve, why = (args.serve, "set on the command line") if args.serve != "auto" else decided_serving()
    print(f"serving: {serve}  ({why})")
    if "p_calibrated" in q.columns and "p_calibrated_ensemble" not in q.columns:
        q["p_calibrated_ensemble"] = q["p_calibrated"]

    meta_path = os.path.join(APP_DATA, "meta.json")
    app_meta = json.load(open(meta_path, encoding="utf-8"))
    app_meta["calibration_status"] = "validated" if calibration_acceptable else "warning: calibration drift on chronological holdout; use risk ranking, not literal probability"
    if serve == "enriched":
        reasons = explain_rows(blend, X, top_n=3)
        q["probability"] = np.round(p, 4)
        q["risk_level"] = [risk_bucket(v) for v in p]
        q["flagged"] = p >= thr
        q["reasons_json"] = [json.dumps([{"feature": f, "label": lab, "contribution": c} for f, c, lab in r]) for r in reasons]
        q["main_driver"] = [r[0][2] if r else "" for r in reasons]
        q["suggested_action"] = [action_for(lvl, bool(fl), r[0][0] if r else None)
                                 for lvl, fl, r in zip(q["risk_level"], q["flagged"], reasons)]
        q["priority"] = (q["probability"] * q["product_category"].map(weight_of)).round(4)
        q["p_calibrated"] = chance
        q = q.sort_values("probability", ascending=False).reset_index(drop=True)
        app_meta["serving"], app_meta["threshold"] = "enriched", thr
        for key in ("p_xgboost", "p_ensemble"):
            if key not in q.columns:
                print(f"Note: {key} is missing from the queue; run backend/add_second_opinion.py for the second opinion.")
    else:
        if app_meta.get("serving") == "enriched":
            raise SystemExit(f"The queue is currently served by the enriched model. To switch back to {serve}, run "
                             f"python backend/add_second_opinion.py --serve {serve}  (it rescores the queue), then run this script again.")
        print(f"Queue stays served by {app_meta.get('serving', 'xgboost')}; the enriched probability is stored as p_enriched (second opinion).")
    q.to_csv(os.path.join(APP_DATA, "queue.csv"), index=False, float_format="%.6g")
    json.dump(app_meta, open(meta_path, "w", encoding="utf-8"), indent=1)
    late = q["is_delayed"].astype(int)
    print(f"queue.csv updated ({len(q):,} orders). Actual late rate {late.mean():.3%}; "
          f"alerts catch {float((q['p_enriched'] >= thr)[late == 1].mean()):.1%} of late orders (enriched threshold).")


if __name__ == "__main__":
    main()
