"""
calibrate_ensemble.py
----------------------
Adds an honest "estimated real chance of being late" to the ensemble.

The ensemble's score is built for ranking (class-balanced), so 0.70 does not
mean 70%. This script fits an isotonic calibration on OUT-OF-FOLD training
predictions, checks it once on the test orders, stores it in
models/ensemble/ensemble_meta.json, and writes p_calibrated into the queue.

    .venv\\Scripts\\activate
    python calibrate_ensemble.py        (about 3 minutes)
"""
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
warnings.filterwarnings("ignore")

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
from sklearn.metrics import brier_score_loss                    # noqa: E402

from ensemble import (apply_calibration, build_hgb, build_lr, build_rf, build_xgb, fit_calibration,   # noqa: E402
                      load_ensemble, oof_predictions, reliability_table, META_FILE)
from features import FEATURE_COLUMNS, TARGET_COLUMN              # noqa: E402
from modeling import NUMERIC, stratified_split                   # noqa: E402

DATA = os.path.join(HERE, "data", "processed", "ontime_model_ready.csv")
ENS_DIR = os.path.join(HERE, "models", "ensemble")
QUEUE = os.path.join(HERE, "backend", "app_data", "queue.csv")


def main():
    blend, meta = load_ensemble(ENS_DIR, FEATURE_COLUMNS)
    df = pd.read_csv(DATA, parse_dates=["order_purchase_timestamp"])
    train, test = stratified_split(df)
    Xtr, ytr = train[FEATURE_COLUMNS].reset_index(drop=True), train[TARGET_COLUMN].reset_index(drop=True)
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    builders = {"xgb": lambda: build_xgb(meta.get("tuned_xgb_params"), spw), "hgb": lambda: build_hgb(meta.get("tuned_hgb_params")),
                "lr": build_lr, "rf": build_rf}
    live = {n: builders[n]() for n, w in meta["weights"].items() if w > 0}
    print(f"Out-of-fold predictions for {list(live)} (5 folds) ...")
    oof = oof_predictions(live, Xtr, ytr, cv=5, log=lambda m: print(" ", m))
    wsum = sum(meta["weights"][n] for n in live)
    blend_oof = sum(meta["weights"][n] * oof[n] for n in live) / wsum
    cal = fit_calibration(blend_oof, ytr)

    p_test = blend.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    yte = test[TARGET_COLUMN].to_numpy()
    chance = apply_calibration(cal, p_test)
    rows, ece = reliability_table(yte, chance, bins=10)
    raw_rows, raw_gap = reliability_table(yte, p_test, bins=10)
    report = {"method": "isotonic regression on out-of-fold training predictions", "n_points": len(cal["x"]),
              "test": {"brier_calibrated": round(float(brier_score_loss(yte, chance)), 4),
                       "brier_if_score_were_a_probability": round(float(brier_score_loss(yte, p_test)), 4),
                       "base_rate_brier": round(float(brier_score_loss(yte, np.full(len(yte), yte.mean()))), 4),
                       "ece_calibrated": round(ece, 4), "ece_raw_score": round(raw_gap, 4)},
              "reliability_calibrated": rows, "reliability_raw": raw_rows}
    print(json.dumps(report["test"], indent=1))

    meta["calibration"] = cal
    meta["calibration_report"] = report
    with open(os.path.join(ENS_DIR, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if os.path.exists(QUEUE):
        q = pd.read_csv(QUEUE, dtype={"order_id": str, "seller_id": str})
        if "p_ensemble" in q.columns:
            q["p_calibrated"] = np.round(apply_calibration(cal, q["p_ensemble"].to_numpy()), 4)
            q.to_csv(QUEUE, index=False, float_format="%.6g")
            print(f"queue.csv: added p_calibrated for {len(q):,} orders")
        else:
            print("queue.csv has no p_ensemble yet; run backend/add_second_opinion.py, then this script again.")


if __name__ == "__main__":
    main()
