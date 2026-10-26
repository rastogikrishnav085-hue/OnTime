"""
Precompute per-order reasons (exact TreeSHAP, top 3) for the checkout-time models that can score the queue.

Why: switching the scoring model in Settings re-scores every queue view. The probabilities are stored in
queue.csv (p_xgboost, p_ensemble, p_enriched); the "why" behind each score takes ~1-2 minutes per model to
compute, so it is computed once here and stored in backend/app_data/queue_explanations.csv.gz.

Run after rebuild_app_data / add_second_opinion:   python backend/add_view_explanations.py
"""
import os
import sys
import warnings

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from backend.app.engine import Engine, APP_DATA  # noqa: E402
import json  # noqa: E402


def main():
    e = Engine()
    out = pd.DataFrame({"order_id": e._base_queue["order_id"].astype(str)})
    for key in ("xgboost", "ensemble"):
        if key not in e.models:
            print(f"  skip {key}: model not loaded")
            continue
        print(f"  explaining the queue with {key} ...", flush=True)
        why = e._explain_fast(e.models[key][0], e._base_queue)
        out[f"reasons_{key}"] = [json.dumps([{"feature": f, "label": lab, "contribution": c} for f, c, lab in r]) for r in why]
    path = os.path.join(APP_DATA, "queue_explanations.csv.gz")
    out.to_csv(path, index=False, compression="gzip")
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB, {len(out):,} orders)")


if __name__ == "__main__":
    main()
