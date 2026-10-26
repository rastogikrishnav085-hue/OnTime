"""
export_risk_summary.py
-----------------------
Runs the held-out test set through the production scorer (src/scoring.py)
and writes the two JSON files the dashboard's "Delivery Risk" page reads:

    dashboard/top_risk_orders.json   12 highest-risk test orders, with their
                                     main risk driver, suggested action and
                                     whether they actually turned out late
    dashboard/risk_summary.json      orders per risk band and how many in each
                                     band were really late, the share of late
                                     orders caught by the alert, and the most
                                     common drivers / actions among flagged orders

Run from the project root, then rebuild the HTML:

    python export_risk_summary.py
    python build_dashboard.py
"""
import json
import os
import sys
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from features import FEATURE_COLUMNS                      # noqa: E402
from modeling import stratified_split                     # noqa: E402
from artifacts import load_model, load_json               # noqa: E402
from inference_validation import load_reference_stats     # noqa: E402
from scoring import score_orders                          # noqa: E402

M = os.path.join(HERE, "models")
OUT = os.path.join(HERE, "dashboard")


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, meta = load_model(M, FEATURE_COLUMNS, data_path=os.path.join(HERE, "data", "processed", "ontime_model_ready.csv"))
    stats = load_reference_stats(os.path.join(M, "reference_stats.json"))
    lookup = load_json(os.path.join(M, "rate_lookup.json"))

    df = pd.read_csv(os.path.join(HERE, "data", "processed", "ontime_model_ready.csv"),
                     parse_dates=["order_purchase_timestamp"])
    test = stratified_split(df)[1].reset_index(drop=True)
    # The test set already carries its leak-free historical rates, so they are
    # passed straight through rather than re-looked-up from the full history.
    out = score_orders(test, model, meta["threshold"], stats, lookup)
    t = test.assign(prob=out.probability.values, level=out.risk_level.values,
                    flagged=out.flagged.values.astype(bool), reasons=out.reasons.values,
                    action=out.suggested_action.values)
    driver = t.reasons.str.split(r" \(\+").str[0]

    top = t.assign(driver=driver).sort_values("prob", ascending=False).head(12)
    rows = [{
        "order_id": r.order_id[:10] + "…", "prob": round(r.prob * 100, 1), "level": r.level,
        "category": r.product_category, "customer_state": r.customer_state,
        "seller_state": r.seller_state, "value": round(r.total_order_value, 2),
        "distance": None if pd.isna(r.seller_customer_distance_km) else round(r.seller_customer_distance_km),
        "driver": r.driver or "", "action": r.action, "actually_late": int(r.is_delayed),
    } for r in top.itertuples()]

    flagged = t[t.flagged]
    summary = {
        "bands": [{"level": b, "n": int((t.level == b).sum()),
                   "late_rate": round(float(t.loc[t.level == b, "is_delayed"].mean()) * 100, 1)}
                  for b in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]],
        "n_test": int(len(t)),
        "n_flagged": int(len(flagged)),
        "flagged_share": round(len(flagged) / len(t) * 100, 1),
        "flagged_late_rate": round(float(flagged.is_delayed.mean()) * 100, 1),
        "late_orders_caught": round(float(flagged.is_delayed.sum() / t.is_delayed.sum()) * 100, 1),
        "base_late_rate": round(float(t.is_delayed.mean()) * 100, 1),
        "top_drivers": [{"driver": k, "n": int(v)}
                        for k, v in driver[t.flagged].value_counts().head(8).items()],
        "actions": [{"action": k, "n": int(v)} for k, v in
                    flagged.action.str.replace("URGENT: ", "", regex=False).value_counts().head(8).items()],
        "n_urgent": int(flagged.action.str.startswith("URGENT").sum()),
    }
    with open(os.path.join(OUT, "top_risk_orders.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "risk_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Flagged {summary['n_flagged']} of {summary['n_test']} test orders "
          f"({summary['flagged_share']}%); {summary['flagged_late_rate']}% of them were really late "
          f"vs {summary['base_late_rate']}% overall; alert catches {summary['late_orders_caught']}% of late orders.")
    for b in summary["bands"]:
        print(f"  {b['level']:<9}{b['n']:>7} orders   {b['late_rate']:>5}% late")


if __name__ == "__main__":
    main()
