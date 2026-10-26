"""
prepare_app_data.py
--------------------
Builds the compact data files the web app serves, from the project's
processed data and saved model. Run once after the notebooks (the zip
ships with the output already built):

    python backend/prepare_app_data.py

Writes backend/app_data/:
    queue.csv       the 19,294 held-out test orders, scored by the saved
                    model through src/scoring.py: probability, band, alert
                    flag, top-3 TreeSHAP reasons, suggested action, plus
                    the order's features and what actually happened
                    (kept separate from anything the model saw).
    insights.json   aggregates over all 96,470 delivered orders for the
                    Insights page (state, category, distance, promise
                    window, weekday, split orders, seller watchlist).
    meta.json       dropdown values and a "typical order" (training-set
                    medians / most common values) used to pre-fill the
                    order simulator.

The queue uses the newest chronological 20% of orders. The enriched production model is trained on the oldest 80%, so this queue is a genuine later-period holdout for that model.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import FEATURE_COLUMNS, TARGET_COLUMN               # noqa: E402
from modeling import stratified_split, NUMERIC, CATEGORICAL       # noqa: E402
from artifacts import load_model, load_json                       # noqa: E402
from inference_validation import load_reference_stats             # noqa: E402
from scoring import score_orders                                  # noqa: E402
from prediction import explain_orders, FEATURE_LABELS             # noqa: E402
from simulation import simulate_logistics, region, sensitivity_test, SIMULATED_COLUMNS, HUB_OF_REGION  # noqa: E402
from clustering import fit_clusters, profile_clusters, assign, PEAK_MONTHS  # noqa: E402
from categories import FOCUS_CATEGORIES, ROADMAP_NOTE, tier_of, weight_of  # noqa: E402

OUT = os.path.join(ROOT, "backend", "app_data")
PROC = os.path.join(ROOT, "data", "processed")


def rate_table(df, col, min_n):
    g = df.groupby(col)[TARGET_COLUMN].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_n].sort_values("mean", ascending=False)
    return [{"key": str(r[col]), "rate": round(float(r["mean"]) * 100, 2), "n": int(r["size"])} for _, r in g.iterrows()]


def bucket_table(df, series, bins, labels):
    b = pd.cut(series, bins, labels=labels)
    g = df.groupby(b, observed=True)[TARGET_COLUMN].agg(["mean", "size"])
    return [{"key": str(k), "rate": round(float(v["mean"]) * 100, 2), "n": int(v["size"])} for k, v in g.iterrows()]


def main():
    os.makedirs(OUT, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, meta = load_model(os.path.join(ROOT, "models"), FEATURE_COLUMNS,
                                 data_path=os.path.join(PROC, "ontime_model_ready.csv"))
    stats = load_reference_stats(os.path.join(ROOT, "models", "reference_stats.json"))
    lookup = load_json(os.path.join(ROOT, "models", "rate_lookup.json"))

    ready = pd.read_csv(os.path.join(PROC, "ontime_model_ready.csv"), parse_dates=["order_purchase_timestamp"])
    joined_path = os.path.join(ROOT, "data", "interim", "olist_joined.csv")
    if not os.path.exists(joined_path):
        raise SystemExit("Joined source table is missing; run python build_dataset.py first.")
    joined = pd.read_csv(joined_path,
                         parse_dates=["order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date", "order_delivered_carrier_date"])
    ready = ready.sort_values("order_purchase_timestamp", kind="stable").reset_index(drop=True)
    # The newest 20% is the strict chronological holdout used by the enriched model.
    cutoff = int(len(ready) * 0.80)
    train, test = ready.iloc[:cutoff].copy(), ready.iloc[cutoff:].copy()
    test = test.reset_index(drop=True)

    # ---------------- queue ----------------
    scored = score_orders(test, model, meta["threshold"], stats, lookup)
    X = test[FEATURE_COLUMNS].copy()
    for c in NUMERIC:
        X[c] = X[c].astype(float)
    reasons = explain_orders(model, X, top_n=3)  # every order, so the detail view is instant

    extra = joined[["order_id", "primary_seller_id", "customer_city", "seller_city",
                    "order_estimated_delivery_date", "order_delivered_customer_date", "delay_days"]]
    q = test.merge(extra, on="order_id", how="left")
    q = q.rename(columns={"primary_seller_id": "seller_id", "order_purchase_timestamp": "purchased_at",
                          "order_estimated_delivery_date": "promised_by", "order_delivered_customer_date": "delivered_at"})
    q["probability"] = scored["probability"].values
    q["risk_level"] = scored["risk_level"].values
    q["flagged"] = scored["flagged"].astype(bool).values
    q["suggested_action"] = scored["suggested_action"].values
    q["confidence"] = scored["confidence"].values
    q["reasons_json"] = [json.dumps([{"feature": f, "label": lab, "contribution": c} for f, c, lab in r]) for r in reasons]
    q["main_driver"] = [r[0][2] if r else "" for r in reasons]

    # ---------------- all delivered orders, enriched ----------------
    d = ready.merge(joined[["order_id", "primary_seller_id", "customer_city", "seller_city", "delay_days",
                            "order_delivered_customer_date"]].rename(columns={"primary_seller_id": "seller_id"}),
                    on="order_id", how="left")
    d["actual_days"] = (d["order_delivered_customer_date"] - d["order_purchase_timestamp"]).dt.total_seconds() / 86400
    d["customer_region"] = d["customer_state"].map(region)
    d["seller_region"] = d["seller_state"].map(region)
    d = pd.concat([d, simulate_logistics(d)], axis=1)

    # ---------------- patterns (clusters of late orders) ----------------
    late = d[d[TARGET_COLUMN] == 1].reset_index(drop=True)
    labels, cparams = fit_clusters(late)
    profiles = profile_clusters(late, labels, cparams)
    id_to_rank = {p["id"]: p["rank"] for p in profiles}

    # queue: simulated fields, regions, pattern, focus tier, priority
    q = q.merge(d[["order_id", "customer_region", "seller_region", "actual_days"] + SIMULATED_COLUMNS], on="order_id", how="left")
    q["pattern"] = [id_to_rank[int(c)] for c in assign(q, cparams)]
    q["focus_tier"] = q["product_category"].map(tier_of)
    q["priority"] = (q["probability"] * q["product_category"].map(weight_of)).round(4)
    q = q.sort_values("probability", ascending=False).reset_index(drop=True)
    q.to_csv(os.path.join(OUT, "queue.csv"), index=False, float_format="%.6g")

    with open(os.path.join(OUT, "patterns.json"), "w", encoding="utf-8") as f:
        json.dump({"n_late": int(len(late)), "k": cparams["k"], "silhouette_by_k": cparams["silhouette_by_k"],
                   "params": {k: cparams[k] for k in ("mean", "std", "centers")},
                   "id_to_rank": {str(k): v for k, v in id_to_rank.items()},
                   "patterns": profiles}, f, indent=1, ensure_ascii=False)

    # ---------------- focus categories ----------------
    late_with_pattern = late.assign(pattern=[id_to_rank[int(c)] for c in labels])
    cats = {}
    base = float(d[TARGET_COLUMN].mean()) * 100
    for key, c in FOCUS_CATEGORIES.items():
        g = d[d.product_category == key]
        gl = late_with_pattern[late_with_pattern.product_category == key]
        peak = g[g.purchase_month.isin(PEAK_MONTHS)]
        off = g[~g.purchase_month.isin(PEAK_MONTHS)]
        monthly = (g.assign(m=g.order_purchase_timestamp.dt.strftime("%Y-%m")).groupby("m")[TARGET_COLUMN]
                   .agg(["mean", "size"]).query("size >= 30"))
        st = g.groupby("customer_state")[TARGET_COLUMN].agg(["mean", "size"]).query("size >= 40").sort_values("mean", ascending=False)
        cats[key] = {**c, "key": key, "stats": {
            "n": int(len(g)), "late_rate": round(float(g[TARGET_COLUMN].mean()) * 100, 2), "platform_rate": round(base, 2),
            "avg_days_late": round(float(g.loc[g[TARGET_COLUMN] == 1, "delay_days"].mean()), 1),
            "late_rate_peak": round(float(peak[TARGET_COLUMN].mean()) * 100, 1) if len(peak) else None,
            "late_rate_offpeak": round(float(off[TARGET_COLUMN].mean()) * 100, 1) if len(off) else None,
            "median_value": round(float(g.total_order_value.median()), 2),
            "median_weight_kg": round(float(g.total_weight_g.median()) / 1000, 2),
            "monthly": [{"month": m, "rate": round(float(r["mean"]) * 100, 2), "n": int(r["size"])} for m, r in monthly.iterrows()],
            "worst_states": [{"key": s, "rate": round(float(r["mean"]) * 100, 1), "n": int(r["size"])} for s, r in st.head(5).iterrows()],
            "pattern_mix": [{"pattern": int(k), "share": round(float(v) * 100, 1)}
                            for k, v in gl.pattern.value_counts(normalize=True).sort_index().items()],
        }}
    with open(os.path.join(OUT, "categories.json"), "w", encoding="utf-8") as f:
        json.dump({"focus": cats, "roadmap_note": ROADMAP_NOTE}, f, indent=1, ensure_ascii=False)

    # ---------------- simulated network ----------------
    def by(col, order=None):
        g = d.groupby(col)[TARGET_COLUMN].agg(["mean", "size"])
        if order:
            g = g.reindex([o for o in order if o in g.index])
        return [{"key": str(k), "rate": round(float(r["mean"]) * 100, 2), "n": int(r["size"])} for k, r in g.iterrows()]
    lanes = (d.groupby(["seller_region", "customer_region"])[TARGET_COLUMN].agg(["mean", "size"]).reset_index()
             .query("size >= 100").sort_values("mean", ascending=False))
    print("Running the simulated-data sensitivity test (two model fits)...")
    sens = sensitivity_test(ready, simulate_logistics(ready.merge(joined[["order_id", "primary_seller_id"]]
                                                                 .rename(columns={"primary_seller_id": "seller_id"}), on="order_id", how="left")),
                            FEATURE_COLUMNS, TARGET_COLUMN, meta["xgb_params"], stratified_split)
    network = {
        "delivery_model": by("sim_delivery_model", ["Direct (point-to-point)", "Regional hub", "Hub-and-spoke (national)"]),
        "transfers": by("sim_hub_transfers"),
        "shipping_mode": by("sim_shipping_mode", ["Express", "Standard", "Economy"]),
        "carrier": by("sim_carrier"),
        "last_mile_hub": by("sim_last_mile_hub"),
        "lanes": [{"from": r.seller_region, "to": r.customer_region, "rate": round(float(r["mean"]) * 100, 2), "n": int(r["size"]),
                   "hubs": f"{HUB_OF_REGION[r.seller_region]} > {HUB_OF_REGION[r.customer_region]}"} for _, r in lanes.iterrows()],
        "sensitivity": sens,
    }
    with open(os.path.join(OUT, "network.json"), "w", encoding="utf-8") as f:
        json.dump(network, f, indent=1, ensure_ascii=False)
    print(f"patterns: {cparams['k']} | sensitivity: {sens}")

    # ---------------- insights (all delivered orders) ----------------
    full = ready.merge(joined[["order_id", "primary_seller_id", "seller_city"]], on="order_id", how="left")
    eda = json.load(open(os.path.join(ROOT, "dashboard", "eda_summary.json"), encoding="utf-8"))
    sellers = (full.groupby("primary_seller_id")
               .agg(rate=(TARGET_COLUMN, "mean"), n=(TARGET_COLUMN, "size"), state=("seller_state", "first"),
                    city=("seller_city", "first"),
                    category=("product_category", lambda s: s.value_counts().index[0]))
               .reset_index())
    watch = sellers[sellers.n >= 30].sort_values(["rate", "n"], ascending=[False, False]).head(25)
    dow = full.groupby("purchase_dow")[TARGET_COLUMN].agg(["mean", "size"])
    insights = {
        "n_orders": int(len(full)),
        "base_rate": round(float(full[TARGET_COLUMN].mean()) * 100, 2),
        "monthly": eda["monthly"],
        "state": rate_table(full, "customer_state", 100),
        "seller_state": rate_table(full, "seller_state", 100),
        "category": rate_table(full, "product_category", 150),
        "distance": bucket_table(full, full.seller_customer_distance_km, [-1, 100, 300, 700, 1500, 1e9],
                                 ["< 100 km", "100–300 km", "300–700 km", "700–1,500 km", "1,500 km +"]),
        "promise": bucket_table(full, full.promised_delivery_days, [-1, 10, 20, 30, 45, 1e9],
                                ["< 10 days", "10–20", "20–30", "30–45", "45 days +"]),
        "split": bucket_table(full, full.n_distinct_sellers, [0, 1, 2, 99], ["1 seller", "2 sellers", "3 +"]),
        "same_state": [{"key": "Same state", "rate": round(float(full[full.same_state == 1][TARGET_COLUMN].mean()) * 100, 2),
                        "n": int((full.same_state == 1).sum())},
                       {"key": "Different state", "rate": round(float(full[full.same_state == 0][TARGET_COLUMN].mean()) * 100, 2),
                        "n": int((full.same_state == 0).sum())}],
        "weekday": [{"key": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(k)], "rate": round(float(v["mean"]) * 100, 2),
                     "n": int(v["size"])} for k, v in dow.iterrows()],
        "seller_watchlist": [{"seller_id": r.primary_seller_id, "rate": round(r.rate * 100, 1), "n": int(r.n),
                              "state": r.state, "city": str(r.city).title(), "category": r.category} for r in watch.itertuples()],
        "n_sellers_30plus": int((sellers.n >= 30).sum()),
    }
    with open(os.path.join(OUT, "insights.json"), "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=1, ensure_ascii=False)

    # ---------------- meta ----------------
    typical = {}
    for c in NUMERIC:
        if c.endswith("historical_delay_rate"):
            continue
        typical[c] = float(np.round(train[c].median(), 3))
    for c in CATEGORICAL:
        typical[c] = str(train[c].mode().iloc[0])
    meta_out = {
        "threshold": meta["threshold"],
        "bands": [{"level": "LOW", "max": 0.30}, {"level": "MEDIUM", "max": 0.60}, {"level": "HIGH", "max": 0.80}, {"level": "CRITICAL", "max": 1.0}],
        "states": sorted(set(train.customer_state) | set(train.seller_state)),
        "categories": sorted(train.product_category.unique().tolist()),
        "payment_types": sorted(train.payment_type.unique().tolist()),
        "typical_order": typical,
        "feature_labels": FEATURE_LABELS,
        "n_queue": int(len(q)),
        "model_saved_at": meta.get("saved_at_utc"),
    }
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=1)
    print(f"queue.csv: {len(q):,} orders ({int(q.flagged.sum()):,} flagged) | insights.json | meta.json -> {OUT}")


if __name__ == "__main__":
    main()
