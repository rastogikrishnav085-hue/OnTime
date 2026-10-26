"""
train_tracking_model.py - third score: "the order is in transit; what do the scans, failed attempts, weather and
strikes say?" Trained on the SIMULATED tracking data set (make_tracking_dataset.py), so every figure is SIMULATED.

Snapshots: every order that is still undelivered AND not yet past its promise at day 2, 4, ... 28 after purchase
(an order already past its promise is late already, so scoring it is not forecasting).
Split: oldest 65% of orders by purchase time train, next 15% validate (threshold + early stopping), newest 20% test.
Baseline: same snapshots, same model, but WITHOUT scans, failed attempts, weather or strikes (real data only).
Run:  .venv\\Scripts\\activate   then   python train_tracking_model.py     (about 10-15 minutes on one core)
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from make_tracking_dataset import OUT, load_orders                       # noqa: E402
from tracking_features import (load_context, TIME_FEATURES, TRACK_FEATURES, WEATHER_FEATURES,  # noqa: E402
                                   build_features)

DAYS = list(range(2, 30, 2))
PARAMS = dict(learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
              tree_method="hist", max_bin=128, enable_categorical=True, n_jobs=1)


def main():
    meta0 = json.load(open(os.path.join(ROOT, "models", "model_metadata.json")))
    base, cats = meta0["feature_columns"], meta0["categorical"]
    d = load_orders()
    r = pd.read_csv(os.path.join(ROOT, "data", "processed", "ontime_model_ready.csv"),
                    usecols=["order_id"] + base).drop_duplicates("order_id")
    d = d.drop(columns=[c for c in base if c in d.columns and c != "order_id"]).merge(r, on="order_id")
    d = d.rename(columns={"order_purchase_timestamp": "purchased_at", "order_estimated_delivery_date": "promised_by",
                          "order_delivered_customer_date": "delivered_at"})
    ev = pd.read_csv(os.path.join(OUT, "tracking_events_simulated.csv"), parse_dates=["event_time"])
    ctx = load_context(OUT)
    categories = {c: sorted(d[c].astype(str).unique().tolist()) for c in cats}

    parts = []
    for k in DAYS:
        as_of = d["purchased_at"] + pd.Timedelta(days=k)
        keep = (d["delivered_at"] > as_of) & (d["promised_by"] > as_of)
        s = d[keep]
        f = build_features(s, ev, ctx, as_of[keep])
        x = pd.concat([s[["order_id", "purchased_at", "is_delayed"] + base], f], axis=1)
        x["snapshot_day"] = k
        parts.append(x)
        print(f"day {k:2d}: {len(x):,} in-transit orders, {x['is_delayed'].mean():.1%} late", flush=True)
    X = pd.concat(parts, ignore_index=True)
    y = X["is_delayed"].astype(int).values
    print(f"{len(X):,} snapshots from {X['order_id'].nunique():,} orders", flush=True)

    cut = d["purchased_at"].quantile([0.65, 0.80]).values
    tr = (X["purchased_at"] <= cut[0]).values
    va = ((X["purchased_at"] > cut[0]) & (X["purchased_at"] <= cut[1])).values
    te = (X["purchased_at"] > cut[1]).values

    def frame(cols):
        F = X[cols].copy()
        for c in cats:
            F[c] = pd.Categorical(F[c].astype(str), categories=categories[c])
        return F

    sets = {"baseline": base + TIME_FEATURES,
            "tracking": base + TIME_FEATURES + TRACK_FEATURES + WEATHER_FEATURES}
    out = {}
    for name, cols in sets.items():
        F = frame(cols)
        m = xgb.XGBClassifier(n_estimators=500, eval_metric="auc", early_stopping_rounds=40, **PARAMS)
        m.fit(F[tr], y[tr], eval_set=[(F[va], y[va])], verbose=False)
        pv, pt = m.predict_proba(F[va])[:, 1], m.predict_proba(F[te])[:, 1]
        ts = np.linspace(0.02, 0.9, 89)
        thr = float(ts[np.argmax([f1_score(y[va], pv >= t) for t in ts])])
        a = pt >= thr
        days = X.loc[te, "snapshot_day"].values
        per_day = {int(k): float(roc_auc_score(y[te][days == k], pt[days == k])) for k in DAYS
                   if (days == k).sum() > 200 and 0 < y[te][days == k].sum() < (days == k).sum()}
        res = dict(roc_auc=float(roc_auc_score(y[te], pt)), pr_auc=float(average_precision_score(y[te], pt)),
                   per_day_mean_auc=float(np.mean(list(per_day.values()))), per_day_auc=per_day, threshold=thr,
                   precision=float(precision_score(y[te], a)), recall=float(recall_score(y[te], a)),
                   f1=float(f1_score(y[te], a)), accuracy=float(accuracy_score(y[te], a)),
                   late_rate_test=float(y[te].mean()), n_test=int(te.sum()), n_train=int(tr.sum()),
                   best_iteration=int(m.best_iteration))
        out[name] = (res, m, cols)
        print(name, {k: round(v, 3) for k, v in res.items() if isinstance(v, float)}, flush=True)

    # Month-by-month: train on every earlier month (up to 150k rows), test on that month. Every second month.
    ym = X["purchased_at"].dt.to_period("M")
    bt = {"baseline": {}, "tracking": {}}
    rng = np.random.default_rng(0)
    for mth in ["2017-05", "2017-07", "2017-09", "2017-11", "2018-01", "2018-03", "2018-05"]:
        trm, tem = (ym < mth).values, (ym == mth).values
        if tem.sum() < 500 or y[tem].sum() < 20:
            continue
        idx = np.where(trm)[0]
        idx = rng.choice(idx, min(len(idx), 150000), replace=False)
        for name, cols in sets.items():
            F = frame(cols)
            m = xgb.XGBClassifier(n_estimators=120, **{**PARAMS, "learning_rate": 0.1, "max_depth": 5})
            m.fit(F.iloc[idx], y[idx])
            bt[name][mth] = float(roc_auc_score(y[tem], m.predict_proba(F[tem])[:, 1]))
        print(mth, {k: round(v[mth], 3) for k, v in bt.items()}, flush=True)
    bt = {k: {"mean_auc": float(np.mean(list(v.values()))), "months": len(v), "by_month": v} for k, v in bt.items()}

    # Final tracking model: refit on all snapshots.
    res, _, cols = out["tracking"]
    F = frame(cols)
    final = xgb.XGBClassifier(n_estimators=int(res["best_iteration"] * 1.15) + 1, eval_metric="auc", **PARAMS)
    final.fit(F, y)
    final.save_model(os.path.join(ROOT, "models", "tracking_model.json"))
    gain = final.get_booster().get_score(importance_type="gain")
    imp = pd.Series({c: gain.get(c, 0.0) for c in cols}).sort_values(ascending=False)
    imp = imp / imp.sum()
    json.dump({
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "simulated": True,
        "stage": "in transit, with simulated tracking data", "feature_columns": cols,
        "tracking_features": TRACK_FEATURES, "weather_features": WEATHER_FEATURES, "time_features": TIME_FEATURES,
        "categorical": categories, "threshold": res["threshold"], "snapshot_days": DAYS,
        "with_tracking": res, "without_tracking": out["baseline"][0], "monthly_backtest": bt,
        "top_features": {k: float(v) for k, v in imp.head(10).items()},
        "n_orders": int(X["order_id"].nunique()), "n_snapshots": int(len(X)),
        "note": ("SIMULATED. The scans, failed attempts, weather and strikes are invented (see DATA_DICTIONARY.md); late "
                 "orders were made more likely to show them, so these scores describe the simulation, not real Olist. "
                 "The 'without tracking' row uses real data only."),
    }, open(os.path.join(ROOT, "models", "tracking_meta.json"), "w"), indent=1)
    print("saved models/tracking_model.json and models/tracking_meta.json")


if __name__ == "__main__":
    main()
