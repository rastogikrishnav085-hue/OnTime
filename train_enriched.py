"""
train_enriched.py
------------------
Trains the enriched CHECKOUT ensemble on the REAL Olist table (data/processed/ontime_model_ready.csv,
built by build_dataset.py). No simulated data is used.

    .venv\\Scripts\\activate
    python build_dataset.py        (once, about 1 minute)
    python train_enriched.py       (about 1 hour on one CPU core; safe to stop and re-run, it resumes)
    python train_enriched.py --quick

Protocol (identical to train_ensemble.py so the numbers are comparable):
  1. Stratified 80/20 split, random_state=42: the SAME 77,176 train / 19,294 test orders as every earlier model.
  2. Hyper-parameters tuned by 3-fold CV on TRAIN only. Hand-set defaults are trial 0 of each search.
  3. 5-fold out-of-fold predictions on TRAIN; blend weights, alert threshold and the probability calibration are
     chosen on those. Nothing here sees the test rows.
  4. The test rows are scored ONCE.
  5. Forward-looking checks: (a) a strict time split (train on the oldest 80% of orders by purchase time, test on the
     newest 20%), (b) a monthly retrain-and-predict backtest.
  6. Ablation: the same LightGBM on the core 33 features only, so the gain is attributed to the new data, not the model.

Writes models/enriched/*, dashboard/enriched_report.json, data/processed/tuning_trials_{xgb,lgbm,hgb}_enriched.csv
"""
import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
warnings.filterwarnings("ignore")

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from scipy.stats import loguniform, randint, uniform               # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score        # noqa: E402

from enriched_features import EXTRA_COLUMNS, GROUPS                 # noqa: E402
from enriched_model import (ENRICHED_COLUMNS, NUMERIC_E, as_frame, build_hgb_e, build_lgbm_e, build_xgb_e)   # noqa: E402
from ensemble import (shift_odds, BlendModel, apply_calibration, choose_threshold, evaluate_scores, fit_calibration, fit_weights,   # noqa: E402
                      oof_predictions, reliability_table, save_ensemble)
from features import FEATURE_COLUMNS, TARGET_COLUMN                # noqa: E402
from modeling import stratified_split                               # noqa: E402
from tuning import random_search                                    # noqa: E402

DATA = os.path.join(HERE, "data", "processed", "ontime_model_ready.csv")
OUT_DIR = os.path.join(HERE, "models", "enriched")
REPORT = os.path.join(HERE, "dashboard", "enriched_report.json")
PROC = os.path.join(HERE, "data", "processed")
CKPT = os.path.join(HERE, "data", "interim")

XGB_SPACE = {"n_estimators": randint(300, 700), "max_depth": randint(3, 8), "learning_rate": loguniform(0.02, 0.08),
             "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.4, 0.5), "min_child_weight": randint(3, 30),
             "reg_lambda": loguniform(1, 40), "reg_alpha": loguniform(1e-3, 3)}
LGBM_SPACE = {"n_estimators": randint(400, 1100), "learning_rate": loguniform(0.012, 0.05), "num_leaves": randint(15, 64),
              "min_child_samples": randint(20, 100), "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.35, 0.5),
              "reg_lambda": loguniform(1, 50), "reg_alpha": loguniform(1e-3, 3), "max_depth": [-1, 6, 8]}
HGB_SPACE = {"learning_rate": loguniform(0.02, 0.08), "max_iter": randint(300, 700), "max_leaf_nodes": randint(15, 64),
             "min_samples_leaf": randint(20, 120), "l2_regularization": loguniform(0.5, 20), "max_depth": [None, 6, 8, 10]}


def log(m):
    print(m, flush=True)


def clean(d):
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def backtest(df, make_blend, step=2, min_train_months=6):
    """Retrain on everything strictly before a month, predict that month (every `step`-th month)."""
    m = df["order_purchase_timestamp"].dt.to_period("M")
    months = sorted(m.unique())
    rows = []
    for i in range(min_train_months, len(months), step):
        tr, te = df[m < months[i]], df[m == months[i]]
        if te[TARGET_COLUMN].nunique() < 2 or len(te) < 100:
            continue
        b = make_blend()
        b.fit(as_frame(tr).reset_index(drop=True), tr[TARGET_COLUMN].reset_index(drop=True))
        auc = roc_auc_score(te[TARGET_COLUMN], b.predict_proba(as_frame(te))[:, 1])
        rows.append({"month": str(months[i]), "auc": round(float(auc), 4), "n": int(len(te)), "late_rate": round(float(te[TARGET_COLUMN].mean()), 4)})
        log(f"    {months[i]}  AUC {auc:.4f}  (n={len(te):,})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()
    t_all = time.time()
    q = args.quick
    n_xgb, n_lgbm, n_hgb = (1, 2, 1) if q else (5, 12, 5)
    os.makedirs(CKPT, exist_ok=True)

    df = pd.read_csv(DATA, parse_dates=["order_purchase_timestamp"])
    train, test = stratified_split(df)
    Xtr, ytr = as_frame(train).reset_index(drop=True), train[TARGET_COLUMN].reset_index(drop=True)
    Xte, yte = as_frame(test), test[TARGET_COLUMN].to_numpy()
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    log(f"Train {len(train):,} / test {len(test):,}; late rate {df[TARGET_COLUMN].mean():.3%}; {len(ENRICHED_COLUMNS)} features "
        f"({len(FEATURE_COLUMNS)} core + {len(EXTRA_COLUMNS)} new)")

    # ---- 1. tuning (train only) ----
    log("\n1. Tuning (3-fold CV on the training rows)")
    best, trials = {}, {}
    for name, builder, space, n in (("xgb", build_xgb_e, XGB_SPACE, n_xgb), ("lgbm", build_lgbm_e, LGBM_SPACE, n_lgbm), ("hgb", build_hgb_e, HGB_SPACE, n_hgb)):
        pipe = builder(None, 1.0) if name != "hgb" else builder(None)
        best[name], trials[name] = random_search(pipe, space, Xtr, ytr, n_iter=n, baseline_params={}, log=log, name=name,
                                                 checkpoint=os.path.join(CKPT, f".ckpt_enriched_{name}{'_quick' if q else ''}_u.json"))
        trials[name].to_csv(os.path.join(PROC, f"tuning_trials_{name}_enriched.csv"), index=False)

    def pipes(_spw=None):
        # trained on the natural class mix; BlendModel(odds_shift=spw) maps the scores to the balanced scale of the risk bands
        return {"xgb": build_xgb_e(best["xgb"], 1.0), "lgbm": build_lgbm_e(best["lgbm"], 1.0), "hgb": build_hgb_e(best["hgb"])}

    # ---- 2. out-of-fold, weights, threshold, calibration ----
    log("\n2. Out-of-fold predictions on the training rows")
    oof_cache = os.path.join(CKPT, f".oof_enriched{'_quick' if q else ''}_u.csv")
    if os.path.exists(oof_cache):
        oof = pd.read_csv(oof_cache)
        log("  (re-using finished out-of-fold predictions)")
    else:
        oof = oof_predictions(pipes(), Xtr, ytr, cv=3 if q else 5, log=log, balance=False)
        oof.to_csv(oof_cache, index=False)
    oof = oof.apply(lambda c: shift_odds(c.to_numpy(), spw))
    weights, oof_auc = fit_weights(oof, ytr)
    blend_oof = sum(weights[n] * oof[n] for n in oof.columns)
    thr, _ = choose_threshold(ytr, blend_oof)
    cal = fit_calibration(blend_oof, ytr)
    log(f"   weights {weights}; out-of-fold blend AUC {oof_auc:.4f}; alert threshold {thr:.3f}")

    # ---- 3. test rows, scored once ----
    log("\n3. Fit on the full training set; score the test rows once")
    blend = BlendModel(pipes(), weights=weights, explainer="xgb", odds_shift=spw).fit(Xtr, ytr)
    member_p = blend.member_probabilities(Xte)
    p_blend = blend.predict_proba(Xte)[:, 1]
    table = []
    for n, label in (("xgb", "XGBoost - enriched features"), ("lgbm", "LightGBM - enriched features"), ("hgb", "Histogram gradient boosting - enriched features")):
        t_n, _ = choose_threshold(ytr, oof[n])
        table.append({"key": n, "label": label, "served": False, **clean(evaluate_scores(yte, member_p[n], t_n))})
    blend_row = {"key": "blend", "label": "Weighted ensemble (blend) - enriched features", "served": True, **clean(evaluate_scores(yte, p_blend, thr))}
    table.append(blend_row)
    for r in table:
        log(f"   {r['label']:<56s} AUC {r['roc_auc']:.4f}  PR-AUC {r['pr_auc']:.4f}  P {r['precision']:.2f}  R {r['recall']:.2f}  F1 {r['f1']:.3f}")
    chance = apply_calibration(cal, p_blend)
    rel, ece = reliability_table(yte, chance, 10)
    _, ece_raw = reliability_table(yte, p_blend, 10)
    cal_report = {"method": "isotonic regression on out-of-fold training predictions",
                  "test": {"brier_calibrated": round(float(brier_score_loss(yte, chance)), 4),
                           "base_rate_brier": round(float(brier_score_loss(yte, np.full(len(yte), yte.mean()))), 4),
                           "ece_calibrated": round(ece, 4), "ece_raw_score": round(ece_raw, 4)},
                  "reliability_calibrated": rel}

    # ---- 4. ablation: same model on core features only ----
    ablation = []
    if not args.no_ablation:
        log("\n4. Ablation (LightGBM, same split): which new data helped?")
        base_cols = FEATURE_COLUMNS

        def fit_cols(cols):
            import enriched_model as em
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OrdinalEncoder
            from lightgbm import LGBMClassifier
            num = [c for c in cols if c not in em.CATEGORICAL_E]
            pre = ColumnTransformer([("num", "passthrough", num), ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), em.CATEGORICAL_E)], sparse_threshold=0)

            class _L(LGBMClassifier):
                def fit(self, X, y, **kw):
                    kw["categorical_feature"] = list(range(len(num), len(num) + len(em.CATEGORICAL_E)))
                    return super().fit(X, y, **kw)
            p = dict(best["lgbm"]); p.update(scale_pos_weight=1.0, random_state=42, n_jobs=1, verbose=-1, subsample_freq=1)
            m = Pipeline([("pre", pre), ("clf", _L(**p))]).fit(Xtr[cols], ytr)
            return float(roc_auc_score(yte, m.predict_proba(Xte[cols])[:, 1]))
        ablation.append({"setting": "Core 33 features only", "roc_auc": round(fit_cols(base_cols), 4)})
        log(f"   core only: {ablation[-1]['roc_auc']}")
        for g, cols in GROUPS.items():
            a = round(fit_cols(base_cols + cols), 4)
            ablation.append({"setting": f"Core + {g} features", "roc_auc": a})
            log(f"   core + {g}: {a}")
        ablation.append({"setting": "Core + all new features", "roc_auc": round(fit_cols(ENRICHED_COLUMNS), 4)})

    # ---- 5. forward-looking checks ----
    time_split, backtest_rows = {}, []
    order = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    cut = int(len(order) * 0.8)
    otr, ote = order.iloc[:cut], order.iloc[cut:]
    log("\n5a. Strict time split: oldest 80% of orders -> newest 20%")
    b_t = BlendModel(pipes(), weights=weights, explainer="xgb", odds_shift=spw).fit(as_frame(otr).reset_index(drop=True), otr[TARGET_COLUMN].reset_index(drop=True))
    time_split["enriched_blend_auc"] = round(float(roc_auc_score(ote[TARGET_COLUMN], b_t.predict_proba(as_frame(ote))[:, 1])), 4)
    time_split["n_train"], time_split["n_test"] = int(len(otr)), int(len(ote))
    time_split["test_from"], time_split["test_to"] = str(ote["order_purchase_timestamp"].min().date()), str(ote["order_purchase_timestamp"].max().date())
    log(f"   enriched blend {time_split['enriched_blend_auc']}")
    if not args.no_backtest:
        log("\n5b. Monthly backtest (every second month)")
        backtest_rows = backtest(df, lambda: BlendModel(pipes(), weights=weights, explainer="xgb", odds_shift=spw), step=2)
    bt_mean = round(float(np.mean([r["auc"] for r in backtest_rows])), 4) if backtest_rows else None

    # ---- 6. save ----
    meta = save_ensemble(blend, OUT_DIR, ENRICHED_COLUMNS, thr, {k: round(v, 4) for k, v in blend_row.items() if isinstance(v, float)},
                         extra={"kind": "enriched_checkout", "tuned_params": best, "calibration": cal, "calibration_report": cal_report,
                                "oof_blend_auc": round(oof_auc, 4), "train_late_rate": round(float(ytr.mean()), 4),
                                "data": "REAL Olist only (data/raw -> data/interim -> data/processed)"})
    # feature importance (gain share of the XGBoost member, per original feature)
    import enriched_model as em
    xm = blend.members["xgb"]
    names = list(NUMERIC_E)
    for cat, cats in zip(em.CATEGORICAL_E, xm.named_steps["pre"].named_transformers_["cat"].categories_):
        names.extend([cat] * len(cats))
    imp = pd.Series(xm.named_steps["clf"].feature_importances_, index=names).groupby(level=0).sum().sort_values(ascending=False)
    importance = [{"feature": k, "label": em.ALL_LABELS.get(k, k), "share": round(float(v), 4), "new": k in EXTRA_COLUMNS} for k, v in imp.head(20).items()]
    report = {
        "generated_utc": meta["saved_at_utc"], "quick_run": bool(q), "data": "REAL Olist data only",
        "n_train": int(len(train)), "n_test": int(len(test)), "n_features": len(ENRICHED_COLUMNS), "n_core": len(FEATURE_COLUMNS), "n_new": len(EXTRA_COLUMNS),
        "protocol": ["Stratified 80/20 split (random_state 42): the same 77,176 train / 19,294 test orders as every earlier model.",
                     "Hyper-parameters tuned by 3-fold CV on train only; weights, alert threshold and calibration chosen on out-of-fold train predictions.",
                     "Test rows scored once. Every new feature uses only information available when the order is placed."],
        "table": table, "weights": weights, "threshold": round(thr, 4), "oof_blend_auc": round(oof_auc, 4),
        "ablation": ablation, "time_split": time_split, "backtest": backtest_rows, "backtest_mean_auc": bt_mean,
        "importance": importance, "calibration": cal_report,
        "tuning": {k: {"trials": int(len(v)), "best_cv_auc": float(v.cv_roc_auc.max()), "baseline_cv_auc": float(v[v.kind == "baseline"].cv_roc_auc.iloc[0]),
                       "best_params": {kk: (vv.item() if hasattr(vv, "item") else vv) for kk, vv in best[k].items()}} for k, v in trials.items()},
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"\nDone in {(time.time() - t_all) / 60:.1f} min. Blend AUC {blend_row['roc_auc']:.4f}, time-split {time_split['enriched_blend_auc']}, "
        f"monthly mean {bt_mean}. Next: python backend/add_enriched_scores.py")


if __name__ == "__main__":
    main()
