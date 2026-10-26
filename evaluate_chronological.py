"""
evaluate_chronological.py
--------------------------
The PRIMARY, honest evaluation of the checkout model: strictly time-ordered.

    .venv\\Scripts\\activate
    python build_dataset.py            (once)
    python evaluate_chronological.py   (about 30 minutes on one CPU core; resumes if stopped)
    python evaluate_chronological.py --quick   (a few minutes, for a smoke test)

Design (written down BEFORE looking at any test number)
  * Orders sorted by purchase time: oldest 65% = TRAIN, next 15% = VALIDATION, newest 20% = TEST.
    Test orders were bought after every train and validation order. Test labels are used once, at the end.
  * Hyper-parameters, blend weights, the alert threshold and the probability calibration are chosen on
    VALIDATION only (nothing is tuned on test, nothing on random folds).
  * Features: every feature uses only information known when the order is placed (see audit_leakage.py).
  * Compared on the same rows: a prevalence baseline, logistic regression, the v4 production XGBoost
    (33 features), the v4 served ensemble recipe (33 features), LightGBM on 33 features, and the enriched
    models (70 features) as single models and as a blend.
  * Reported: ROC-AUC (with a bootstrap 95% interval), PR-AUC, precision / recall / F1 / confusion matrix at the
    validation-chosen threshold, recall in the top 10% of orders, Brier score and calibration error,
    prevalence-aware baselines.
  * Rolling-origin backtest: retrain on everything before a month, score that month (same months for every model).
  * Group ablation: which of the new feature groups helps on the time-ordered test set.
  * Random-split numbers are only a secondary diagnostic.
  * Decision rule for what the app serves by default (fixed in advance, see `decide`): the enriched blend must beat the
    served v4 ensemble on the test block with a 95% interval above zero, win the rolling backtest, and be calibrated.

Writes dashboard/chronological_report.json (read by the app and by backend/add_enriched_scores.py).
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

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402
from sklearn.compose import ColumnTransformer                               # noqa: E402
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score   # noqa: E402
from sklearn.model_selection import ParameterSampler                        # noqa: E402
from sklearn.pipeline import Pipeline                                       # noqa: E402
from sklearn.preprocessing import OrdinalEncoder                            # noqa: E402

import enriched_model as em                                                 # noqa: E402
from enriched_features import GROUPS                                        # noqa: E402
from enriched_model import ENRICHED_COLUMNS, as_frame, build_hgb_e, build_lgbm_e, build_xgb_e   # noqa: E402
from ensemble import (BlendModel, apply_calibration, build_hgb, build_lr, build_xgb, choose_threshold,   # noqa: E402
                      evaluate_scores, fit_calibration, fit_weights, reliability_table, shift_odds)
from features import FEATURE_COLUMNS, TARGET_COLUMN                         # noqa: E402
from modeling import NUMERIC, stratified_split                              # noqa: E402
from train_enriched import HGB_SPACE, LGBM_SPACE, XGB_SPACE                 # noqa: E402
from tuning import _plain                                                   # noqa: E402

DATA = os.path.join(HERE, "data", "processed", "ontime_model_ready.csv")
REPORT = os.path.join(HERE, "dashboard", "chronological_report.json")
CACHE = os.path.join(HERE, "data", "interim")
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.65, 0.15

# the v4 recipes, exactly as shipped (dashboard/ensemble_report.json and models/model_metadata.json)
V4_XGB_PROD = dict(n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, min_child_weight=5)
V4_XGB_ENS = {'colsample_bytree': 0.811649, 'gamma': 0.661796, 'learning_rate': 0.023152, 'max_depth': 9, 'min_child_weight': 5,
              'n_estimators': 248, 'reg_alpha': 0.153884, 'reg_lambda': 1.377497, 'subsample': 0.824497}
V4_HGB_ENS = {'l2_regularization': 0.019949, 'learning_rate': 0.057581, 'max_depth': None, 'max_iter': 249, 'max_leaf_nodes': 54, 'min_samples_leaf': 43}
V4_WEIGHTS = {"xgb": 0.55, "hgb": 0.40, "lr": 0.05}


def log(m):
    print(m, flush=True)


def core_frame(df):
    X = df[FEATURE_COLUMNS].copy()
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


def build_v4_blend(spw_unused=None):
    return BlendModel({"xgb": build_xgb(V4_XGB_ENS), "hgb": build_hgb(V4_HGB_ENS), "lr": build_lr()}, weights=V4_WEIGHTS)


def lgbm_on(cols, params):
    """LightGBM (natural class mix) restricted to `cols`, categories read as true categories."""
    from lightgbm import LGBMClassifier
    cats = em.CATEGORICAL_E
    num = [c for c in cols if c not in cats]
    pre = ColumnTransformer([("num", "passthrough", num),
                             ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), cats)],
                            sparse_threshold=0)

    class _L(LGBMClassifier):
        def fit(self, X, y, **kw):
            kw["categorical_feature"] = list(range(len(num), len(num) + len(cats)))
            return super().fit(X, y, **kw)
    p = dict(build_lgbm_e().named_steps["clf"].get_params())
    p.update(params or {})
    p.update(scale_pos_weight=1.0, n_jobs=1, verbose=-1, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", _L(**p))])


def tune(name, builder, space, n, Xtr, ytr, Xva, yva, quick):
    """Trial 0 = the hand-set defaults. Score = ROC-AUC on the VALIDATION block (later than every train row)."""
    cands = [{}] + [{k: _plain(v) for k, v in p.items()} for p in ParameterSampler(space, n_iter=n, random_state=SEED)]
    best, rows = None, []
    for i, params in enumerate(cands):
        t0 = time.time()
        m = builder(params)
        m.fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        rows.append({"trial": i, "kind": "baseline" if i == 0 else "search", "val_roc_auc": round(float(auc), 5), "seconds": round(time.time() - t0, 1)})
        log(f"  [{name} {i + 1}/{len(cands)}] validation AUC {auc:.4f}  ({time.time() - t0:.0f}s)")
        if best is None or auc > best[0]:
            best = (auc, params, m)
    return best[1], best[2], rows


def boot_auc(y, preds, B=300, seed=0):
    """Paired bootstrap of ROC-AUC over test orders (the same resample for every model)."""
    rng = np.random.default_rng(seed)
    n, out = len(y), {k: [] for k in preds}
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        for k, p in preds.items():
            out[k].append(roc_auc_score(y[i], p[i]))
    return {k: np.asarray(v) for k, v in out.items()}


def ci(a):
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def block(y, p, thr):
    """Everything the report quotes for one model on one set of orders."""
    ev = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in evaluate_scores(y, p, thr).items()}
    pred = p >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    top = max(int(round(len(p) * 0.10)), 1)
    idx = np.argsort(-p)[:top]
    ev["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    ev["precision_at_top10pct"] = round(float(y[idx].mean()), 4)
    ev["accuracy"] = round(float((pred == (y == 1)).mean()), 4)
    return ev


def decide(report):
    """The serving rule, fixed before any test number was seen."""
    t = report["test_models"]
    enr, served = t["enriched_blend"], t["v4_ensemble_core"]
    d_ci = report["paired_auc_difference_vs_v4_ensemble"]["ci95"]
    bt = report["rolling_backtest"]
    checks = {
        "test_auc_ci_above_zero_vs_v4_ensemble": bool(d_ci[0] > 0),
        "backtest_mean_auc_higher": bool(bt["mean_auc"]["enriched_blend"] > bt["mean_auc"]["v4_ensemble_core"]),
        "backtest_wins_at_least_60pct_of_months": bool(bt["enriched_wins_share"] >= 0.60),
        "calibration_ece_at_most_3pp": bool(report["calibration"]["test"]["ece_calibrated"] <= 0.03),
    }
    ok = all(checks.values())
    why = (f"Time-ordered test block: enriched blend AUC {enr['roc_auc']} vs v4 ensemble {served['roc_auc']} "
           f"(difference {report['paired_auc_difference_vs_v4_ensemble']['mean']:+.4f}, 95% interval {d_ci[0]:+.4f} to {d_ci[1]:+.4f}). "
           f"Rolling backtest mean AUC {bt['mean_auc']['enriched_blend']} vs {bt['mean_auc']['v4_ensemble_core']}, "
           f"enriched better in {bt['enriched_wins']} of {bt['n_months']} months. ")
    why += ("All four checks pass, so the enriched blend is served by default." if ok
            else "Not every check passes (" + ", ".join(k for k, v in checks.items() if not v) + "), so the v4 ensemble stays the default.")
    return {"serve": "enriched" if ok else "ensemble", "checks": checks, "reason": why}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few trials, few backtest months: a smoke test, not a result")
    ap.add_argument("--no-backtest", action="store_true")
    args = ap.parse_args()
    q = args.quick
    t_all = time.time()
    n_xgb, n_lgbm, n_hgb = (1, 1, 1) if q else (5, 8, 4)
    os.makedirs(CACHE, exist_ok=True)

    df = pd.read_csv(DATA, parse_dates=["order_purchase_timestamp"]).sort_values("order_purchase_timestamp", kind="stable").reset_index(drop=True)
    n = len(df)
    i1, i2 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    tr, va, te = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    ytr, yva, yte = (x[TARGET_COLUMN].to_numpy() for x in (tr, va, te))
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    span = lambda x: [str(x["order_purchase_timestamp"].min().date()), str(x["order_purchase_timestamp"].max().date())]   # noqa: E731
    log(f"Chronological split: train {len(tr):,} {span(tr)}  validation {len(va):,} {span(va)}  test {len(te):,} {span(te)}")
    log(f"Late rate: train {ytr.mean():.3%}  validation {yva.mean():.3%}  test {yte.mean():.3%}")
    assert tr["order_purchase_timestamp"].max() <= va["order_purchase_timestamp"].min() <= te["order_purchase_timestamp"].min()

    Xtr, Xva, Xte = (as_frame(x).reset_index(drop=True) for x in (tr, va, te))
    Ctr, Cva, Cte = (core_frame(x).reset_index(drop=True) for x in (tr, va, te))
    ytr_s, yva_s = pd.Series(ytr), pd.Series(yva)

    # ------------------------------------------------------------------ 1. fit everything on TRAIN, tune on VALIDATION
    preds_path = os.path.join(CACHE, f".chrono_preds{'_quick' if q else ''}.npz")
    meta_path = os.path.join(CACHE, f".chrono_meta{'_quick' if q else ''}.json")
    if os.path.exists(preds_path) and os.path.exists(meta_path):
        z = np.load(preds_path)
        pv, pt = {k[3:]: z[k] for k in z.files if k.startswith("va_")}, {k[3:]: z[k] for k in z.files if k.startswith("te_")}
        meta = json.load(open(meta_path))
        log("\n1. (re-using finished model fits and predictions)")
    else:
        log("\n1. Tuning on the validation block; fitting on the training block")
        pv, pt, meta = {}, {}, {"best": {}, "trials": {}}
        best = {}
        for name, builder, space, nn in (("xgb", lambda p: build_xgb_e(p, 1.0), XGB_SPACE, n_xgb),
                                         ("lgbm", lambda p: build_lgbm_e(p, 1.0), LGBM_SPACE, n_lgbm),
                                         ("hgb", lambda p: build_hgb_e(p), HGB_SPACE, n_hgb)):
            params, model, rows = tune(name, builder, space, nn, Xtr, ytr_s, Xva, yva_s, q)
            best[name] = params
            meta["best"][name], meta["trials"][name] = params, rows
            pv[f"{name}_e"], pt[f"{name}_e"] = (model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1])
        log("  baselines (33 core features)")
        m = build_xgb(V4_XGB_PROD, spw).fit(Ctr, ytr_s)                      # the v4 production XGBoost, class-balanced
        pv["xgb_core_v4"], pt["xgb_core_v4"] = m.predict_proba(Cva)[:, 1], m.predict_proba(Cte)[:, 1]
        b4 = build_v4_blend().fit(Ctr, ytr_s)                                # the v4 served ensemble recipe
        pv["v4_ensemble_core"], pt["v4_ensemble_core"] = b4.predict_proba(Cva)[:, 1], b4.predict_proba(Cte)[:, 1]
        m = build_lr().fit(Ctr, ytr_s)
        pv["lr_core"], pt["lr_core"] = m.predict_proba(Cva)[:, 1], m.predict_proba(Cte)[:, 1]
        m = lgbm_on(FEATURE_COLUMNS, best["lgbm"]).fit(Xtr[FEATURE_COLUMNS], ytr_s)
        pv["lgbm_core"], pt["lgbm_core"] = m.predict_proba(Xva[FEATURE_COLUMNS])[:, 1], m.predict_proba(Xte[FEATURE_COLUMNS])[:, 1]
        np.savez(preds_path, **{f"va_{k}": v for k, v in pv.items()}, **{f"te_{k}": v for k, v in pt.items()})
        json.dump(meta, open(meta_path, "w"), default=float)
    best = meta["best"]

    # members trained on the natural mix are mapped to the balanced scale of the risk bands
    sh = lambda p: shift_odds(p, spw)   # noqa: E731
    mem = ["xgb_e", "lgbm_e", "hgb_e"]
    oof_va = pd.DataFrame({k[:-2]: sh(pv[k]) for k in mem})
    weights, val_blend_auc = fit_weights(oof_va, yva)
    blend_va = sum(weights[k[:-2]] * sh(pv[k]) for k in mem)
    blend_te = sum(weights[k[:-2]] * sh(pt[k]) for k in mem)
    thr, _ = choose_threshold(yva, blend_va)
    cal = fit_calibration(blend_va, yva)
    log(f"   blend weights (chosen on validation) {weights}; validation AUC {val_blend_auc:.4f}; alert threshold {thr:.3f}")

    # ------------------------------------------------------------------ 2. test block, scored once
    log("\n2. Test block (newest 20% of orders), scored once")
    scores_te = {"enriched_blend": blend_te, "enriched_xgb": sh(pt["xgb_e"]), "enriched_lightgbm": sh(pt["lgbm_e"]), "enriched_hgb": sh(pt["hgb_e"]),
                 "lightgbm_core": sh(pt["lgbm_core"]), "xgb_core_v4": pt["xgb_core_v4"], "v4_ensemble_core": pt["v4_ensemble_core"], "lr_core": pt["lr_core"]}
    scores_va = {"enriched_blend": blend_va, "enriched_xgb": sh(pv["xgb_e"]), "enriched_lightgbm": sh(pv["lgbm_e"]), "enriched_hgb": sh(pv["hgb_e"]),
                 "lightgbm_core": sh(pv["lgbm_core"]), "xgb_core_v4": pv["xgb_core_v4"], "v4_ensemble_core": pv["v4_ensemble_core"], "lr_core": pv["lr_core"]}
    labels = {"enriched_blend": "Enriched ensemble (blend), 70 features", "enriched_xgb": "XGBoost, 70 features", "enriched_lightgbm": "LightGBM, 70 features",
              "enriched_hgb": "Histogram gradient boosting, 70 features", "lightgbm_core": "LightGBM, 33 core features",
              "xgb_core_v4": "v4 production XGBoost, 33 features", "v4_ensemble_core": "v4 served ensemble, 33 features", "lr_core": "Logistic regression, 33 features"}
    test_models = {}
    for k, p in scores_te.items():
        t_k, _ = choose_threshold(yva, scores_va[k])
        test_models[k] = {"label": labels[k], **block(yte, p, t_k)}
        r = test_models[k]
        log(f"   {labels[k]:<46s} AUC {r['roc_auc']:.4f}  PR-AUC {r['pr_auc']:.4f}  P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}")
    prevalence = {"test_late_rate": round(float(yte.mean()), 4), "roc_auc": 0.5, "pr_auc": round(float(yte.mean()), 4),
                  "accuracy_if_all_on_time": round(float(1 - yte.mean()), 4),
                  "note": "A model that predicts 'on time' for everything is already this accurate and catches nothing; judge by AUC, PR-AUC, precision and recall."}
    bs = boot_auc(yte, {k: scores_te[k] for k in ("enriched_blend", "v4_ensemble_core", "xgb_core_v4")})
    for k in bs:
        test_models[k]["roc_auc_ci95"] = ci(bs[k])
    diff = bs["enriched_blend"] - bs["v4_ensemble_core"]
    diff2 = bs["enriched_blend"] - bs["xgb_core_v4"]
    paired = {"mean": round(float(diff.mean()), 4), "ci95": ci(diff), "bootstrap_resamples": int(len(diff))}
    paired2 = {"mean": round(float(diff2.mean()), 4), "ci95": ci(diff2)}
    log(f"   enriched blend minus v4 ensemble: {paired['mean']:+.4f}  95% interval {paired['ci95']}")

    chance_te = apply_calibration(cal, blend_te)
    rel, ece = reliability_table(yte, chance_te, 10)
    _, ece_raw = reliability_table(yte, blend_te, 10)
    calibration = {"method": "isotonic regression fitted on the validation block", "test": {
        "brier_calibrated": round(float(brier_score_loss(yte, chance_te)), 4), "base_rate_brier": round(float(brier_score_loss(yte, np.full(len(yte), yte.mean()))), 4),
        "ece_calibrated": round(ece, 4), "ece_raw_score": round(ece_raw, 4)}, "reliability_calibrated": rel}
    # monthly late rate on the test block (shows drift)
    mon = te.assign(m=te["order_purchase_timestamp"].dt.to_period("M").astype(str)).groupby("m")[TARGET_COLUMN].agg(["mean", "size"])
    drift = [{"month": k, "late_rate": round(float(r["mean"]), 4), "n": int(r["size"])} for k, r in mon.iterrows()]

    # ------------------------------------------------------------------ 3. group ablation (time-ordered test block)
    abl_path = os.path.join(CACHE, f".chrono_ablation{'_quick' if q else ''}.json")
    if os.path.exists(abl_path):
        ablation = json.load(open(abl_path))
        log("\n3. (re-using ablation)")
    else:
        log("\n3. Ablation: LightGBM (validation-tuned settings) on core + one new feature group")
        ablation = []
        for setting, cols in [("Core 33 features only", FEATURE_COLUMNS)] + [(f"Core + {g}", FEATURE_COLUMNS + c) for g, c in GROUPS.items()] + [("Core + all new features", ENRICHED_COLUMNS)]:
            m = lgbm_on(cols, best["lgbm"]).fit(Xtr[cols], ytr_s)
            a = float(roc_auc_score(yte, m.predict_proba(Xte[cols])[:, 1]))
            ablation.append({"setting": setting, "n_features": len(cols), "test_roc_auc": round(a, 4)})
            log(f"   {setting:<28s} {a:.4f}")
        json.dump(ablation, open(abl_path, "w"))

    # ------------------------------------------------------------------ 4. rolling-origin backtest
    bt_path = os.path.join(CACHE, f".chrono_backtest{'_quick' if q else ''}.json")
    backtest = {"rows": [], "n_months": 0, "mean_auc": {}, "enriched_wins": 0, "enriched_wins_share": None}
    if not args.no_backtest:
        if os.path.exists(bt_path):
            rows = json.load(open(bt_path))
            log("\n4. (re-using backtest)")
        else:
            log("\n4. Rolling-origin backtest: retrain on all earlier months, score the next month")
            mth = df["order_purchase_timestamp"].dt.to_period("M")
            months = sorted(mth.unique())
            rows = []
            for i in range(6, len(months), 4 if q else 2):
                trn, tst = df[mth < months[i]], df[mth == months[i]]
                if tst[TARGET_COLUMN].nunique() < 2 or len(tst) < 100:
                    continue
                y_tr, y_ts = trn[TARGET_COLUMN].reset_index(drop=True), tst[TARGET_COLUMN].to_numpy()
                s_f = float((y_tr == 0).sum() / (y_tr == 1).sum())
                r = {"month": str(months[i]), "n": int(len(tst)), "late_rate": round(float(y_ts.mean()), 4)}
                Ct, Cs = core_frame(trn).reset_index(drop=True), core_frame(tst)
                r["xgb_core_v4"] = float(roc_auc_score(y_ts, build_xgb(V4_XGB_PROD, s_f).fit(Ct, y_tr).predict_proba(Cs)[:, 1]))
                r["v4_ensemble_core"] = float(roc_auc_score(y_ts, build_v4_blend().fit(Ct, y_tr).predict_proba(Cs)[:, 1]))
                et, es = as_frame(trn).reset_index(drop=True), as_frame(tst)
                eb = BlendModel({"xgb": build_xgb_e(best["xgb"], 1.0), "lgbm": build_lgbm_e(best["lgbm"], 1.0), "hgb": build_hgb_e(best["hgb"])},
                                weights=weights, explainer="xgb", odds_shift=s_f).fit(et, y_tr)
                r["enriched_blend"] = float(roc_auc_score(y_ts, eb.predict_proba(es)[:, 1]))
                rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
                log(f"    {r['month']}  n={r['n']:>5,}  xgb33 {r['xgb_core_v4']:.4f}  v4-ensemble {r['v4_ensemble_core']:.4f}  enriched {r['enriched_blend']:.4f}")
                json.dump(rows, open(bt_path, "w"))
            json.dump(rows, open(bt_path, "w"))
        if rows:
            b = pd.DataFrame(rows)
            wins = int((b["enriched_blend"] > b["v4_ensemble_core"]).sum())
            backtest = {"rows": rows, "n_months": int(len(b)), "enriched_wins": wins, "enriched_wins_share": round(wins / len(b), 3),
                        "mean_auc": {k: round(float(b[k].mean()), 4) for k in ("xgb_core_v4", "v4_ensemble_core", "enriched_blend")}}

    # ------------------------------------------------------------------ 5. refit on train+validation (secondary), random split (secondary)
    log("\n5. Secondary diagnostics")
    tv = pd.concat([tr, va])
    y_tv = tv[TARGET_COLUMN].reset_index(drop=True)
    s_tv = float((y_tv == 0).sum() / (y_tv == 1).sum())
    eb = BlendModel({"xgb": build_xgb_e(best["xgb"], 1.0), "lgbm": build_lgbm_e(best["lgbm"], 1.0), "hgb": build_hgb_e(best["hgb"])},
                    weights=weights, explainer="xgb", odds_shift=s_tv).fit(as_frame(tv).reset_index(drop=True), y_tv)
    refit_auc = round(float(roc_auc_score(yte, eb.predict_proba(Xte)[:, 1])), 4)
    log(f"   enriched blend refit on train+validation, test AUC {refit_auc}")
    rtr, rte = stratified_split(df)
    ry, rty = rtr[TARGET_COLUMN].reset_index(drop=True), rte[TARGET_COLUMN].to_numpy()
    r_core = roc_auc_score(rty, build_xgb(V4_XGB_PROD, float((ry == 0).sum() / (ry == 1).sum())).fit(core_frame(rtr).reset_index(drop=True), ry).predict_proba(core_frame(rte))[:, 1])
    r_enr = roc_auc_score(rty, lgbm_on(ENRICHED_COLUMNS, best["lgbm"]).fit(as_frame(rtr).reset_index(drop=True)[ENRICHED_COLUMNS], ry).predict_proba(as_frame(rte)[ENRICHED_COLUMNS])[:, 1])
    random_split = {"xgb_core_v4": round(float(r_core), 4), "lightgbm_enriched": round(float(r_enr), 4),
                    "note": "Random stratified 80/20 split (the same 77,176 / 19,294 orders as every earlier model). Secondary only: it lets orders from the same weeks and sellers sit on both sides."}
    log(f"   random split: v4 XGBoost {random_split['xgb_core_v4']}  enriched LightGBM {random_split['lightgbm_enriched']}")

    report = {
        "generated_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "quick_run": bool(q), "seed": SEED,
        "data": "REAL Olist data only (no simulated feature anywhere)",
        "split": {"train": {"n": int(len(tr)), "from_to": span(tr), "late_rate": round(float(ytr.mean()), 4)},
                  "validation": {"n": int(len(va)), "from_to": span(va), "late_rate": round(float(yva.mean()), 4)},
                  "test": {"n": int(len(te)), "from_to": span(te), "late_rate": round(float(yte.mean()), 4)}},
        "protocol": ["Orders sorted by purchase time: oldest 65% train, next 15% validation, newest 20% test (all later than every train order).",
                     "Hyper-parameters, blend weights, alert threshold and calibration are chosen on the validation block only; the test block is scored once.",
                     "Every feature is known when the order is placed; audit_leakage.py scrambles all later outcomes and checks no feature moves."],
        "prevalence_baseline": prevalence, "test_models": test_models, "blend_weights": weights, "threshold": round(float(thr), 4),
        "tuned_params": best, "paired_auc_difference_vs_v4_ensemble": paired, "paired_auc_difference_vs_v4_xgboost": paired2,
        "calibration": calibration, "test_late_rate_by_month": drift, "ablation": ablation, "rolling_backtest": backtest,
        "refit_train_plus_validation_test_auc": refit_auc, "random_split_secondary": random_split,
        "target_090": {"goal": 0.90, "reached_on_test_block": bool(test_models["enriched_blend"]["roc_auc"] >= 0.90),
                       "best_test_roc_auc": test_models["enriched_blend"]["roc_auc"],
                       "note": "0.90 was an aspiration, not a number to force. Olist has no carrier scans, weather, strike or live backlog data; see the report for what would be needed."},
        "runtime_minutes": round((time.time() - t_all) / 60, 1),
    }
    if backtest["rows"]:
        report["decision"] = decide(report)
    else:
        report["decision"] = {"serve": "ensemble", "reason": "No backtest was run, so the default is not changed.", "checks": {}}
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT if not q else REPORT.replace(".json", "_quick.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"\nDecision: serve {report['decision']['serve']}. {report['decision']['reason']}")
    log(f"Done in {(time.time() - t_all) / 60:.1f} min.")


if __name__ == "__main__":
    main()
