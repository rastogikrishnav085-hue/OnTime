"""
train_ensemble.py
------------------
Tunes the candidate models, builds the weighted ensemble, compares everything
fairly, and writes what the app shows on its Model page.

    .venv\\Scripts\\activate
    python train_ensemble.py            full run (about 20-30 minutes on one CPU core)
    python train_ensemble.py --quick    small search (about 3 minutes), for checking the pipeline

Protocol (the same split as the notebooks, so numbers are comparable):
  1. Stratified 80/20 split, random_state=42 (77,176 train / 19,294 test).
  2. Tune XGBoost and histogram gradient boosting by 3-fold CV on TRAIN only.
     The current production settings are trial 0 of the XGBoost search.
  3. Out-of-fold (5-fold) probabilities on TRAIN for every member; the blend
     weights and the alert threshold are chosen on those. Nothing here sees
     the test rows.
  4. Score the test rows ONCE for every model and the blend.
  5. Rolling-origin backtest (retrain monthly, predict the next month) for the
     production model, the tuned XGBoost and the blend.
  6. The blend becomes the served model only if it beats the production
     XGBoost on the test split AND on the backtest. Otherwise the report says
     so and the app keeps XGBoost.

Writes: models/ensemble/*, dashboard/ensemble_report.json,
        data/processed/tuning_trials_xgb.csv, tuning_trials_hgb.csv
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

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from artifacts import load_json, load_model                     # noqa: E402
from ensemble import (BlendModel, build_hgb, build_lr, build_rf, build_xgb, choose_threshold,   # noqa: E402
                      evaluate_scores, fit_weights, oof_predictions, save_ensemble, stacked_cv_score)
from features import FEATURE_COLUMNS, TARGET_COLUMN             # noqa: E402
from modeling import rolling_origin_backtest, stratified_split  # noqa: E402
from tuning import HGB_SPACE, XGB_SPACE, random_search          # noqa: E402

DATA = os.path.join(HERE, "data", "processed", "ontime_model_ready.csv")
MODELS = os.path.join(HERE, "models")
OUT_DIR = os.path.join(MODELS, "ensemble")
REPORT = os.path.join(HERE, "dashboard", "ensemble_report.json")


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny search, for a fast check")
    ap.add_argument("--xgb-trials", type=int, default=None)
    ap.add_argument("--hgb-trials", type=int, default=None)
    ap.add_argument("--no-rf", action="store_true", help="leave the random forest out of the candidates")
    ap.add_argument("--no-backtest", action="store_true")
    args = ap.parse_args()
    n_xgb = args.xgb_trials or (4 if args.quick else 24)
    n_hgb = args.hgb_trials or (3 if args.quick else 10)
    t_all = time.time()

    df = pd.read_csv(DATA, parse_dates=["order_purchase_timestamp"])
    train, test = stratified_split(df)
    Xtr, ytr = train[FEATURE_COLUMNS], train[TARGET_COLUMN].reset_index(drop=True)
    Xte, yte = test[FEATURE_COLUMNS], test[TARGET_COLUMN].to_numpy()
    Xtr = Xtr.reset_index(drop=True)
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    log(f"Train {len(train):,} / test {len(test):,}; late rate {df[TARGET_COLUMN].mean():.3%}; scale_pos_weight {spw:.2f}")

    # ---- production model (reference) ----
    prod, prod_meta = load_model(MODELS, FEATURE_COLUMNS, data_path=DATA, log=log)
    prod_params = {k: v for k, v in prod_meta["xgb_params"].items()
                   if k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight")}

    # ---- 1. tuning ----
    log("\n1. Tuning XGBoost (3-fold CV on the training rows)")
    xgb_best, xgb_trials = random_search(build_xgb(scale_pos_weight=spw), _clf_space(XGB_SPACE), Xtr, ytr,
                                         n_iter=n_xgb, baseline_params=prod_params, log=log, name="xgb",
                                         checkpoint=os.path.join(HERE, "data", "processed", ".ckpt_xgb.json"))
    log("\n   Tuning histogram gradient boosting")
    hgb_best, hgb_trials = random_search(build_hgb(), _clf_space(HGB_SPACE), Xtr, ytr,
                                         n_iter=n_hgb, baseline_params=None, log=log, name="hgb",
                                         checkpoint=os.path.join(HERE, "data", "processed", ".ckpt_hgb.json"))
    for _f in (".ckpt_xgb.json", ".ckpt_hgb.json"):
        pass  # checkpoints are removed at the end of a successful run
    xgb_trials.to_csv(os.path.join(HERE, "data", "processed", "tuning_trials_xgb.csv"), index=False)
    hgb_trials.to_csv(os.path.join(HERE, "data", "processed", "tuning_trials_hgb.csv"), index=False)
    base_row = xgb_trials[xgb_trials.kind == "baseline"].iloc[0]
    best_row = xgb_trials.iloc[0]

    # ---- 2. out-of-fold predictions, weights, threshold ----
    log("\n2. Out-of-fold predictions on the training rows (5 folds)")
    pipes = {"xgb": build_xgb(xgb_best, spw), "hgb": build_hgb(hgb_best), "lr": build_lr()}
    if not args.no_rf:
        pipes["rf"] = build_rf()
    oof = oof_predictions(pipes, Xtr, ytr, cv=3 if args.quick else 5, log=log)
    oof_scores = {n: evaluate_scores(ytr, oof[n], 0.5) for n in oof.columns}
    weights, oof_blend_auc = fit_weights(oof, ytr)
    blend_oof = sum(weights[n] * oof[n] for n in oof.columns)
    thr_blend, _ = choose_threshold(ytr, blend_oof)
    thr_xgb, _ = choose_threshold(ytr, oof["xgb"])
    stack_auc, stack_ap = stacked_cv_score(oof, ytr)
    log(f"   blend weights {weights}; OOF AUC {oof_blend_auc:.4f}; stacker OOF AUC {stack_auc:.4f}")

    # ---- 3. fit on full train, score the test rows once ----
    log("\n3. Fitting on the full training set and scoring the test rows once")
    blend = BlendModel(pipes, weights=weights, explainer="xgb").fit(Xtr, ytr)
    probs = blend.member_probabilities(Xte)
    p_blend = blend.predict_proba(Xte)[:, 1]
    p_prod = prod.predict_proba(Xte)[:, 1]
    prod_thr = float(prod_meta["threshold"])
    table = [{"key": "production_xgb", "label": "XGBoost - current production model", "served": False,
              **evaluate_scores(yte, p_prod, prod_thr)}]
    for n in pipes:
        thr = thr_xgb if n == "xgb" else choose_threshold(ytr, oof[n])[0]
        table.append({"key": f"{n}_tuned" if n in ("xgb", "hgb") else n,
                      "label": {"xgb": "XGBoost - re-tuned (wider search)", "hgb": "Histogram gradient boosting - tuned",
                                "rf": "Random forest", "lr": "Logistic regression"}[n],
                      "served": False, **evaluate_scores(yte, probs[n], thr)})
    table.append({"key": "blend", "label": "Weighted ensemble (blend)", "served": False,
                  **evaluate_scores(yte, p_blend, thr_blend)})
    stack_row = {"key": "stack_cv", "label": "Stacked ensemble (meta-learner), cross-validated on train - not served",
                 "served": False, "roc_auc": stack_auc, "pr_auc": stack_ap}
    for r in table:
        log(f"   {r['label']:<58s} AUC {r['roc_auc']:.4f}  PR-AUC {r['pr_auc']:.4f}  F1 {r['f1']:.3f}")

    # ---- 4. rolling backtest ----
    backtest = {}
    if not args.no_backtest:
        log("\n4. Rolling-origin backtest (retrain monthly, predict the next month)")
        df_bt = df.copy()

        def xgb_prod_builder():
            return _Refit(build_xgb(prod_params, spw))

        def xgb_tuned_builder():
            return _Refit(build_xgb(xgb_best, spw))

        def blend_builder():
            return BlendModel({n: build_xgb(xgb_best, spw) if n == "xgb" else (build_hgb(hgb_best) if n == "hgb" else
                               build_lr() if n == "lr" else build_rf()) for n in pipes}, weights=weights)
        for key, builder in (("production_xgb", xgb_prod_builder), ("xgb_tuned", xgb_tuned_builder), ("blend", blend_builder)):
            t0 = time.time()
            bt = rolling_origin_backtest(df_bt, builder)
            backtest[key] = [{"month": r.test_month, "auc": round(float(r.roc_auc), 4), "n": int(r.n_test)} for r in bt.itertuples()]
            log(f"   {key:<16s} mean AUC {bt.roc_auc.mean():.4f} over {len(bt)} months ({time.time() - t0:.0f}s)")
    bt_mean = {k: round(float(np.mean([r["auc"] for r in v])), 4) for k, v in backtest.items()}

    # ---- 5. decision ----
    # Rule, stated before the full run: the monthly backtest is the primary test because it is how the
    # system is meant to be used (retrain on a schedule, predict the next month), and the random split
    # mixes time. The blend is served only if (a) it beats the production XGBoost on the backtest by at
    # least 0.005 mean AUC, winning in at least 60% of the months, and (b) it is not worse on the random
    # test split by more than 0.005 AUC.
    prod_row = table[0]
    blend_row = next(r for r in table if r["key"] == "blend")
    test_diff = blend_row["roc_auc"] - prod_row["roc_auc"]
    bt_diff, bt_wins, bt_months = None, None, None
    if backtest:
        a = {r["month"]: r["auc"] for r in backtest["blend"]}
        b = {r["month"]: r["auc"] for r in backtest["production_xgb"]}
        common = sorted(set(a) & set(b))
        d = np.array([a[m] - b[m] for m in common])
        bt_diff, bt_wins, bt_months = float(d.mean()), int((d > 0).sum()), len(common)
    serve_blend = bool(backtest and bt_diff >= 0.005 and bt_wins / bt_months >= 0.6 and test_diff >= -0.005)
    for r in table:
        r["served"] = (r["key"] == "blend") if serve_blend else (r["key"] == "production_xgb")
    if backtest:
        facts = (f"Random test split: blend {blend_row['roc_auc']:.4f} vs production XGBoost {prod_row['roc_auc']:.4f} "
                 f"(difference {test_diff:+.4f}). Monthly backtest: blend {bt_mean['blend']} vs {bt_mean['production_xgb']} mean AUC "
                 f"(difference {bt_diff:+.4f}, blend better in {bt_wins} of {bt_months} months).")
    else:
        facts = f"Random test split: blend {blend_row['roc_auc']:.4f} vs production XGBoost {prod_row['roc_auc']:.4f}. Backtest skipped."
    reason = facts + (" The blend is served: it clears both bars (better on the monthly backtest by at least 0.005 AUC in at least 60% of "
                      "months, and no worse than 0.005 AUC on the random split)." if serve_blend else
                      " The blend is NOT served by default (it must be better on the monthly backtest by at least 0.005 AUC in at least 60% "
                      "of months, and no worse than 0.005 AUC on the random split). It stays available as a selectable engine.")
    log("\n5. Decision: " + reason)

    # ---- 6. save ----
    metrics = {k: round(v, 4) for k, v in blend_row.items() if isinstance(v, float)}
    meta = save_ensemble(blend, OUT_DIR, FEATURE_COLUMNS, thr_blend, metrics,
                         extra={"serve_by_default": serve_blend, "decision": reason,
                                "tuned_xgb_params": xgb_best, "tuned_hgb_params": hgb_best,
                                "oof_blend_auc": round(oof_blend_auc, 4)})
    report = {
        "generated_utc": meta["saved_at_utc"], "quick_run": bool(args.quick),
        "protocol": ["Stratified 80/20 split (random_state 42): 77,176 train / 19,294 test, the same split as notebooks 05-08.",
                     "Hyperparameters tuned by 3-fold CV on train only; weights and alert threshold chosen on out-of-fold train predictions.",
                     "Test rows scored once. Alert thresholds are F1-optimal on train (out-of-fold), never on test."],
        "tuning": {
            "xgb": {"trials": int(len(xgb_trials)), "baseline_cv_auc": float(base_row.cv_roc_auc),
                    "best_cv_auc": float(best_row.cv_roc_auc), "best_params": xgb_best,
                    "top": [{k: (v if k != "params" else v) for k, v in r.items() if k in ("trial", "kind", "cv_roc_auc", "cv_pr_auc", "params")}
                            for r in xgb_trials.head(5).to_dict("records")],
                    "all_cv_auc": [float(x) for x in xgb_trials.cv_roc_auc.tolist()]},
            "hgb": {"trials": int(len(hgb_trials)), "best_cv_auc": float(hgb_trials.iloc[0].cv_roc_auc),
                    "best_params": hgb_best, "all_cv_auc": [float(x) for x in hgb_trials.cv_roc_auc.tolist()]},
        },
        "weights": weights, "oof_auc": {n: round(oof_scores[n]["roc_auc"], 4) for n in oof.columns},
        "oof_blend_auc": round(oof_blend_auc, 4), "stack": stack_row,
        "models": table, "backtest": backtest, "backtest_mean_auc": bt_mean,
        "served": "blend" if serve_blend else "production_xgb", "decision": reason,
        "paired_backtest": {"mean_diff": bt_diff, "months_blend_better": bt_wins, "months": bt_months, "test_diff": test_diff},
        "member_labels": {n: __import__("ensemble").MEMBER_LABELS.get(n, n) for n in pipes},
        "runtime_minutes": round((time.time() - t_all) / 60, 1),
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for _f in (".ckpt_xgb.json", ".ckpt_hgb.json"):
        _p = os.path.join(HERE, "data", "processed", _f)
        if os.path.exists(_p):
            os.remove(_p)
    log(f"\nDone in {report['runtime_minutes']} min. Report: {REPORT}\nEnsemble saved to {OUT_DIR}")


def _clf_space(space):
    """The search helpers take plain classifier parameter names."""
    return space


class _Refit:
    """A pipeline that sets scale_pos_weight from the training labels on fit,
    for the rolling backtest (each month has its own class balance)."""

    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y):
        from sklearn.base import clone
        self.m = clone(self.pipe)
        self.m.named_steps["clf"].set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


if __name__ == "__main__":
    main()
