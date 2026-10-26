"""
tuning.py
----------
Hyperparameter search for the OnTime models, using only scikit-learn,
SciPy and XGBoost (nothing extra to install).

What it does, and why it is built this way
  * Random search over wide, sensible ranges (Bergstra & Bengio, 2012:
    random search matches or beats grid search for the same budget
    because only a few hyperparameters matter for any one data set).
  * Every candidate is scored by stratified K-fold cross-validation on
    the TRAINING rows only. The held-out test rows are never seen here.
  * The model's current settings are always trial 0, so "did tuning
    help?" is answered on identical folds, not by comparing against a
    number from a different protocol.
  * Every trial is kept (not just the winner) so the report can show
    how flat or steep the search landscape was.
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold

# ---- search spaces (keys are the classifier's own parameter names) ----------
XGB_SPACE = {
    "n_estimators": randint(150, 700),
    "max_depth": randint(3, 10),
    "learning_rate": loguniform(0.02, 0.2),
    "subsample": uniform(0.6, 0.4),            # 0.6 - 1.0
    "colsample_bytree": uniform(0.5, 0.5),     # 0.5 - 1.0
    "min_child_weight": randint(1, 20),
    "reg_lambda": loguniform(0.5, 20),
    "reg_alpha": loguniform(1e-3, 5),
    "gamma": uniform(0, 2),
}

HGB_SPACE = {
    "learning_rate": loguniform(0.02, 0.2),
    "max_iter": randint(150, 500),
    "max_leaf_nodes": randint(15, 64),
    "min_samples_leaf": randint(20, 120),
    "l2_regularization": loguniform(0.01, 10),
    "max_depth": [None, 6, 8, 10],
}


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 6)
    return v


def cross_validate_auc(pipe, X, y, cv=3, seed=42):
    """Mean ROC-AUC and PR-AUC of `pipe` over stratified folds of (X, y)."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    aucs, aps = [], []
    for tr, va in skf.split(X, y):
        m = clone(pipe)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], p))
        aps.append(average_precision_score(y.iloc[va], p))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(aps))


def random_search(pipe, space, X, y, n_iter=30, cv=3, seed=42, baseline_params=None, log=print, name="model", checkpoint=None):
    """Random search with the baseline as trial 0.

    Returns (best_params, trials_dataframe). `pipe` is a Pipeline whose last
    step is called "clf"; `space` keys are the classifier's own parameters.
    The data frame has one row per trial, sorted best-first.
    """
    candidates = []
    if baseline_params is not None:
        candidates.append(("baseline", dict(baseline_params)))
    for p in ParameterSampler(space, n_iter=n_iter, random_state=seed):
        candidates.append(("search", {k: _plain(v) for k, v in p.items()}))

    rows, t_start = [], time.time()
    if checkpoint and os.path.exists(checkpoint):
        # resume: trials already finished in an interrupted run are kept (same seed -> same candidates)
        done = pd.read_json(checkpoint, orient="records")
        rows = done.to_dict("records")[: len(candidates)]
        log(f"  [{name}] resuming: {len(rows)} trial(s) already done")
    for i, (kind, params) in enumerate(candidates):
        if i < len(rows):
            continue
        cand = clone(pipe)
        cand.set_params(**{f"clf__{k}": v for k, v in params.items()})
        t0 = time.time()
        auc, auc_sd, ap = cross_validate_auc(cand, X, y, cv=cv, seed=seed)
        rows.append({"trial": i, "kind": kind, "cv_roc_auc": round(auc, 5), "cv_roc_auc_sd": round(auc_sd, 5),
                     "cv_pr_auc": round(ap, 5), "seconds": round(time.time() - t0, 1), "params": params})
        if checkpoint:
            pd.DataFrame(rows).to_json(checkpoint, orient="records")
        log(f"  [{name} {i + 1}/{len(candidates)}] cv AUC={auc:.4f} (±{auc_sd:.4f}) PR-AUC={ap:.4f}  "
            f"{'<- current settings' if kind == 'baseline' else ''}")
    trials = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    best = dict(trials.loc[0, "params"])
    log(f"  {name}: best CV AUC {trials.loc[0, 'cv_roc_auc']:.4f} after {len(candidates)} trials "
        f"({time.time() - t_start:.0f}s).")
    return best, trials
