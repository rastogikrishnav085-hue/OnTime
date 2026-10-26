"""
modeling.py
------------
Trains and evaluates the OnTime models. Four candidates now:
Logistic Regression, Decision Tree, Random Forest, and XGBoost, each
with a hyperparameter-tuned variant on top of the original defaults.

Also adds two things the first version of this project was missing:
  - decision-threshold tuning (0.5 is an arbitrary default, not a
    business-justified cutoff for an imbalanced problem)
  - a rolling-origin backtest that actually tests periodic retraining
    as the fix for the seasonality problem, instead of only reporting
    the single time-split failure as a limitation.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS, TARGET_COLUMN

CATEGORICAL = ["product_category", "customer_state", "seller_state", "payment_type"]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]


def time_based_split(df, date_col="order_purchase_timestamp", test_frac=0.2):
    df = df.sort_values(date_col).reset_index(drop=True)
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def stratified_split(df, target=TARGET_COLUMN, test_frac=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df, test_size=test_frac, stratify=df[target], random_state=random_state
    )
    return train_df.copy(), test_df.copy()


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def build_models():
    """Baseline models, using reasonable-but-not-tuned defaults. Kept
    around so the report can still show 'baseline vs tuned' honestly."""
    pre = build_preprocessor()
    models = {
        "logistic_regression": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "decision_tree": Pipeline([
            ("pre", pre),
            ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                            class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=12,
                                            min_samples_leaf=20, class_weight="balanced",
                                            random_state=42, n_jobs=1)),
        ]),
        "xgboost": Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, eval_metric="auc",
                tree_method="hist", random_state=42,
            )),
        ]),
    }
    return models


# ---------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------
# NOTE ON HARDWARE: this environment has a single CPU core. That makes
# n_jobs=-1 actively counterproductive (process-spawning overhead with
# no real parallelism) and rules out standard k-fold RandomizedSearchCV
# for Random Forest, which costs cv x n_iter full-forest fits -- at
# ~100s per 300-tree fit that blows past any reasonable time budget.
# XGBoost's histogram-based training is drastically cheaper per fit
# (~3s vs ~100s for a comparable RF), so it gets a proper search;
# Random Forest gets a smaller, capped-size holdout search instead of
# being dropped altogether.
RF_PARAM_GRID = {
    "clf__n_estimators": [60, 100],
    "clf__max_depth": [8, 10],
    "clf__min_samples_leaf": [30, 50],
    "clf__max_features": ["sqrt"],
}

XGB_PARAM_GRID = {
    "clf__n_estimators": [200, 300, 400, 600],
    "clf__max_depth": [3, 4, 5, 6, 8],
    "clf__learning_rate": [0.03, 0.05, 0.08, 0.12, 0.2],
    "clf__subsample": [0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 0.9, 1.0],
    "clf__min_child_weight": [1, 3, 5, 10],
}


def _force_single_thread(pipe):
    from sklearn.base import clone
    p = clone(pipe)
    if hasattr(p.named_steps["clf"], "n_jobs"):
        p.named_steps["clf"].set_params(n_jobs=1)
    return p


def tune_model_holdout(pipe, param_grid, X_train, y_train, n_iter=15, val_frac=0.2,
                        random_state=42, scoring_fn=roc_auc_score, verbose=True):
    """
    Random search using a SINGLE train/validation split rather than
    k-fold CV. On single-core hardware this is n_iter fits instead of
    n_iter x cv fits -- a 3x (or more) speedup for the same number of
    configurations tried, at the cost of a noisier estimate of each
    configuration's quality. Given the alternative was "doesn't finish",
    this is the right trade here. The winning configuration is refit on
    the FULL X_train (train + validation) before being returned.
    """
    rng = np.random.RandomState(random_state)
    tr_idx, val_idx = train_test_split(
        np.arange(len(X_train)), test_size=val_frac, stratify=y_train, random_state=random_state
    )
    Xtr, ytr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    Xval, yval = X_train.iloc[val_idx], y_train.iloc[val_idx]

    keys = list(param_grid.keys())
    tried, best_score, best_params = [], -np.inf, None
    for i in range(n_iter):
        params = {k: param_grid[k][rng.randint(len(param_grid[k]))] for k in keys}
        cand = _force_single_thread(pipe)
        cand.set_params(**params)
        cand.fit(Xtr, ytr)
        score = scoring_fn(yval, cand.predict_proba(Xval)[:, 1])
        tried.append({**{k.replace("clf__", ""): v for k, v in params.items()}, "val_roc_auc": score})
        if verbose:
            print(f"  [{i+1}/{n_iter}] auc={score:.4f}  {params}")
        if score > best_score:
            best_score, best_params = score, params

    best_pipe = _force_single_thread(pipe)
    best_pipe.set_params(**best_params)
    best_pipe.fit(X_train, y_train)  # refit on full training data
    return best_pipe, best_params, best_score, pd.DataFrame(tried)


def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred),
        "proba": proba,
        "pred": pred,
        "threshold": threshold,
    }


def find_best_threshold(y_true, proba, metric="f1"):
    """Scans precision_recall_curve's thresholds and returns the one
    that maximises F1 (or, if metric='precision_at_recall', the lowest
    threshold that still hits a target recall -- useful for an
    early-warning system where missing a delay is costlier than a
    false alarm)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = np.where((precisions + recalls) > 0, 2 * precisions * recalls / (precisions + recalls + 1e-12), 0)
    best_idx = np.argmax(f1s[:-1])  # last point has no matching threshold
    return float(thresholds[best_idx]), float(f1s[best_idx])


def select_operating_threshold(pipe_with_best_params, X_train, y_train, val_frac=0.15, random_state=7):
    """
    Selects the F1-optimal decision threshold using a validation slice
    carved out of TRAINING data only -- never the test set.

    This exists to fix a real bug from the first version of this
    notebook: `find_best_threshold` was originally called on
    (y_test, proba_test) and the resulting threshold was then evaluated
    on that SAME y_test -- tuning a knob on the test set and then
    grading yourself with it. That's leakage, and it makes the reported
    F1 optimistically biased (the threshold is fit to that exact test
    set's noise, not to the population it's drawn from).

    The fix: split off a validation slice from the training data,
    select the threshold there, and only ever apply it to the test set
    -- never fit anything using the test set's labels.
    """
    from sklearn.base import clone
    tr_idx, val_idx = train_test_split(
        np.arange(len(X_train)), test_size=val_frac, stratify=y_train, random_state=random_state
    )
    Xtr, ytr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    Xval, yval = X_train.iloc[val_idx], y_train.iloc[val_idx]

    probe = clone(pipe_with_best_params)
    if hasattr(probe.named_steps["clf"], "n_jobs"):
        probe.named_steps["clf"].set_params(n_jobs=1)
    probe.fit(Xtr, ytr)
    proba_val = probe.predict_proba(Xval)[:, 1]
    threshold, val_f1 = find_best_threshold(yval.values, proba_val)
    return threshold, val_f1


def risk_bucket(p):
    if p < 0.30:
        return "LOW"
    elif p < 0.60:
        return "MEDIUM"
    elif p < 0.80:
        return "HIGH"
    else:
        return "CRITICAL"


def top_feature_names(preprocessor):
    num_names = NUMERIC
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL))
    return num_names + cat_names


# ---------------------------------------------------------------------
# Rolling-origin backtest: the actual fix for the seasonality problem
# ---------------------------------------------------------------------
def rolling_origin_backtest(df, model_builder, min_train_months=6, date_col="order_purchase_timestamp"):
    """
    Simulates periodic retraining instead of a single train-once model.
    For each calendar month after the initial training window, trains
    on everything strictly before that month and evaluates on that
    month alone, then slides forward. This is what exposed the
    seasonality problem in the first version of this project (a single
    static time split), and it's also the direct fix: a deployed system
    would retrain on a schedule like this rather than never.
    """
    df = df.copy()
    df["_month"] = df[date_col].dt.to_period("M")
    months = sorted(df["_month"].unique())
    if len(months) <= min_train_months:
        raise ValueError("Not enough distinct months for a rolling backtest.")

    records = []
    for i in range(min_train_months, len(months)):
        test_month = months[i]
        train_mask = df["_month"] < test_month
        test_mask = df["_month"] == test_month
        train_df, test_df = df[train_mask], df[test_mask]
        if test_df[TARGET_COLUMN].nunique() < 2 or len(test_df) < 30:
            continue  # too few orders / no positive examples that month to score AUC

        pipe = model_builder()
        pipe.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
        proba = pipe.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
        auc = roc_auc_score(test_df[TARGET_COLUMN], proba)
        records.append({
            "test_month": str(test_month), "n_train": len(train_df), "n_test": len(test_df),
            "delay_rate": float(test_df[TARGET_COLUMN].mean()), "roc_auc": float(auc),
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = pd.read_csv("../data/processed/ontime_model_ready.csv", parse_dates=["order_purchase_timestamp"])

    train_df, test_df = stratified_split(df)
    print(f"Train: {len(train_df)} orders / Test: {len(test_df)} orders "
          f"(stratified random split, base rate {df[TARGET_COLUMN].mean():.3f})")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    models = build_models()
    results = {}
    for name, pipe in models.items():
        if name == "xgboost":
            pipe.named_steps["clf"].set_params(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
        pipe.fit(X_train, y_train)
        res = evaluate(pipe, X_test, y_test)
        results[name] = res
        print(f"\n{name}")
        print(f"  precision={res['precision']:.3f} recall={res['recall']:.3f} "
              f"f1={res['f1']:.3f} roc_auc={res['roc_auc']:.3f}")

