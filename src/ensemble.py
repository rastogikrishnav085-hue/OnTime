"""
ensemble.py
------------
The ensemble side of OnTime: a small set of different model families, a
weighted blend of them, and safe saving / loading.

Why a plain weighted average of probabilities (and not stacking)
  Every member is trained with class balancing, so they all speak the same
  "probability" language, and the blend keeps the scale the rest of the app
  is built on (risk bands at 0.30 / 0.60 / 0.80, the alert threshold, the
  planner sweep). A stacked meta-learner would also be evaluated in
  train_ensemble.py, but it outputs calibrated probabilities (about 8% for
  a typical order), which would silently break those bands.

Weights and the alert threshold are chosen on out-of-fold predictions for
the TRAINING rows only. The test rows are used once, at the end.

Explanations: reasons shown in the app are exact TreeSHAP values from the
XGBoost member (the strongest single model), not from the whole blend.
`BlendModel.named_steps` points at that member so the existing explanation
code works unchanged.
"""
import hashlib
import itertools
import json
import os
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from modeling import CATEGORICAL, NUMERIC, build_preprocessor, find_best_threshold

ENSEMBLE_DIR_NAME = "ensemble"
META_FILE = "ensemble_meta.json"
MEMBER_LABELS = {
    "xgb": "XGBoost (gradient boosting)",
    "hgb": "Histogram gradient boosting",
    "rf": "Random forest",
    "lr": "Logistic regression",
}


class EnsembleError(RuntimeError):
    """The ensemble files are missing, changed, or do not match the code."""


# ----------------------------------------------------------------- builders
def build_tree_preprocessor():
    """Numbers as they are, categories as integer codes. HistGradientBoosting
    reads the codes as categories, which is faster and usually better than
    one-hot columns for a feature like product_category (70+ values)."""
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
         CATEGORICAL),
    ])


def build_xgb(params=None, scale_pos_weight=1.0):
    p = dict(n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
             min_child_weight=5, eval_metric="auc", tree_method="hist", random_state=42, n_jobs=1)
    p.update(params or {})
    p["scale_pos_weight"] = scale_pos_weight
    return Pipeline([("pre", build_preprocessor()), ("clf", XGBClassifier(**p))])


def build_hgb(params=None):
    p = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)
    p.update(params or {})
    mask = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)
    clf = HistGradientBoostingClassifier(categorical_features=mask, class_weight="balanced", random_state=42, **p)
    return Pipeline([("pre", build_tree_preprocessor()), ("clf", clf)])


def build_lr(C=1.0):
    return Pipeline([("pre", build_preprocessor()),
                     ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))])


def build_rf(params=None):
    p = dict(n_estimators=120, max_depth=10, min_samples_leaf=30, max_features="sqrt",
             class_weight="balanced_subsample", random_state=42, n_jobs=1)
    p.update(params or {})
    return Pipeline([("pre", build_preprocessor()), ("clf", RandomForestClassifier(**p))])


def shift_odds(p, factor):
    """Map probabilities from a model trained on the NATURAL class mix to the class-balanced scale the risk bands
    use: odds are multiplied by `factor` (= negatives / positives of the training rows). Monotone, so ROC-AUC
    and PR-AUC do not change; only the numeric scale does."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    o = p / (1.0 - p) * float(factor)
    return o / (1.0 + o)


# ----------------------------------------------------------------- the blend
class BlendModel:
    """Weighted average of member probabilities, with a scikit-learn style API
    (`fit`, `predict_proba`) so it plugs into the scorer and the backtest."""

    def __init__(self, members, weights=None, explainer="xgb", odds_shift=None):
        self.odds_shift = None if odds_shift is None else float(odds_shift)
        if not members:
            raise ValueError("A blend needs at least one member.")
        self.members = dict(members)
        names = list(self.members)
        w = {n: 1.0 for n in names} if weights is None else {n: float(weights.get(n, 0.0)) for n in names}
        total = sum(w.values())
        if total <= 0 or any(v < 0 for v in w.values()):
            raise ValueError("Blend weights must be non-negative and not all zero.")
        self.weights = {n: v / total for n, v in w.items()}
        self.explainer = explainer if explainer in self.members else names[0]

    # -- scikit-learn-ish
    def fit(self, X, y):
        y = pd.Series(y).reset_index(drop=True)
        spw = float((y == 0).sum() / max((y == 1).sum(), 1))
        fitted = {}
        for name, pipe in self.members.items():
            m = clone(pipe)
            # with odds_shift the members learn the natural mix and scores are shifted afterwards
            if self.odds_shift is None and "scale_pos_weight" in m.named_steps["clf"].get_params():
                m.named_steps["clf"].set_params(scale_pos_weight=spw)
            m.fit(X, y)
            fitted[name] = m
        self.members = fitted
        return self

    def member_probabilities(self, X):
        out = {n: m.predict_proba(X)[:, 1] for n, m in self.members.items()}
        if self.odds_shift is not None:
            out = {n: shift_odds(v, self.odds_shift) for n, v in out.items()}
        return out

    def predict_proba(self, X):
        probs = self.member_probabilities(X)
        p = sum(self.weights[n] * probs[n] for n in self.members if self.weights[n] > 0)
        return np.column_stack([1.0 - p, p])

    @property
    def named_steps(self):
        """Lets prediction.explain_orders() explain the strongest member."""
        return self.members[self.explainer].named_steps


# ----------------------------------------------------------------- evaluation
def evaluate_scores(y, p, threshold):
    """The numbers the report compares models on."""
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)
    top = max(int(round(len(p) * 0.10)), 1)
    idx = np.argsort(-p)[:top]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "flagged_share": float(pred.mean()),
        "recall_at_top10pct": float(y[idx].sum() / max(y.sum(), 1)),
        "threshold": float(threshold),
    }


# ----------------------------------------------------------------- OOF + weights
def oof_predictions(pipes, X, y, cv=5, seed=42, log=print, balance=True):
    """Out-of-fold probabilities for each pipeline, on the same folds.
    Returns a DataFrame (rows = X rows, columns = member names)."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    out = {n: np.zeros(len(X)) for n in pipes}
    y = pd.Series(y).reset_index(drop=True)
    Xr = X.reset_index(drop=True)
    for k, (tr, va) in enumerate(skf.split(Xr, y), 1):
        spw = float((y.iloc[tr] == 0).sum() / (y.iloc[tr] == 1).sum())
        for n, pipe in pipes.items():
            m = clone(pipe)
            if balance and "scale_pos_weight" in m.named_steps["clf"].get_params():
                m.named_steps["clf"].set_params(scale_pos_weight=spw)
            m.fit(Xr.iloc[tr], y.iloc[tr])
            out[n][va] = m.predict_proba(Xr.iloc[va])[:, 1]
        log(f"  out-of-fold {k}/{cv} done")
    return pd.DataFrame(out)


def fit_weights(oof, y, step=0.05, metric=roc_auc_score):
    """Non-negative weights summing to 1 that maximise `metric` on the
    out-of-fold probabilities. Exhaustive search on a grid of `step`: with
    three or four members that is a few thousand evaluations, and it cannot
    get stuck in a local optimum the way a numerical optimiser can."""
    names = list(oof.columns)
    n_steps = int(round(1 / step))
    y = np.asarray(y)
    cols = {n: oof[n].to_numpy() for n in names}
    best, best_w = -1.0, None
    for combo in itertools.product(range(n_steps + 1), repeat=len(names) - 1):
        last = n_steps - sum(combo)
        if last < 0:
            continue
        ws = list(combo) + [last]
        p = sum(w * cols[n] for w, n in zip(ws, names)) / n_steps
        s = metric(y, p)
        if s > best:
            best, best_w = s, ws
    return {n: w / n_steps for n, w in zip(names, best_w)}, float(best)


def stacked_cv_score(oof, y, seed=42):
    """ROC-AUC / PR-AUC of a logistic-regression stacker over the members'
    out-of-fold log-odds, itself scored by cross-validation. For comparison
    only: it is not served (see the module docstring)."""
    eps = 1e-6
    Z = np.log(np.clip(oof.to_numpy(), eps, 1 - eps) / (1 - np.clip(oof.to_numpy(), eps, 1 - eps)))
    y = np.asarray(y)
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, y):
        meta = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])
        p[va] = meta.predict_proba(Z[va])[:, 1]
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def choose_threshold(y, p):
    """F1-optimal alert threshold on (out-of-fold) predictions."""
    thr, f1 = find_best_threshold(np.asarray(y), np.asarray(p))
    return float(thr), float(f1)


# ----------------------------------------------------------------- save / load
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions():
    import sklearn
    import xgboost
    return {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "pandas": pd.__version__, "numpy": np.__version__}


def save_ensemble(blend, out_dir, feature_columns, threshold, metrics, extra=None):
    """Write every member to `out_dir` and a metadata file with their hashes."""
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for name, pipe in blend.members.items():
        if name == "xgb":
            pre_f, bst_f = "xgb_preprocessor.joblib", "xgb_booster.json"
            joblib.dump(pipe.named_steps["pre"], os.path.join(out_dir, pre_f))
            pipe.named_steps["clf"].save_model(os.path.join(out_dir, bst_f))
            files[name] = {"preprocessor": pre_f, "booster": bst_f}
        else:
            f = f"{name}_pipeline.joblib"
            joblib.dump(pipe, os.path.join(out_dir, f), compress=3)
            files[name] = {"pipeline": f}
    for name, parts in files.items():
        for key, fname in list(parts.items()):
            parts[key] = {"file": fname, "sha256": _sha256(os.path.join(out_dir, fname))}
    meta = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": list(feature_columns),
        "weights": blend.weights, "explainer": blend.explainer, "odds_shift": blend.odds_shift, "threshold": float(threshold),
        "members": [{"name": n, "label": MEMBER_LABELS.get(n, n), "files": files[n]} for n in blend.members],
        "metrics_on_test": metrics, "library_versions": _versions(),
    }
    meta.update(extra or {})
    with open(os.path.join(out_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_ensemble(out_dir, expected_feature_columns):
    """Load the saved ensemble, or raise EnsembleError saying why not.
    Hash mismatch, a changed feature list or an unreadable member all stop
    the load; the app then keeps serving the single XGBoost model."""
    meta_path = os.path.join(out_dir, META_FILE)
    if not os.path.exists(meta_path):
        raise EnsembleError("No ensemble has been trained yet (models/ensemble/ensemble_meta.json is missing).")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if list(expected_feature_columns) != meta["feature_columns"]:
        raise EnsembleError("The ensemble was trained on a different feature list than the current code uses. Retrain it.")
    members = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for m in meta["members"]:
                parts = m["files"]
                for key, info in parts.items():
                    path = os.path.join(out_dir, info["file"])
                    if not os.path.exists(path):
                        raise EnsembleError(f"Ensemble file {info['file']} is missing.")
                    if _sha256(path) != info["sha256"]:
                        raise EnsembleError(f"{info['file']} does not match the hash recorded when the ensemble was saved.")
                if m["name"] == "xgb":
                    clf = XGBClassifier()
                    clf.load_model(os.path.join(out_dir, parts["booster"]["file"]))
                    pre = joblib.load(os.path.join(out_dir, parts["preprocessor"]["file"]))
                    members["xgb"] = Pipeline([("pre", pre), ("clf", clf)])
                else:
                    members[m["name"]] = joblib.load(os.path.join(out_dir, parts["pipeline"]["file"]))
    except EnsembleError:
        raise
    except Exception as e:  # noqa: BLE001 - any unreadable member means "no ensemble"
        raise EnsembleError(f"An ensemble member could not be loaded ({e.__class__.__name__}: {e}).") from e
    blend = BlendModel(members, weights=meta["weights"], explainer=meta.get("explainer", "xgb"), odds_shift=meta.get("odds_shift"))
    return blend, meta


# ----------------------------------------------------------------- calibration
def fit_calibration(scores, y):
    """Isotonic map from the blend's score to the observed late rate.

    The blend is trained with class balancing, so a score of 0.7 does NOT mean
    a 70% chance of being late (the real chance is far lower). The risk bands
    only need the ranking, but people read the number as a chance, so the app
    also shows this calibrated estimate. Fitted on out-of-fold training
    predictions; returned as plain lists so it can live in a JSON file."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(scores), np.asarray(y))
    xs = np.asarray(iso.X_thresholds_, dtype=float)
    ys = np.asarray(iso.y_thresholds_, dtype=float)
    return {"x": [round(float(v), 5) for v in xs], "y": [round(float(v), 5) for v in ys]}


def apply_calibration(cal, p):
    """Calibrated chance of being late for score(s) p (clipped to the fitted range)."""
    return np.interp(np.asarray(p, dtype=float), cal["x"], cal["y"])


def reliability_table(y, chance, bins=10):
    """Observed late rate versus mean predicted chance in equal-sized bins, plus the
    expected calibration error (size-weighted mean absolute gap)."""
    y, chance = np.asarray(y), np.asarray(chance, dtype=float)
    order = np.argsort(chance)
    rows, ece = [], 0.0
    for chunk in np.array_split(order, bins):
        pred, obs = float(chance[chunk].mean()), float(y[chunk].mean())
        rows.append({"n": int(len(chunk)), "predicted": round(pred, 4), "observed": round(obs, 4)})
        ece += len(chunk) / len(y) * abs(pred - obs)
    return rows, float(ece)
