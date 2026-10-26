"""Model monitoring: uncertainty, calibration, time-sliced stability, drift and outcome tracking.

Everything here is computed from the held-out queue (newest 20% of real Olist orders, never used for
training) or from files the user uploads. Nothing is generated or simulated.
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

MIN_ROWS_FOR_AUC = 30
DRIFT_FEATURES = ["promised_delivery_days", "seller_customer_distance_km", "total_price", "total_freight",
                  "total_weight_g", "approval_delay_hours", "seller_historical_delay_rate",
                  "category_historical_delay_rate", "customer_state_historical_delay_rate"]


def _auc(y, p):
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def _budget_metrics(y, p, share):
    n = len(p)
    k = max(1, int(round(n * share)))
    top = np.argsort(-p, kind="stable")[:k]
    tp = int(y[top].sum())
    return tp / k, (tp / y.sum() if y.sum() else None)


def bootstrap_ci(y, p, share=0.241, n_boot=300, seed=7):
    """95% bootstrap intervals for ROC-AUC, PR-AUC and precision/recall at an alert budget."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(y) < MIN_ROWS_FOR_AUC or len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(seed)
    out = {"roc_auc": [], "pr_auc": [], "precision": [], "recall": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0 or y[i].sum() == len(i):
            continue
        out["roc_auc"].append(roc_auc_score(y[i], p[i]))
        out["pr_auc"].append(average_precision_score(y[i], p[i]))
        pr, rc = _budget_metrics(y[i], p[i], share)
        out["precision"].append(pr)
        out["recall"].append(rc)
    point = {"roc_auc": _auc(y, p), "pr_auc": float(average_precision_score(y, p))}
    point["precision"], point["recall"] = _budget_metrics(y, p, share)
    res = {}
    for k, v in out.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        res[k] = {"estimate": float(point[k]), "low": float(lo), "high": float(hi)}
    return {"n": int(len(y)), "late": int(y.sum()), "alert_share": share, "bootstrap_samples": len(out["roc_auc"]),
            "level": 0.95, "metrics": res}


def calibration(y, p, bins=10):
    """Reliability table, Brier score and expected calibration error."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p).astype(float), 0, 1)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows, ece = [], 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        mp, ob = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(mp - ob)
        rows.append({"bin": b + 1, "n": int(m.sum()), "mean_score": round(mp, 4), "observed_late_rate": round(ob, 4)})
    return {"bins": rows, "brier": float(brier_score_loss(y, p)), "brier_baseline": float(brier_score_loss(y, np.full(len(y), y.mean()))),
            "ece": float(ece), "reading": "Close mean_score and observed_late_rate means the number can be read as a probability; "
                                         "a large gap means it is a ranking score only."}


def monthly_stability(df, score_col="probability", label_col="is_delayed", time_col="purchased_at", share=0.241):
    """Metrics per purchase month of the held-out queue. The model is NOT retrained here: this shows whether
    one fixed model holds up across the months it was tested on."""
    t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    d = df.assign(_m=t.dt.strftime("%Y-%m")).dropna(subset=["_m"])
    rows = []
    for m, g in d.groupby("_m"):
        y, p = g[label_col].astype(int).to_numpy(), g[score_col].astype(float).to_numpy()
        row = {"month": m, "orders": int(len(g)), "late": int(y.sum()), "late_rate": round(float(y.mean()), 4)}
        if len(g) >= MIN_ROWS_FOR_AUC and 0 < y.sum() < len(y):
            row["roc_auc"] = round(_auc(y, p), 4)
            row["precision_at_budget"], row["recall_at_budget"] = [round(float(v), 4) for v in _budget_metrics(y, p, share)]
        else:
            row["roc_auc"] = row["precision_at_budget"] = row["recall_at_budget"] = None
        rows.append(row)
    vals = [r["roc_auc"] for r in rows if r["roc_auc"] is not None]
    return {"months": rows, "auc_min": min(vals) if vals else None, "auc_max": max(vals) if vals else None,
            "note": "Monthly slices of the held-out queue with one fixed model; small months are noisy and shown without AUC when under 30 orders."}


def psi(reference, current, bins=10):
    """Population Stability Index. <0.10 stable, 0.10-0.25 moderate shift, >0.25 major shift."""
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy()
    if len(ref) < 20 or len(cur) < 20:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref)
    b = np.histogram(cur, edges)[0] / len(cur)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def _status(v):
    return None if v is None else ("stable" if v < 0.10 else "moderate shift" if v < 0.25 else "major shift")


def drift_report(reference_df, current_df, score_col="probability", features=None):
    feats = [f for f in (features or DRIFT_FEATURES) if f in reference_df.columns and f in current_df.columns]
    rows = []
    for f in feats + ([score_col] if score_col in reference_df.columns and score_col in current_df.columns else []):
        v = psi(reference_df[f], current_df[f])
        rows.append({"feature": f, "psi": None if v is None else round(v, 4), "status": _status(v)})
    worst = max([r["psi"] for r in rows if r["psi"] is not None], default=None)
    action = ("No action needed." if worst is None or worst < 0.10 else
              "Review the shifted inputs; compare recent outcomes with predictions before trusting alerts." if worst < 0.25 else
              "Major shift: pause automated actions, validate on recent labelled outcomes and consider retraining.")
    return {"rows": rows, "max_psi": worst, "recommendation": action, "n_reference": int(len(reference_df)), "n_current": int(len(current_df)),
            "thresholds": "PSI < 0.10 stable, 0.10 to 0.25 moderate, > 0.25 major"}


def early_vs_late_drift(df, time_col="purchased_at"):
    t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    order = t.sort_values().index
    half = len(order) // 2
    return drift_report(df.loc[order[:half]], df.loc[order[half:]])


def outcomes_report(queue, raw_csv: bytes, threshold: float):
    """Join user-supplied real outcomes (order_id + is_delayed) to the scores OnTime gave those orders."""
    try:
        out = pd.read_csv(io.BytesIO(raw_csv))
    except Exception as ex:  # noqa: BLE001
        raise ValueError(f"Could not read the outcomes file as CSV ({ex.__class__.__name__}).")
    if "order_id" not in out.columns or "is_delayed" not in out.columns:
        raise ValueError("The outcomes file needs the columns order_id and is_delayed (1/0, true/false, late/on_time).")
    out = out.drop_duplicates("order_id", keep="last")
    mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0, "late": 1, "on_time": 0, "on-time": 0, "1.0": 1, "0.0": 0}
    lab = out["is_delayed"].astype(str).str.strip().str.lower().map(mapping)
    bad = int(lab.isna().sum())
    out = out.assign(is_delayed=lab).dropna(subset=["is_delayed"])
    m = out.merge(queue[["order_id", "probability"]], on="order_id", how="inner")
    res = {"outcomes_rows": int(len(out)), "unusable_label_rows": bad, "matched_to_scored_orders": int(len(m)),
           "unmatched": int(len(out) - len(m))}
    if len(m) < MIN_ROWS_FOR_AUC or m["is_delayed"].nunique() < 2:
        res["status"] = "insufficient"
        res["message"] = f"Need at least {MIN_ROWS_FOR_AUC} matched orders containing both late and on-time cases."
        return res
    y, p = m["is_delayed"].astype(int).to_numpy(), m["probability"].astype(float).to_numpy()
    flagged = p >= threshold
    res.update({"status": "ok", "late": int(y.sum()), "late_rate": float(y.mean()), "ci": bootstrap_ci(y, p, share=float(flagged.mean()) or 0.1),
                "alerts": int(flagged.sum()),
                "alert_precision": float(y[flagged].mean()) if flagged.any() else None,
                "alert_recall": float(flagged[y == 1].mean()) if y.sum() else None,
                "calibration": calibration(y, p, bins=5)})
    return res
