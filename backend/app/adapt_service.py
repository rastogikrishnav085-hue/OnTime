"""
adapt_service.py - the server side of "Adapt to your data".

Flow
  1. preview(): the user uploads a CSV; it is stored, columns are matched to
     OnTime's concepts, and a suggested mapping (editable in the page) is
     returned together with whether the file carries real outcomes.
  2. run(): with the confirmed mapping, the file is turned into the model's
     inputs. It is scored by the Olist model ("transfer"); if outcomes exist
     the transfer is checked against them; if asked, a model is trained on
     the user's own history and compared on the SAME test rows.
  3. score_with(): a trained adapted model scores a further file.
"""
import json
import os
import re
import time
import uuid

import numpy as np
import pandas as pd

from .engine import ROOT, FORM_FIELDS  # noqa: F401  (ROOT also puts src/ on the path)
import adapter as ad                    # noqa: E402
from modeling import risk_bucket        # noqa: E402
from scoring import read_orders_csv     # noqa: E402

ADAPT_DIR = os.environ.get("ONTIME_ADAPT_DIR", os.path.join(ROOT, "backend", "adapted"))
MAX_ROWS = int(os.environ.get("ONTIME_MAX_ADAPT_ROWS", "20000"))
MAX_UPLOADS_KEPT = 12
TRANSFER_SAMPLE = 8000


class AdaptError(ValueError):
    """A problem the user can fix; the message says how."""


# ------------------------------------------------------------------ storage
def _uploads():
    d = os.path.join(ADAPT_DIR, "uploads")
    os.makedirs(d, exist_ok=True)
    return d


def _models_dir():
    d = os.path.join(ADAPT_DIR, "models")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_id(s):
    if not re.fullmatch(r"[a-z0-9\-]{4,60}", s or ""):
        raise AdaptError("Unknown data set id.")
    return s


def _store_upload(raw, filename):
    d = _uploads()
    files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")), key=os.path.getmtime)
    for old in files[: max(0, len(files) - MAX_UPLOADS_KEPT + 1)]:
        try:
            os.remove(old)
        except OSError:
            pass
    ds_id = uuid.uuid4().hex[:12]
    with open(os.path.join(d, ds_id + ".csv"), "wb") as f:
        f.write(raw)
    with open(os.path.join(d, ds_id + ".json"), "w", encoding="utf-8") as f:
        json.dump({"filename": filename}, f)
    return ds_id


def _load_upload(ds_id):
    path = os.path.join(_uploads(), _safe_id(ds_id) + ".csv")
    if not os.path.exists(path):
        raise AdaptError("That upload is no longer on the server. Upload the file again.")
    df = read_orders_csv(path)
    name = ds_id
    try:
        name = json.load(open(os.path.join(_uploads(), ds_id + ".json"), encoding="utf-8"))["filename"]
    except (OSError, ValueError, KeyError):
        pass
    return df, name


# ------------------------------------------------------------------ helpers
def feature_weights(engine):
    """Importance of each model input (summed over one-hot columns), from the production XGBoost."""
    cached = getattr(engine, "_adapt_weights", None)
    if cached:
        return cached
    from prediction import _transformed_to_original
    pipe = engine.models["xgboost"][0]
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    imp = clf.feature_importances_
    names = _transformed_to_original(pre)
    w = {}
    for n, v in zip(names, imp):
        w[n] = w.get(n, 0.0) + float(v)
    engine._adapt_weights = w
    return w


def _concept_catalog():
    out = []
    for c in ad.CONCEPTS:
        out.append({"key": c["key"], "label": c["label"], "kind": c["kind"], "units": c.get("unit_options"),
                    "bases": c.get("basis_options"), "required": c.get("required", False)})
    return out


def _clean_mapping(df, mapping):
    clean = {}
    for key, entry in (mapping or {}).items():
        if key not in ad.CONCEPT_BY_KEY:
            raise AdaptError(f"'{key}' is not something OnTime can map.")
        src = (entry or {}).get("source")
        if src in (None, ""):
            continue
        if src not in df.columns:
            raise AdaptError(f"The file has no column called '{src}'.")
        e = {"source": src}
        c = ad.CONCEPT_BY_KEY[key]
        if (entry.get("unit") in (c.get("unit_options") or [])):
            e["unit"] = entry["unit"]
        if (entry.get("basis") in (c.get("basis_options") or [])):
            e["basis"] = entry["basis"]
        clean[key] = e
    used = [e["source"] for e in clean.values()]
    return clean


def _series_safe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return str(v)[:60]


# ------------------------------------------------------------------ step 1
def preview(engine, raw, filename):
    if not raw or not raw.strip():
        raise AdaptError("The file is empty.")
    try:
        df = read_orders_csv(raw)
    except Exception as ex:  # noqa: BLE001
        raise AdaptError(f"Could not read the file as CSV ({ex.__class__.__name__}). Save it as comma-separated .csv and try again.")
    if len(df.columns) < 2:
        raise AdaptError("The file has only one column. Check that it is a comma- or semicolon-separated CSV with a header row.")
    if len(df) < 20:
        raise AdaptError(f"The file has only {len(df)} rows. At least 20 are needed to map and score it.")
    if len(df) > MAX_ROWS:
        raise AdaptError(f"The file has {len(df):,} rows; the limit is {MAX_ROWS:,}. Use a sample or split it (set ONTIME_MAX_ADAPT_ROWS in .env to raise the limit).")
    ds_id = _store_upload(raw, filename or "upload.csv")
    mapping = ad.suggest_mapping(df)
    outcome = _outcome_info(df, mapping)
    cols = []
    for c in df.columns:
        s = df[c].dropna()
        cols.append({"name": c, "sample": [_series_safe(v) for v in s.head(3).tolist()], "n_missing": int(df[c].isna().sum())})
    return {"dataset_id": ds_id, "filename": filename, "n_rows": int(len(df)), "n_columns": int(len(df.columns)), "columns": cols,
            "suggested": mapping, "concepts": _concept_catalog(), "outcome": outcome,
            "min_rows_to_learn": 300}


def _outcome_info(df, mapping):
    late, how = ad.detect_outcome(df, mapping)
    if late is None:
        return {"found": False, "how": how}
    ok = late.dropna()
    return {"found": True, "how": how, "n_labelled": int(len(ok)), "late_rate": round(float(ok.mean()), 4),
            "can_learn": bool(len(ok) >= 300 and ok.sum() >= 30)}


# ------------------------------------------------------------------ metrics
def _rank_metrics(y, p, thr):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "late_rate": round(float(y.mean()), 4)}
    if len(np.unique(y)) < 2 or len(y) < 30:
        out["note"] = "Too few late (or on-time) orders with an outcome to measure accuracy."
        return out
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    top = max(int(round(len(p) * 0.10)), 1)
    idx = np.argsort(-p)[:top]
    out.update({"roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
                "precision": round(tp / max(int(pred.sum()), 1), 4), "recall": round(tp / max(int(y.sum()), 1), 4),
                "flagged_share": round(float(pred.mean()), 4),
                "recall_at_top10pct": round(float(y[idx].sum() / max(y.sum(), 1)), 4)})
    return out


def _verdict(auc):
    if auc is None:
        return {"level": "unknown", "text": "No outcomes in the file, so the transfer could not be measured."}
    if auc >= 0.75:
        return {"level": "good", "text": "The Olist model transfers well to this data. You can use it as it is."}
    if auc >= 0.65:
        return {"level": "fair", "text": "The Olist model carries some signal over to this data, but a model trained on your own history will usually do better."}
    return {"level": "weak", "text": "The Olist model does not transfer well to this data (close to guessing). Train a model on your own history instead."}


# ------------------------------------------------------------------ step 2
def run(engine, dataset_id, mapping, train=False, name=None, model_key=None):
    df, filename = _load_upload(dataset_id)
    mapping = _clean_mapping(df, mapping)
    if not mapping:
        raise AdaptError("No columns are mapped. Choose at least the order date, the route (state or distance) and the promised date or window.")
    typical = engine.meta["typical_order"]
    t0 = time.time()
    table, info = ad.apply_mapping(df, mapping, typical, feature_weights(engine))
    late, how = ad.detect_outcome(df, mapping)
    key, _model, thr = engine.resolve_model(model_key)

    # ---- transfer: score with the Olist model
    n = len(table)
    samp_idx = np.arange(n) if n <= TRANSFER_SAMPLE else np.sort(np.random.RandomState(1).choice(n, TRANSFER_SAMPLE, replace=False))
    scored = engine.score_batch(table.iloc[samp_idx].reset_index(drop=True), key)
    ok = scored[scored.status == "SCORED"]
    rejected = scored[scored.status == "REJECTED"]
    reasons = {}
    for r in rejected.reject_reasons:
        k = re.sub(r"'[^']*'=[-\d.eE+]+|\[[^\]]*\]|\([^)]*\)", "…", str(r))[:110]
        reasons[k] = reasons.get(k, 0) + 1
    transfer = {
        "model": key, "model_label": engine.models_info()[0 if key == "xgboost" else 1]["label"], "threshold": thr,
        "rows_scored_for_check": int(len(scored)), "scored": int(len(ok)), "rejected": int(len(rejected)),
        "low_confidence": int((ok.confidence == "LOW_CONFIDENCE").sum()) if len(ok) else 0,
        "flagged_share": round(float(ok.flagged.astype(bool).mean()), 4) if len(ok) else None,
        "bands": {b: int((ok.risk_level == b).sum()) for b in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
        "reject_reasons": sorted(({"reason": k, "n": v} for k, v in reasons.items()), key=lambda r: -r["n"])[:4],
    }
    transfer["metrics"] = None
    if late is not None and len(ok):
        yy = late.iloc[samp_idx].reset_index(drop=True)[scored.status == "SCORED"]
        has = yy.notna()
        if has.sum() >= 30:
            transfer["metrics"] = _rank_metrics(yy[has], ok.probability[has.values].astype(float), thr)
    transfer["verdict"] = _verdict(transfer["metrics"] and transfer["metrics"].get("roc_auc"))

    result = {"dataset_id": dataset_id, "filename": filename, "n_rows": int(n), "mapping": mapping, "coverage": {
        "n_real": info["n_real"], "n_features": info["n_features"], "importance_coverage": info["importance_coverage"],
        "provenance": info["provenance"], "defaulted": info["defaulted"], "notes": info["notes"]},
        "outcome": ({"found": True, "how": how, "n_labelled": int(late.notna().sum()), "late_rate": round(float(late.mean()), 4)}
                    if late is not None else {"found": False, "how": how}),
        "transfer": transfer, "adapted": None, "elapsed_s": None}
    low_cov = info["importance_coverage"] < 0.5
    result["coverage"]["warning"] = (
        f"Only {info['importance_coverage'] * 100:.0f}% of the model's weight rests on inputs that came from your file; the rest are "
        "typical-order values. Treat the scores as indicative and map more columns if you can.") if low_cov else None

    # ---- optional: learn from the user's own history
    if train:
        if late is None:
            raise AdaptError(how)
        pt = ad.to_datetime(df[mapping["purchase_time"]["source"]]) if "purchase_time" in mapping else None
        dt = ad.to_datetime(df[mapping["delivered_time"]["source"]]) if "delivered_time" in mapping else None
        pipe, spec, rep = ad.train_adaptive(table, info, late, pt, dt)
        rows, y_te, p_te = rep.pop("_test_rows")
        # the Olist model on exactly the same test rows (for a fair comparison)
        cmp_rows = rows if len(rows) <= 6000 else rows[:6000]
        sc = engine.score_batch(table.iloc[cmp_rows].reset_index(drop=True), key)
        good = (sc.status == "SCORED").values
        base = None
        if good.sum() >= 30:
            base = _rank_metrics(np.asarray(y_te[: len(cmp_rows)])[good], sc.probability[good].astype(float), thr)
        mid = re.sub(r"[^a-z0-9]+", "-", (name or os.path.splitext(filename or "data")[0]).lower()).strip("-")[:30] or "data"
        mid = f"{mid}-{uuid.uuid4().hex[:6]}"
        ad.save_adapted(os.path.join(_models_dir(), mid), pipe, spec, mapping, name or filename)
        rep["same_rows_olist_model"] = base
        rep["model_id"] = mid
        rep["name"] = name or filename
        rep["gain_auc"] = round(rep["roc_auc"] - base["roc_auc"], 4) if base and "roc_auc" in base else None
        rep["verdict"] = _compare_verdict(rep, base)
        result["adapted"] = rep
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def _compare_verdict(rep, base):
    if not base or "roc_auc" not in base:
        return "The model trained on your history was tested on your most recent orders."
    d = rep["roc_auc"] - base["roc_auc"]
    split = "your most recent 20% of orders" if rep["split"] == "time" else "a random 20% of your orders"
    if d > 0.02:
        return f"On {split}, the model trained on your own history beats the Olist model by {d:.3f} ROC-AUC. Use it for this data."
    if d < -0.02:
        return f"On {split}, the Olist model is {-d:.3f} ROC-AUC better than the one trained on your history (probably too little history). Keep the Olist model for now."
    return f"On {split}, the two models are about equal ({d:+.3f} ROC-AUC). The Olist model is the simpler choice; retrain when you have more history."


# ------------------------------------------------------------------ step 3: adapted models
def list_models():
    out = []
    d = _models_dir()
    for mid in sorted(os.listdir(d)):
        p = os.path.join(d, mid, "adapted.json")
        if os.path.exists(p):
            try:
                m = json.load(open(p, encoding="utf-8"))
                out.append({"model_id": mid, "name": m.get("name"), "saved_at": m.get("saved_at_utc"), "metrics": m["spec"].get("metrics"),
                            "split": m["spec"].get("split"), "features": m["spec"]["numeric"] + m["spec"]["categorical"]})
            except (OSError, ValueError, KeyError):
                continue
    return sorted(out, key=lambda r: r.get("saved_at") or "", reverse=True)


def delete_model(model_id):
    import shutil
    d = os.path.join(_models_dir(), _safe_id(model_id))
    if not os.path.isdir(d):
        raise AdaptError("No such adapted model.")
    shutil.rmtree(d)


def score_with(engine, model_id, raw):
    d = os.path.join(_models_dir(), _safe_id(model_id))
    if not os.path.exists(os.path.join(d, "adapted.json")):
        raise AdaptError("No such adapted model.")
    pipe, meta = ad.load_adapted(d)
    try:
        df = read_orders_csv(raw)
    except Exception as ex:  # noqa: BLE001
        raise AdaptError(f"Could not read the file as CSV ({ex.__class__.__name__}).")
    if len(df) > MAX_ROWS:
        raise AdaptError(f"The file has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
    mapping = {k: v for k, v in meta["mapping"].items() if v["source"] in df.columns}
    missing = [v["source"] for v in meta["mapping"].values() if v["source"] not in df.columns]
    if "purchase_time" not in mapping and any(f in meta["spec"]["numeric"] for f in ("purchase_month", "purchase_dow")):
        raise AdaptError("This file has no order date column with the same name as the training file, but the model needs it.")
    table, info = ad.apply_mapping(df, mapping, engine.meta["typical_order"], feature_weights(engine))
    p = ad.predict_adaptive(pipe, meta["spec"], table)
    thr = meta["spec"]["threshold"]
    rows = []
    for i in range(len(table)):
        rows.append({"row": i, "order_id": table["order_id"].iloc[i], "probability": round(float(p[i]), 4),
                     "risk_level": risk_bucket(p[i]), "flagged": bool(p[i] >= thr)})
    return {"model_id": model_id, "name": meta.get("name"), "threshold": thr, "n_rows": len(rows),
            "flagged": int(sum(r["flagged"] for r in rows)), "columns_missing_from_file": missing, "rows": rows}
