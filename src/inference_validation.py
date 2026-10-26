"""
inference_validation.py
------------------------
Guardrails for scoring ONE new order at prediction time -- this is a
different problem from data_validation.py (which checks a whole raw
dataset before training). Two failure modes handled here:

  1. Malformed input: a required field is missing, wrong type, NaN/Inf,
     or a numeric value that's structurally impossible (e.g. negative
     weight). This is REJECTED outright -- the system refuses to turn
     a broken input into a confident-looking number, which is worse
     than refusing to answer.

  2. Well-formed but out-of-distribution input: everything required is
     present and legal, but e.g. the shipping distance is far beyond
     anything the model was trained on. This IS scored (refusing to
     score anything even slightly unusual would make the system
     useless), but flagged LOW_CONFIDENCE with the specific reason,
     rather than presented with the same confidence as a routine order.

Both checks are driven by reference_stats.json, built once from the
training data by build_reference_stats() and saved alongside the model.
"""

import json
import numpy as np
import pandas as pd

REAL_NUMBER_TYPES = (int, float, np.integer, np.floating)


def is_missing(v):
    """None, NaN and blank/whitespace-only strings all mean 'no value'.
    A blank cell in a CSV arrives as NaN, not None, so checking only for
    None (as the first version did) let blank categorical fields through
    to be scored as an 'unknown category'."""
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def build_reference_stats(train_df, numeric_cols, categorical_cols, rate_cols=()):
    """
    rate_cols: numeric columns that are themselves bounded rates/
    probabilities (e.g. a historical delay rate in [0, 1]) where an
    unusually HIGH value is exactly the signal the model is supposed to
    catch, not a sign of bad input. These get a structural [0, 1]
    sanity bound instead of a percentile-based "unusual value" warning
    -- percentile-flagging them would systematically mark the riskiest,
    most-worth-flagging orders as "low confidence", which defeats the
    point of an early-warning system. This was caught by testing the
    guardrails against a real order before shipping them, not assumed.
    """
    stats = {"numeric": {}, "categorical": {}, "rate_cols": list(rate_cols)}
    for c in numeric_cols:
        s = train_df[c].dropna()
        stats["numeric"][c] = {
            "min": float(s.min()), "max": float(s.max()),
            "p01": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
        }
    for c in categorical_cols:
        stats["categorical"][c] = sorted(train_df[c].dropna().astype(str).unique().tolist())
    return stats


def save_reference_stats(stats, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_reference_stats(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_order(record, feature_columns, numeric_cols, categorical_cols, reference_stats):
    """
    record: dict of {feature_name: value} describing ONE order.
    Returns {"ok": bool, "reject_reasons": [...], "warnings": [...]}.
    ok=False means "do not attempt to score this" -- the reasons say why.
    """
    reject_reasons = []
    warnings = []

    missing = [f for f in feature_columns if f not in record or is_missing(record[f])]
    if missing:
        reject_reasons.append(f"Missing required field(s): {missing}")

    for c in numeric_cols:
        if c not in record or is_missing(record[c]):
            continue
        v = record[c]
        if isinstance(v, bool) or not isinstance(v, REAL_NUMBER_TYPES):
            reject_reasons.append(f"'{c}' must be numeric, got {type(v).__name__} ({v!r}).")
            continue
        v = float(v)
        if np.isnan(v) or np.isinf(v):
            reject_reasons.append(f"'{c}' is NaN or infinite.")
            continue
        ref = reference_stats["numeric"].get(c)
        if ref is None:
            continue
        is_rate = c in reference_stats.get("rate_cols", [])
        # structural impossibilities: negative where training data never was
        if v < 0 and ref["min"] >= 0:
            reject_reasons.append(f"'{c}'={v:g} is negative, which never occurs in training data for this field.")
            continue
        if is_rate:
            # a bounded rate/probability feature: only flag genuinely
            # impossible values, since an unusually HIGH rate here is
            # exactly the risk signal the model is meant to use, not a
            # sign of bad input.
            if v > 1.0001:
                reject_reasons.append(f"'{c}'={v:g} is a rate feature but exceeds 1.0, which is structurally impossible.")
            continue
        # Warn only on real EXTRAPOLATION: a value outside anything seen in
        # training. An earlier version warned outside the 1st-99th
        # percentile, but with ~25 numeric features that flagged 14% of
        # perfectly ordinary held-out orders (1% tails add up) versus
        # ~0% for genuine out-of-range values -- a warning that fires on
        # one order in seven carries no information.
        if v < ref["min"] or v > ref["max"]:
            warnings.append(
                f"'{c}'={v:g} is outside the range seen in training "
                f"[{ref['min']:.2f}, {ref['max']:.2f}] -- the model is extrapolating, "
                f"so this prediction may be less reliable."
            )

    for c in categorical_cols:
        if c not in record or is_missing(record[c]):
            continue
        vocab = reference_stats["categorical"].get(c, [])
        if str(record[c]) not in vocab:
            warnings.append(f"'{c}'={record[c]!r} was not seen during training -- the model will treat it as unknown.")

    return {"ok": len(reject_reasons) == 0, "reject_reasons": reject_reasons, "warnings": warnings}


def predict_with_guardrails(pipe, record, feature_columns, numeric_cols, categorical_cols,
                             reference_stats, threshold=0.5, risk_bucket_fn=None):
    """
    The single entry point a real caller would use: validate first,
    only score if validation passes, and always say which of the two
    happened rather than returning a bare number either way.
    """
    check = validate_order(record, feature_columns, numeric_cols, categorical_cols, reference_stats)
    if not check["ok"]:
        return {
            "status": "REJECTED",
            "reject_reasons": check["reject_reasons"],
            "probability": None, "risk_level": None, "confidence": None,
        }

    X = pd.DataFrame([{f: record[f] for f in feature_columns}])
    proba = float(pipe.predict_proba(X)[:, 1][0])
    result = {
        "status": "SCORED",
        "probability": round(proba, 4),
        "above_threshold": bool(proba >= threshold),
        "confidence": "LOW_CONFIDENCE" if check["warnings"] else "NORMAL",
        "warnings": check["warnings"],
    }
    if risk_bucket_fn is not None:
        result["risk_level"] = risk_bucket_fn(proba)
    return result
