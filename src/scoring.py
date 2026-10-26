"""
scoring.py
-----------
Scores a table of NEW orders end to end:

    raw order fields  ->  fill the 3 historical-rate features from the
    lookup  ->  validate every row  ->  score the valid rows in one batch
    ->  attach per-order reasons for the ones flagged as risky

Every input row gets exactly one output row with a status, so nothing
disappears silently: a row is SCORED, or REJECTED with the reasons.
"""

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
from modeling import NUMERIC, CATEGORICAL, risk_bucket
from inference_validation import validate_order, is_missing
from prediction import explain_orders

RATE_FEATURES = {
    # model feature                      : (key column in the input, lookup table)
    "seller_historical_delay_rate":         ("seller_id", "seller"),
    "category_historical_delay_rate":       ("product_category", "category"),
    "customer_state_historical_delay_rate": ("customer_state", "customer_state"),
}
THIN_HISTORY = 5  # fewer prior orders than this -> rate is noisy, say so

# What a logistics coordinator should actually DO about a flagged order,
# keyed by the factor pushing its risk up the most. A score on its own
# is not actionable; "call the seller today" is.
ACTION_BY_DRIVER = {
    "seller_historical_delay_rate": "Contact the seller to confirm dispatch date; escalate if not handed to carrier within 48h.",
    "seller_customer_distance_km": "Book an express / long-haul carrier option and track the first-mile scan closely.",
    "customer_state_historical_delay_rate": "Use the most reliable carrier for this region and warn the customer proactively.",
    "customer_state": "Use the most reliable carrier for this region and warn the customer proactively.",
    "avg_freight": "High freight cost signals a hard route; confirm carrier and pickup slot in advance.",
    "category_historical_delay_rate": "Check stock and packing readiness for this category before the carrier pickup slot.",
    "promised_delivery_days": "Promise window is tight for this route; send the customer an updated, realistic ETA now.",
    "approval_delay_hours": "Payment/approval was slow; fast-track picking so the lost time is recovered.",
    "total_weight_g": "Heavy parcel; confirm the carrier has capacity for this weight class.",
    "total_freight": "High freight cost signals a hard route; confirm carrier and pickup slot in advance.",
    "purchase_month": "Peak-season order; prioritise in the dispatch queue and pad the customer ETA.",
}
DEFAULT_ACTION = "Prioritise in today's dispatch queue and monitor tracking daily."
ACTION_BY_BAND = {
    "LOW": "No action needed.",
    "MEDIUM": "Monitor: review if no carrier scan within 3 days.",
    "HIGH": "Close to the alert line: check for a carrier scan within 2 days.",
}


def suggested_action(risk_level, flagged, top_driver=None):
    """One plain-English next step for an order.

    Unflagged orders get a band-level instruction (nothing / monitor).
    Flagged orders get an action matched to the main driver of their risk,
    prefixed with URGENT for CRITICAL ones so they sort to the top of a
    coordinator's list."""
    if risk_level is None:
        return ""
    if not flagged:
        return ACTION_BY_BAND.get(risk_level, ACTION_BY_BAND["MEDIUM"])
    action = ACTION_BY_DRIVER.get(top_driver, DEFAULT_ACTION)
    return f"URGENT: {action}" if risk_level == "CRITICAL" else action


def read_orders_csv(source):
    """Read a CSV of orders the way people actually produce them: UTF-8 with
    or without Excel's byte-order mark, or a Windows ANSI file; comma- or
    semicolon-separated. `source` is a path or raw bytes."""
    import io
    raw = source if isinstance(source, (bytes, bytearray)) else open(source, "rb").read()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    first = text.split("\n", 1)[0]
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def required_input_columns():
    """What a caller must supply: every model feature except the three
    rates (those are filled from the lookup unless supplied directly)."""
    return [c for c in FEATURE_COLUMNS if c not in RATE_FEATURES]


def fill_rate_features(orders, lookup):
    """Returns (orders_with_rates, warnings_per_row, info_per_row)."""
    orders = orders.copy()
    warns = [[] for _ in range(len(orders))]
    info = [[] for _ in range(len(orders))]
    for col, (key_col, table) in RATE_FEATURES.items():
        values = []
        for i in range(len(orders)):
            supplied = orders[col].iloc[i] if col in orders.columns else np.nan
            if not is_missing(supplied):
                values.append(supplied)
                continue
            key = orders[key_col].iloc[i] if key_col in orders.columns else None
            entry = None if (key is None or is_missing(key)) else lookup[table].get(str(key))
            if entry is None:
                values.append(lookup["global_rate"])
                why = "no seller_id given" if key_col == "seller_id" and (key is None or is_missing(key)) \
                    else f"{key_col}={key!r} has no history"
                warns[i].append(f"{col}: {why}; used the platform-wide rate ({lookup['global_rate']:.3f}).")
            else:
                values.append(entry["rate"])
                if entry["n_orders"] < THIN_HISTORY:
                    info[i].append(f"{key_col}={key!r} has only {entry['n_orders']} past order(s); its rate is noisy.")
        orders[col] = values
    return orders, warns, info


def score_orders(orders, model, threshold, reference_stats, lookup, top_n=3):
    if not isinstance(orders, pd.DataFrame) or len(orders) == 0:
        raise ValueError("No orders to score: the input is empty.")

    missing_cols = [c for c in required_input_columns() if c not in orders.columns]
    if missing_cols:
        raise ValueError(
            f"Input is missing {len(missing_cols)} required column(s): {missing_cols}. "
            "Run with --template to see the expected layout."
        )

    orders = orders.reset_index(drop=True)
    filled, rate_warns, rate_info = fill_rate_features(orders, lookup)
    model_features = list(FEATURE_COLUMNS)
    enriched_warns = [None] * len(filled)
    if getattr(model, "extra_columns", None):
        extras, enriched_warns = model.serving_extras(filled)
        filled = pd.concat([filled, extras], axis=1)
        model_features += list(model.extra_columns)

    n = len(filled)
    out = pd.DataFrame({
        "row": range(n),
        "order_id": filled["order_id"] if "order_id" in filled.columns else [None] * n,
        "status": "REJECTED", "probability": np.nan, "risk_level": None, "flagged": None,
        "confidence": None, "reasons": "", "suggested_action": "", "warnings": "", "reject_reasons": "", "notes": "",
    })

    records = filled[FEATURE_COLUMNS].to_dict("records")
    valid_idx = []
    for i, rec in enumerate(records):
        check = validate_order(rec, FEATURE_COLUMNS, NUMERIC, CATEGORICAL, reference_stats)
        all_warnings = rate_warns[i] + check["warnings"]
        if enriched_warns[i]:
            all_warnings.append(enriched_warns[i])
        out.at[i, "warnings"] = " | ".join(all_warnings)
        out.at[i, "notes"] = " | ".join(rate_info[i])
        if check["ok"]:
            valid_idx.append(i)
            out.at[i, "confidence"] = "LOW_CONFIDENCE" if all_warnings else "NORMAL"
        else:
            out.at[i, "reject_reasons"] = " | ".join(check["reject_reasons"])

    if valid_idx:
        X = filled.loc[valid_idx, model_features].copy()
        for c in NUMERIC:
            X[c] = X[c].astype(float)  # safe: every value in these rows passed validation
        proba = model.predict_proba(X)[:, 1]
        flagged = proba >= threshold
        out.loc[valid_idx, "status"] = "SCORED"
        out.loc[valid_idx, "probability"] = np.round(proba, 4)
        out.loc[valid_idx, "risk_level"] = [risk_bucket(p) for p in proba]
        out.loc[valid_idx, "flagged"] = flagged
        drivers = {}
        if flagged.any():
            if hasattr(model, "explain"):
                why = model.explain(X[flagged], top_n=top_n)
            else:
                why = explain_orders(model, X[flagged], top_n=top_n)
            for i, reasons in zip(np.array(valid_idx)[flagged], why):
                out.at[i, "reasons"] = "; ".join(f"{label} (+{c:.2f})" for _, c, label in reasons)
                drivers[i] = reasons[0][0] if reasons else None
        for i, is_flagged in zip(valid_idx, flagged):
            if hasattr(model, "action_for"):
                out.at[i, "suggested_action"] = model.action_for(out.at[i, "risk_level"], bool(is_flagged), drivers.get(i))
            else:
                out.at[i, "suggested_action"] = suggested_action(
                    out.at[i, "risk_level"], bool(is_flagged), drivers.get(i))
    return out


def summarise(scored):
    lines = [f"Rows: {len(scored)}   scored: {(scored.status == 'SCORED').sum()}   "
             f"rejected: {(scored.status == 'REJECTED').sum()}"]
    ok = scored[scored.status == "SCORED"]
    if len(ok):
        lines.append(f"Flagged as at-risk: {int(ok.flagged.sum())} of {len(ok)}")
        lines.append("Risk levels: " + ", ".join(
            f"{k}={int(v)}" for k, v in ok.risk_level.value_counts().reindex(
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0).items()))
        lines.append(f"Low-confidence rows: {int((ok.confidence == 'LOW_CONFIDENCE').sum())}")
        urgent = ok.suggested_action.str.startswith("URGENT").sum() if "suggested_action" in ok else 0
        lines.append(f"Urgent actions: {int(urgent)}")
    return "\n".join(lines)
