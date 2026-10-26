#!/usr/bin/env python3
"""Evaluate OnTime predictions against observed, labelled delivery outcomes.

This utility deliberately does not generate labels or infer real-world quality from
synthetic benchmark files. Supply a predictions CSV (with order_id, probability,
and preferably scored_at) and a separate outcomes CSV (with order_id plus either
is_delayed or delivered_at and promised_by). A chronological holdout is required
for the report to be eligible for operational review.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _label_column(outcomes: pd.DataFrame) -> tuple[pd.Series, str]:
    if "is_delayed" in outcomes.columns:
        raw = outcomes["is_delayed"]
        if raw.dtype == bool:
            labels = raw.astype(int)
        else:
            normalized = raw.astype(str).str.strip().str.lower()
            mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0, "late": 1, "on_time": 0, "on-time": 0}
            labels = normalized.map(mapping)
        if labels.isna().any():
            raise ValueError("outcomes.is_delayed must contain only 0/1, true/false, yes/no, late/on_time values.")
        return labels.astype(int), "provided is_delayed label"
    if {"delivered_at", "promised_by"}.issubset(outcomes.columns):
        delivered = pd.to_datetime(outcomes["delivered_at"], errors="coerce", utc=True)
        promised = pd.to_datetime(outcomes["promised_by"], errors="coerce", utc=True)
        valid = delivered.notna() & promised.notna()
        if not valid.all():
            raise ValueError(f"{int((~valid).sum())} rows lack parseable delivered_at/promised_by dates; resolve these before validation.")
        return (delivered > promised).astype(int), "derived from observed delivered_at > promised_by"
    raise ValueError("Outcomes must include is_delayed or both delivered_at and promised_by columns.")


def _metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any] | None:
    if frame.empty or frame["is_delayed"].nunique() < 2:
        return None
    y = frame["is_delayed"].astype(int).to_numpy()
    p = frame["probability"].astype(float).to_numpy()
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(frame)),
        "late_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alerts": int(pred.sum()),
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn),
    }


def validate(predictions_path: str | Path, outcomes_path: str | Path, *, threshold: float = 0.5,
             min_rows: int = 200, timestamp_column: str = "scored_at") -> dict[str, Any]:
    predictions = pd.read_csv(predictions_path, dtype={"order_id": str})
    outcomes = pd.read_csv(outcomes_path, dtype={"order_id": str})
    for name, frame in (("predictions", predictions), ("outcomes", outcomes)):
        if "order_id" not in frame.columns:
            raise ValueError(f"{name} CSV must contain order_id.")
        frame["order_id"] = frame["order_id"].astype(str).str.strip()
        if frame["order_id"].eq("").any():
            raise ValueError(f"{name} contains blank order_id values.")
        if frame["order_id"].duplicated().any():
            duplicate_count = int(frame["order_id"].duplicated(keep=False).sum())
            raise ValueError(f"{name} contains duplicate order_id values ({duplicate_count} rows). Resolve duplicates; do not silently select one.")
    if "probability" not in predictions.columns:
        raise ValueError("Predictions CSV must contain probability (the model's numeric score from 0 to 1).")
    all_prediction_ids = set(predictions["order_id"])
    predictions["probability"] = pd.to_numeric(predictions["probability"], errors="coerce")
    valid_p = predictions["probability"].notna() & np.isfinite(predictions["probability"]) & predictions["probability"].between(0, 1)
    invalid_probability_ids = set(predictions.loc[~valid_p, "order_id"])
    invalid_probabilities = int((~valid_p).sum())
    predictions = predictions.loc[valid_p].copy()
    if predictions.empty:
        raise ValueError("No valid probability rows remain; values must be finite numbers between 0 and 1.")

    labels, label_source = _label_column(outcomes)
    outcomes = outcomes.copy()
    outcomes["is_delayed"] = labels.to_numpy()
    merged = predictions.merge(outcomes[["order_id", "is_delayed"] + (["delivered_at", "promised_by"] if {"delivered_at", "promised_by"}.issubset(outcomes.columns) else [])],
                               on="order_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No order_id values matched between predictions and outcomes.")

    temporal = None
    timestamp_available = timestamp_column in predictions.columns
    time_split_detail = ""
    if timestamp_available:
        time_values = pd.to_datetime(merged[timestamp_column], errors="coerce", utc=True)
        if time_values.isna().any():
            raise ValueError(f"{timestamp_column} contains missing or unparseable timestamps; chronological validation is not possible.")
        merged = merged.assign(_validation_time=time_values).sort_values("_validation_time", kind="stable")
        split_index = int(math.floor(len(merged) * 0.8))
        if split_index > 0 and split_index < len(merged):
            cutoff = merged["_validation_time"].iloc[split_index]
            # Keep all predictions sharing a timestamp on the same side of the split.
            train_period = merged[merged["_validation_time"] < cutoff]
            holdout = merged[merged["_validation_time"] >= cutoff]
        else:
            train_period, holdout = merged.iloc[0:0], merged.iloc[0:0]
        if not train_period.empty and not holdout.empty:
            temporal = {
                "method": "chronological last-20-percent holdout (timestamps are not split across partitions)",
                "cutoff_utc": holdout["_validation_time"].min().isoformat(),
                "earlier_period": {"n": int(len(train_period)), "start_utc": train_period["_validation_time"].min().isoformat(), "end_utc": train_period["_validation_time"].max().isoformat()},
                "holdout": {"start_utc": holdout["_validation_time"].min().isoformat(), "end_utc": holdout["_validation_time"].max().isoformat(), "metrics": _metrics(holdout, threshold)},
            }
            time_split_detail = "chronological holdout created"
        else:
            time_split_detail = "not enough matched rows for a chronological split"
    else:
        time_split_detail = f"missing {timestamp_column}; chronological holdout not created"

    overall = _metrics(merged, threshold)
    holdout_metrics = temporal["holdout"]["metrics"] if temporal else None
    eligible = bool(len(merged) >= min_rows and overall is not None and holdout_metrics is not None and holdout_metrics is not None)
    return {
        "report_version": 1,
        "status": "ELIGIBLE_FOR_HUMAN_REVIEW" if eligible else "INSUFFICIENT_EVIDENCE",
        "operational_approval": False,
        "message": "Metrics are computed only from matched prediction rows and observed outcome labels. Eligibility is not automatic approval; review data provenance, drift, calibration, segment performance, and intervention impact before deployment.",
        "inputs": {"predictions_file": Path(predictions_path).name, "outcomes_file": Path(outcomes_path).name, "score_column": "probability", "label_source": label_source, "timestamp_column": timestamp_column if timestamp_available else None},
        "quality": {
            "prediction_rows_in_file": int(pd.read_csv(predictions_path).shape[0]),
            "outcome_rows_in_file": int(len(outcomes)),
            "valid_prediction_rows": int(len(predictions)),
            "matched_rows": int(len(merged)),
            "unmatched_prediction_rows": int(len(predictions) - len(merged)),
            "outcome_rows_without_prediction": int((~outcomes["order_id"].isin(all_prediction_ids)).sum()),
            "outcome_rows_with_invalid_prediction": int(outcomes["order_id"].isin(invalid_probability_ids).sum()),
            "invalid_probability_rows_dropped": invalid_probabilities,
            "positive_labels": int(merged["is_delayed"].sum()),
            "negative_labels": int((merged["is_delayed"] == 0).sum()),
            "minimum_rows_required": int(min_rows),
            "timestamp_check": time_split_detail,
        },
        "overall_metrics": overall,
        "chronological_validation": temporal,
        "limitations": [
            "This report does not itself prove that interventions prevent delays.",
            "Prediction scores must be generated before the outcomes become known.",
            "A chronological holdout is necessary but not sufficient; assess calibration and performance by route, carrier, region, and time window.",
            "Do not use synthetic benchmark rows as evidence of real-world accuracy.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="CSV exported from OnTime scoring; must include order_id, probability and scored_at.")
    parser.add_argument("--outcomes", required=True, help="Separate CSV of observed outcomes keyed by order_id.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Alert threshold for the supplied scores (default: 0.5).")
    parser.add_argument("--min-rows", type=int, default=200, help="Minimum matched rows for review eligibility (default: 200).")
    parser.add_argument("--timestamp-column", default="scored_at", help="Prediction timestamp column used for chronological holdout (default: scored_at).")
    parser.add_argument("--out", default="operational_validation_report.json", help="Where to write the JSON report.")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.min_rows < 20:
        parser.error("--min-rows must be at least 20")
    try:
        report = validate(args.predictions, args.outcomes, threshold=args.threshold, min_rows=args.min_rows, timestamp_column=args.timestamp_column)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "matched_rows": report["quality"]["matched_rows"],
                      "overall_metrics": report["overall_metrics"], "holdout_metrics": (report.get("chronological_validation") or {}).get("holdout", {}).get("metrics"),
                      "report": str(out)}, indent=2, allow_nan=False))
    return 0 if report["status"] == "ELIGIBLE_FOR_HUMAN_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
