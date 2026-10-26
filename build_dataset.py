"""
build_dataset.py
-----------------
One command from the REAL Olist files to the table every model is trained on.

    .venv\\Scripts\\activate
    python build_dataset.py

    data/raw/        the 9 original Olist CSV files, untouched (real data)
         |  validate (src/data_validation.py), join, clean, label
         v
    data/interim/    olist_joined.csv            one row per delivered order, all joined columns (no features yet)
         |  core features (src/features.py) + enriched features (src/enriched_features.py)
         v
    data/processed/  ontime_model_ready.csv      what the models read: core + enriched columns
                     data_quality_report.json    row counts, missing values, label rate, leakage self-checks
                     raw_manifest.json           SHA-256 of every raw file, before and after the build
    models/enriched/context.json                 latest history snapshot used to score hand-typed orders

Simulated data is NOT used here. It lives in data/synthetic/ and has its own scripts.
Runs in about a minute.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_validation import format_report, validate_raw_dataset                      # noqa: E402
from enriched_features import (EXTRA_COLUMNS, build_extra_features, build_serving_context,   # noqa: E402
                               save_context)
from features import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_table              # noqa: E402
from preprocessing import build_master_table                                          # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
INTERIM = os.path.join(ROOT, "data", "interim")
PROCESSED = os.path.join(ROOT, "data", "processed")
HOLIDAYS = os.path.join(ROOT, "data", "external", "holidays_brazil.csv")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_hashes(raw_dir):
    return {f: sha256(os.path.join(raw_dir, f)) for f in sorted(os.listdir(raw_dir)) if f.endswith(".csv")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(INTERIM, exist_ok=True)
    os.makedirs(PROCESSED, exist_ok=True)
    hashes_before = raw_hashes(args.raw)

    if not args.skip_validation:
        rep = validate_raw_dataset(args.raw)
        print(format_report(rep))
        if rep["verdict"] == "REJECT":
            raise SystemExit("Raw data rejected, nothing built.")

    print("Joining the raw tables ...")
    joined, _all_orders = build_master_table(args.raw)
    items = pd.read_csv(os.path.join(args.raw, "olist_order_items_dataset.csv"), usecols=["order_id", "shipping_limit_date"])
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"], errors="coerce")
    joined = joined.merge(items.groupby("order_id")["shipping_limit_date"].min().rename("shipping_limit_min"), on="order_id", how="left")
    joined.to_csv(os.path.join(INTERIM, "olist_joined.csv"), index=False)
    print(f"  data/interim/olist_joined.csv  {joined.shape}")

    print("Building core features ...")
    core = build_feature_table(joined)
    print("Building enriched features (history, network state, promise, place) ...")
    extra = build_extra_features(core.reset_index(drop=True), HOLIDAYS)
    core = core.reset_index(drop=True)
    ready = pd.concat([core[["order_id", TARGET_COLUMN, "order_purchase_timestamp"] + FEATURE_COLUMNS], extra], axis=1)
    # NOTE: row order is kept exactly as the project always had it, so the stratified 80/20 split (random_state 42)
    # is identical to the one every earlier model and notebook used.

    # ---- self-checks: fail loudly rather than ship a quietly wrong table
    assert ready["order_id"].is_unique, "duplicate order ids"
    assert set(ready[TARGET_COLUMN].unique()) <= {0, 1}
    assert not ready[FEATURE_COLUMNS].isna().any().any(), "core features must have no missing values"
    names = " ".join(ready.columns).lower()
    for bad in ("delivered_customer", "delay_days", "review", "score"):
        assert bad not in names, f"leaky column name in model table: {bad}"
    ready.to_csv(os.path.join(PROCESSED, "ontime_model_ready.csv"), index=False, float_format="%.7g")
    print(f"  data/processed/ontime_model_ready.csv  {ready.shape}")

    ctx = build_serving_context(core, extra)
    os.makedirs(os.path.join(ROOT, "models", "enriched"), exist_ok=True)
    save_context(ctx, os.path.join(ROOT, "models", "enriched", "context.json"))

    hashes_after = raw_hashes(args.raw)
    assert hashes_before == hashes_after, "raw files changed during the build (they must never be modified)"
    with open(os.path.join(PROCESSED, "raw_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"checked_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
                   "raw_dir": "data/raw", "unchanged_by_build": True, "sha256": hashes_after}, f, indent=2)

    miss = ready[EXTRA_COLUMNS].isna().mean()
    quality = {
        "built_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "source": "REAL Olist raw files in data/raw (no simulated data)",
        "rows": int(len(ready)), "late_rate": round(float(ready[TARGET_COLUMN].mean()), 4),
        "first_purchase": str(ready["order_purchase_timestamp"].min()), "last_purchase": str(ready["order_purchase_timestamp"].max()),
        "core_features": len(FEATURE_COLUMNS), "enriched_features": len(EXTRA_COLUMNS),
        "enriched_share_missing": {k: round(float(v), 4) for k, v in miss[miss > 0].items()},
        "leak_rules": ["history uses only orders handed over / delivered strictly before this order's purchase time",
                       "network state uses only 'not yet delivered at purchase time'",
                       "no delivery date, delay, review or tracking column is a feature"],
    }
    with open(os.path.join(PROCESSED, "data_quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)
    print(f"Done in {time.time() - t0:.0f}s. Next: python train_enriched.py")


if __name__ == "__main__":
    main()
