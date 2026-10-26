#!/usr/bin/env python3
"""Audit the nine original Olist CSVs without modifying them.

Writes data/processed/olist_raw_audit.json and DATA_AUDIT_REPORT.md.
Use this to reproduce file hashes, dimensions, key checks and timestamp coverage.
"""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_JSON = ROOT / "data" / "processed" / "olist_raw_audit.json"
OUT_MD = ROOT / "DATA_AUDIT_REPORT.md"
TABLES = {
    "olist_customers_dataset.csv": ["customer_id"],
    "olist_geolocation_dataset.csv": [],
    "olist_order_items_dataset.csv": ["order_id", "order_item_id"],
    "olist_order_payments_dataset.csv": ["order_id", "payment_sequential"],
    "olist_order_reviews_dataset.csv": ["review_id"],
    "olist_orders_dataset.csv": ["order_id"],
    "olist_products_dataset.csv": ["product_id"],
    "olist_sellers_dataset.csv": ["seller_id"],
    "product_category_name_translation.csv": ["product_category_name"],
}
DATE_COLUMNS = {
    "olist_orders_dataset.csv": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"],
    "olist_order_reviews_dataset.csv": ["review_creation_date", "review_answer_timestamp"],
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    results = {}
    missing = []
    for name, keys in TABLES.items():
        path = RAW / name
        if not path.exists():
            missing.append(name); continue
        df = pd.read_csv(path, low_memory=False)
        key_missing = int(df[keys].isna().any(axis=1).sum()) if keys else None
        duplicate_key_rows = int(df.duplicated(keys).sum()) if keys else None
        date_ranges = {}
        for col in DATE_COLUMNS.get(name, []):
            vals = pd.to_datetime(df[col], errors="coerce")
            date_ranges[col] = {"min": str(vals.min()) if vals.notna().any() else None,
                                "max": str(vals.max()) if vals.notna().any() else None,
                                "nulls": int(vals.isna().sum())}
        results[name] = {
            "rows": int(len(df)), "columns": int(len(df.columns)), "column_names": list(df.columns),
            "sha256": sha256(path), "bytes": path.stat().st_size,
            "key_columns": keys, "rows_with_missing_key": key_missing,
            "duplicate_key_rows": duplicate_key_rows,
            "columns_with_nulls": {c: int(n) for c, n in df.isna().sum().items() if n},
            "date_ranges": date_ranges,
        }
        print(f"{name}: {len(df):,} rows x {len(df.columns)} cols; sha256={results[name]['sha256'][:12]}", flush=True)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "source": "Original Olist CSVs in data/raw; audit is read-only",
              "missing_files": missing, "all_nine_present": not missing,
              "tables": results,
              "interpretation": "Multiple rows per order are expected in items, payments, reviews and geolocation. Duplicate-key counts are checks, not automatic errors. Raw files were not modified."}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Original Olist Data Audit", "", f"Generated (UTC): {report['generated_utc']}", "",
             "This is a read-only audit of the nine original CSV files in `data/raw/`. It does not modify or normalize source data.", "",
             "| File | Rows | Columns | Key duplicate rows | SHA-256 (first 16 chars) |", "|---|---:|---:|---:|---|"]
    for name, d in results.items():
        dup = "n/a" if d["duplicate_key_rows"] is None else f"{d['duplicate_key_rows']:,}"
        lines.append(f"| `{name}` | {d['rows']:,} | {d['columns']} | {dup} | `{d['sha256'][:16]}` |")
    lines += ["", "## Interpretation", "", report["interpretation"], "",
              "For the full column list, null counts, date ranges and complete hashes, see `data/processed/olist_raw_audit.json`.", "",
              "## Evidence boundary", "", "The Olist source supports historical order and delivery-outcome analysis. It does not contain observed carrier GPS scans, hub congestion logs, or comprehensive dispatch telemetry. Those signals in the project's generated benchmarks remain synthetic."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if missing:
        raise SystemExit(f"Missing raw files: {missing}")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")

if __name__ == "__main__": main()
