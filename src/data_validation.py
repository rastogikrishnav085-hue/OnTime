"""
data_validation.py
-------------------
Gatekeeper for the raw input, run BEFORE preprocessing.py touches
anything. If the person hands this project a folder that's missing a
file, has the wrong columns (e.g. a renamed export, or files swapped),
or is too corrupted to build a trustworthy is_delayed label from, this
says so explicitly and returns a REJECT verdict with reasons -- instead
of preprocessing.py silently doing its best with whatever's there and
producing a master table that looks fine but is quietly wrong.

Usage:
    from data_validation import validate_raw_dataset
    report = validate_raw_dataset("../data/raw")
    if report["verdict"] == "REJECT":
        raise SystemExit("Dataset rejected:\\n" + format_report(report))
"""

import os
import pandas as pd
import numpy as np

REQUIRED_FILES = {
    "olist_orders_dataset.csv": {
        "required_cols": [
            "order_id", "customer_id", "order_status", "order_purchase_timestamp",
            "order_approved_at", "order_delivered_carrier_date",
            "order_delivered_customer_date", "order_estimated_delivery_date",
        ],
        "min_rows": 1000,
    },
    "olist_order_items_dataset.csv": {
        "required_cols": ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
        "min_rows": 1000,
    },
    "olist_customers_dataset.csv": {
        "required_cols": ["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
        "min_rows": 1000,
    },
    "olist_sellers_dataset.csv": {
        "required_cols": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
        "min_rows": 10,
    },
    "olist_products_dataset.csv": {
        "required_cols": ["product_id", "product_category_name"],
        "min_rows": 100,
    },
    "product_category_name_translation.csv": {
        "required_cols": ["product_category_name", "product_category_name_english"],
        "min_rows": 1,
    },
    "olist_order_payments_dataset.csv": {
        "required_cols": ["order_id", "payment_type", "payment_installments", "payment_value"],
        "min_rows": 1000,
    },
    "olist_geolocation_dataset.csv": {
        "required_cols": ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
        "min_rows": 100,
    },
}

# files the pipeline can technically limp on without (with a WARN, not a FAIL)
OPTIONAL_FILES = {"olist_order_reviews_dataset.csv"}


def _check(name, status, message):
    assert status in ("OK", "WARN", "FAIL")
    return {"check": name, "status": status, "message": message}


# numeric columns to type-check per file, beyond the schema/row checks above
NUMERIC_COLUMNS_TO_CHECK = {
    "olist_order_items_dataset.csv": ["price", "freight_value"],
    "olist_order_payments_dataset.csv": ["payment_value", "payment_installments"],
    "olist_geolocation_dataset.csv": ["geolocation_lat", "geolocation_lng"],
    "olist_products_dataset.csv": ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
}


def _check_numeric_columns(filename, df, checks):
    """
    Coerces the columns that the pipeline treats as numeric and reports
    how many values didn't survive that coercion, instead of letting a
    corrupted column (e.g. text sneaked into a price field) crash a
    later comparison with a raw, unhandled pandas TypeError. Returns the
    coerced (numeric, NaN-for-bad-values) versions of each column so
    later checks can use them safely rather than the raw column.
    """
    coerced = {}
    for col in NUMERIC_COLUMNS_TO_CHECK.get(filename, []):
        if col not in df.columns:
            continue
        raw_missing = df[col].isna().mean()
        numeric = pd.to_numeric(df[col], errors="coerce")
        newly_bad = numeric.isna().mean() - raw_missing
        if newly_bad > 0.01:
            checks.append(_check(
                f"types:{filename}.{col}", "FAIL" if newly_bad > 0.2 else "WARN",
                f"{newly_bad:.1%} of non-null values in '{col}' are not valid numbers "
                f"(e.g. text where a price/weight/coordinate was expected).",
            ))
        coerced[col] = numeric
    return coerced


def _check_file_presence_and_schema(raw_dir, filename, spec, checks):
    path = os.path.join(raw_dir, filename)
    if not os.path.exists(path):
        sev = "WARN" if filename in OPTIONAL_FILES else "FAIL"
        checks.append(_check(f"file:{filename}", sev, "File not found."))
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        checks.append(_check(f"file:{filename}", "FAIL", f"Could not be parsed as CSV ({e.__class__.__name__})."))
        return None
    if df.shape[1] == 1 and len(REQUIRED_FILES.get(filename, {}).get("required_cols", [])) > 1:
        checks.append(_check(f"schema:{filename}", "FAIL",
                              "File parsed as a single column -- likely wrong delimiter or wrong file entirely."))
        return df

    missing = [c for c in spec["required_cols"] if c not in df.columns]
    if missing:
        checks.append(_check(f"schema:{filename}", "FAIL", f"Missing required column(s): {missing}"))
    else:
        checks.append(_check(f"schema:{filename}", "OK", "All required columns present."))

    if len(df) < spec["min_rows"]:
        checks.append(_check(f"rows:{filename}", "FAIL",
                              f"Only {len(df):,} rows found, expected at least {spec['min_rows']:,}."))
    else:
        checks.append(_check(f"rows:{filename}", "OK", f"{len(df):,} rows."))
    return df


def validate_raw_dataset(raw_dir):
    checks = []
    tables = {}
    coerced_numeric = {}
    for filename, spec in REQUIRED_FILES.items():
        df = _check_file_presence_and_schema(raw_dir, filename, spec, checks)
        tables[filename] = df
        if df is not None:
            coerced_numeric[filename] = _check_numeric_columns(filename, df, checks)

    orders = tables.get("olist_orders_dataset.csv")
    items = tables.get("olist_order_items_dataset.csv")

    # can't run any further structural checks without these two -- bail early
    if orders is None or items is None or "order_id" not in (orders.columns if orders is not None else []):
        return _finalize(checks)

    # --- duplicate primary keys ---
    dup_orders = orders["order_id"].duplicated().sum()
    checks.append(_check(
        "duplicates:orders.order_id",
        "FAIL" if dup_orders > 0 else "OK",
        f"{dup_orders:,} duplicate order_id values (each order must appear once)." if dup_orders
        else "No duplicate order_id values.",
    ))

    # --- referential integrity: items must point at real orders ---
    if "order_id" in items.columns:
        orphan_rate = (~items["order_id"].isin(orders["order_id"])).mean()
        sev = "FAIL" if orphan_rate > 0.5 else ("WARN" if orphan_rate > 0.02 else "OK")
        checks.append(_check(
            "referential:items->orders", sev,
            f"{orphan_rate:.1%} of order_items rows reference an order_id absent from orders.",
        ))

    products = tables.get("olist_products_dataset.csv")
    if products is not None and "product_id" in items.columns and "product_id" in products.columns:
        orphan_rate = (~items["product_id"].isin(products["product_id"])).mean()
        sev = "FAIL" if orphan_rate > 0.5 else ("WARN" if orphan_rate > 0.05 else "OK")
        checks.append(_check(
            "referential:items->products", sev,
            f"{orphan_rate:.1%} of order_items rows reference a product_id absent from products.",
        ))

    # --- date sanity ---
    date_cols = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                 "order_delivered_customer_date", "order_estimated_delivery_date"]
    for col in date_cols:
        if col not in orders.columns:
            continue
        raw_missing = orders[col].isna().mean()
        parsed = pd.to_datetime(orders[col], errors="coerce")
        newly_bad = parsed.isna().mean() - raw_missing
        if newly_bad > 0.05:
            checks.append(_check(f"dates:{col}", "WARN",
                                  f"{newly_bad:.1%} of non-null values in {col} could not be parsed as dates."))

    if all(c in orders.columns for c in ["order_purchase_timestamp", "order_delivered_customer_date"]):
        purchase = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
        delivered = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
        both = purchase.notna() & delivered.notna()
        if both.sum() > 0:
            impossible = (delivered[both] < purchase[both]).mean()
            checks.append(_check(
                "dates:sequence", "WARN" if impossible > 0.001 else "OK",
                f"{impossible:.2%} of orders show a delivery date earlier than the purchase date "
                f"(handled as data-entry errors in preprocessing, but a high rate here would be a red flag).",
            ))

    # --- can the target label even be built? ---
    if all(c in orders.columns for c in ["order_status", "order_delivered_customer_date", "order_estimated_delivery_date"]):
        buildable = (
            (orders["order_status"] == "delivered")
            & orders["order_delivered_customer_date"].notna()
            & orders["order_estimated_delivery_date"].notna()
        ).sum()
        checks.append(_check(
            "label:buildable", "FAIL" if buildable < 500 else "OK",
            f"{buildable:,} orders have everything needed to build the is_delayed label "
            f"(need >= 500 for a usable train/test split).",
        ))

    # --- value sanity on items (use the type-coerced series, not the raw column,
    #     so a text-corrupted price/freight column can't crash this with a raw
    #     pandas TypeError instead of a clean check result) ---
    items_numeric = coerced_numeric.get("olist_order_items_dataset.csv", {})
    price = items_numeric.get("price")
    freight = items_numeric.get("freight_value")
    if price is not None and freight is not None:
        valid = price.notna() & freight.notna()
        neg_rate = ((price[valid] < 0) | (freight[valid] < 0)).mean() if valid.any() else 0.0
        checks.append(_check(
            "values:items_negative", "WARN" if neg_rate > 0 else "OK",
            f"{neg_rate:.2%} of item rows have a negative price or freight value." if neg_rate > 0
            else "No negative price/freight values.",
        ))

    # --- geolocation coordinate sanity (rough Brazil bounding box) ---
    geo_numeric = coerced_numeric.get("olist_geolocation_dataset.csv", {})
    geo_lat, geo_lng = geo_numeric.get("geolocation_lat"), geo_numeric.get("geolocation_lng")
    if geo_lat is not None and geo_lng is not None:
        valid = geo_lat.notna() & geo_lng.notna()
        if valid.any():
            in_bounds = geo_lat[valid].between(-34, 6) & geo_lng[valid].between(-74, -32)
            out_rate = 1 - in_bounds.mean()
            checks.append(_check(
                "values:geo_bounds", "WARN" if out_rate > 0.05 else "OK",
                f"{out_rate:.1%} of geolocation rows fall outside Brazil's bounding box "
                f"(dropped automatically before computing distances).",
            ))

    return _finalize(checks)


def _finalize(checks):
    fails = [c for c in checks if c["status"] == "FAIL"]
    warns = [c for c in checks if c["status"] == "WARN"]
    verdict = "REJECT" if fails else ("ACCEPT_WITH_WARNINGS" if warns else "ACCEPT")
    return {"verdict": verdict, "n_checks": len(checks), "n_fail": len(fails), "n_warn": len(warns), "checks": checks}


def format_report(report):
    lines = [f"Verdict: {report['verdict']}  "
             f"({report['n_checks']} checks, {report['n_fail']} failed, {report['n_warn']} warnings)", ""]
    icon = {"OK": "  ok  ", "WARN": " warn ", "FAIL": " FAIL "}
    for c in report["checks"]:
        if c["status"] == "OK":
            continue  # keep the printed report focused on what needs attention
        lines.append(f"[{icon[c['status']]}] {c['check']}: {c['message']}")
    if report["n_fail"] == 0 and report["n_warn"] == 0:
        lines.append("All checks passed cleanly.")
    return "\n".join(lines)


if __name__ == "__main__":
    report = validate_raw_dataset("../data/raw")
    print(format_report(report))
    if report["verdict"] == "REJECT":
        raise SystemExit(1)
