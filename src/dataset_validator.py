"""Dataset validator for OnTime v2 adaptive upload.

The validator checks that a zip archive contains all required Olist CSV files and that each CSV
contains the expected columns (a lightweight sanity check).  It returns a JSON‑serialisable
dictionary with a boolean ``valid`` flag and a list of ``issues`` describing any problems.
"""

import os
import zipfile
import pandas as pd

REQUIRED_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

# Expected columns for a few key files – this is only a minimal example.
EXPECTED_COLUMNS = {
    "olist_orders_dataset.csv": ["order_id", "order_status", "order_purchase_timestamp", "seller_id", "customer_id"],
    "olist_customers_dataset.csv": ["customer_id", "customer_unique_id", "customer_zip_code_prefix"],
    "olist_sellers_dataset.csv": ["seller_id", "seller_zip_code_prefix"],
    "olist_geolocation_dataset.csv": ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
}


def validate_zip(zip_path: str) -> dict:
    """Validate the contents of a zip file.

    Parameters
    ----------
    zip_path: str
        Path to the uploaded zip archive.

    Returns
    -------
    dict
        ``{"valid": bool, "issues": list}``
    """
    issues = []
    if not zipfile.is_zipfile(zip_path):
        return {"valid": False, "issues": ["File is not a valid zip archive."]}

    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = [os.path.basename(name) for name in z.namelist() if not name.endswith('/')]
        # Check required files
        for req in REQUIRED_FILES:
            if req not in namelist:
                issues.append(f"Missing required file: {req}")

        # Light column checks for a subset of files
        for file, cols in EXPECTED_COLUMNS.items():
            if file in namelist:
                try:
                    with z.open(file) as f:
                        df = pd.read_csv(f, nrows=5)
                    missing = [c for c in cols if c not in df.columns]
                    if missing:
                        issues.append(f"{file} missing columns: {', '.join(missing)}")
                except Exception as e:
                    issues.append(f"Error reading {file}: {e}")

    return {"valid": len(issues) == 0, "issues": issues}
