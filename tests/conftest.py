import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _synthetic_olist(dirpath, n_orders=1500):
    """A small, valid, Olist-shaped dataset - enough rows to clear the
    validator's minimums, independent of the real (large) raw files."""
    i = np.arange(n_orders)
    purchase = pd.Timestamp("2017-01-01") + pd.to_timedelta(i * 2, unit="h")
    status = np.where(i % 20 == 0, "canceled", "delivered")
    delivered = purchase + pd.Timedelta(days=9)
    orders = pd.DataFrame({
        "order_id": [f"o{k}" for k in i], "customer_id": [f"c{k}" for k in i],
        "order_status": status,
        "order_purchase_timestamp": purchase,
        "order_approved_at": purchase + pd.Timedelta(hours=1),
        "order_delivered_carrier_date": purchase + pd.Timedelta(days=1),
        "order_delivered_customer_date": np.where(status == "delivered", delivered, pd.NaT),
        "order_estimated_delivery_date": purchase + pd.Timedelta(days=12),
    })
    items = pd.DataFrame({
        "order_id": [f"o{k % n_orders}" for k in range(n_orders + 100)],
        "order_item_id": 1,
        "product_id": [f"p{k % 150}" for k in range(n_orders + 100)],
        "seller_id": [f"s{k % 20}" for k in range(n_orders + 100)],
        "price": 10.0 + (np.arange(n_orders + 100) % 50), "freight_value": 5.0 + (np.arange(n_orders + 100) % 10),
    })
    customers = pd.DataFrame({"customer_id": [f"c{k}" for k in i], "customer_zip_code_prefix": 1000 + i % 50,
                              "customer_city": "x", "customer_state": "SP"})
    sellers = pd.DataFrame({"seller_id": [f"s{k}" for k in range(20)], "seller_zip_code_prefix": 1000 + np.arange(20),
                            "seller_city": "x", "seller_state": "SP"})
    products = pd.DataFrame({"product_id": [f"p{k}" for k in range(150)], "product_category_name": "cat_a",
                             "product_weight_g": 500.0, "product_length_cm": 20.0,
                             "product_height_cm": 10.0, "product_width_cm": 10.0})
    translation = pd.DataFrame({"product_category_name": ["cat_a"], "product_category_name_english": ["category_a"]})
    payments = pd.DataFrame({"order_id": [f"o{k}" for k in i], "payment_type": "credit_card",
                             "payment_installments": 1, "payment_value": 20.0})
    geo = pd.DataFrame({"geolocation_zip_code_prefix": 1000 + np.arange(300) % 60,
                        "geolocation_lat": -23.5 + (np.arange(300) % 10) * 0.01,
                        "geolocation_lng": -46.6 - (np.arange(300) % 10) * 0.01})
    files = {
        "olist_orders_dataset.csv": orders, "olist_order_items_dataset.csv": items,
        "olist_customers_dataset.csv": customers, "olist_sellers_dataset.csv": sellers,
        "olist_products_dataset.csv": products, "product_category_name_translation.csv": translation,
        "olist_order_payments_dataset.csv": payments, "olist_geolocation_dataset.csv": geo,
    }
    for name, frame in files.items():
        frame.to_csv(os.path.join(dirpath, name), index=False)


@pytest.fixture
def good_dir(tmp_path):
    _synthetic_olist(str(tmp_path))
    return str(tmp_path)


@pytest.fixture
def project_root():
    return ROOT
