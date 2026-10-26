"""
preprocessing.py
-----------------
Loads the raw Olist CSVs, cleans them and joins them into a single
order-level table. This file only builds the RAW joined table plus the
target variable. Feature engineering (distances, historical rates,
encodings) happens separately in features.py so the two concerns don't
get tangled together.
"""

import pandas as pd
import numpy as np

RAW_DIR = "../data/raw"


def load_raw_tables(raw_dir=RAW_DIR):
    orders = pd.read_csv(f"{raw_dir}/olist_orders_dataset.csv")
    items = pd.read_csv(f"{raw_dir}/olist_order_items_dataset.csv")
    customers = pd.read_csv(f"{raw_dir}/olist_customers_dataset.csv")
    sellers = pd.read_csv(f"{raw_dir}/olist_sellers_dataset.csv")
    products = pd.read_csv(f"{raw_dir}/olist_products_dataset.csv")
    translation = pd.read_csv(f"{raw_dir}/product_category_name_translation.csv")
    payments = pd.read_csv(f"{raw_dir}/olist_order_payments_dataset.csv")
    geo = pd.read_csv(f"{raw_dir}/olist_geolocation_dataset.csv")
    return orders, items, customers, sellers, products, translation, payments, geo


DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def clean_orders(orders):
    orders = orders.copy()
    for c in DATE_COLS:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")

    # We can only build a genuine "was it late" label for orders that were
    # actually delivered and where we know both dates being compared.
    # Cancelled / unavailable / still-in-transit orders don't give us a
    # clean ground truth, so they get dropped from the modelling set (they
    # are kept in the raw table for the EDA / operational-overview charts).
    usable = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
        & orders["order_purchase_timestamp"].notna()
    ].copy()

    # A handful of rows have order_delivered_customer_date earlier than
    # order_purchase_timestamp or other impossible sequences -- these are
    # data entry errors in the source dataset and are removed.
    bad_sequence = usable["order_delivered_customer_date"] < usable["order_purchase_timestamp"]
    usable = usable[~bad_sequence]

    usable["delay_days"] = (
        usable["order_delivered_customer_date"] - usable["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0
    usable["is_delayed"] = (usable["delay_days"] > 0).astype(int)

    return orders, usable


def aggregate_items(items, products, translation):
    items = items.copy()
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"], errors="coerce")

    products = products.merge(translation, on="product_category_name", how="left")
    products["product_category_name_english"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    ).fillna("unknown")

    items = items.merge(
        products[
            [
                "product_id",
                "product_category_name_english",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
                "product_name_lenght",
                "product_description_lenght",
                "product_photos_qty",
            ]
        ],
        on="product_id",
        how="left",
    )

    items["item_volume_cm3"] = (
        items["product_length_cm"] * items["product_height_cm"] * items["product_width_cm"]
    )

    # most-common category per order (a single order can technically mix
    # categories if it has several different products, so we take the mode)
    def mode_or_first(s):
        m = s.mode()
        return m.iloc[0] if len(m) else np.nan

    order_level = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_distinct_products=("product_id", "nunique"),
        n_distinct_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        avg_item_price=("price", "mean"),
        max_item_price=("price", "max"),
        total_freight=("freight_value", "sum"),
        avg_freight=("freight_value", "mean"),
        total_weight_g=("product_weight_g", "sum"),
        avg_weight_g=("product_weight_g", "mean"),
        total_volume_cm3=("item_volume_cm3", "sum"),
        avg_photos_qty=("product_photos_qty", "mean"),
        avg_name_length=("product_name_lenght", "mean"),
        avg_description_length=("product_description_lenght", "mean"),
    ).reset_index()

    category = items.groupby("order_id")["product_category_name_english"].agg(mode_or_first).reset_index()
    category.columns = ["order_id", "product_category"]

    # primary seller = seller responsible for the highest-value item in the order
    idx = items.groupby("order_id")["price"].idxmax()
    primary_seller = items.loc[idx, ["order_id", "seller_id"]].rename(columns={"seller_id": "primary_seller_id"})

    order_level = order_level.merge(category, on="order_id", how="left")
    order_level = order_level.merge(primary_seller, on="order_id", how="left")
    order_level["total_order_value"] = order_level["total_price"] + order_level["total_freight"]

    return order_level


def aggregate_payments(payments):
    def mode_or_first(s):
        m = s.mode()
        return m.iloc[0] if len(m) else np.nan

    agg = payments.groupby("order_id").agg(
        n_payment_installments=("payment_installments", "max"),
        total_payment_value=("payment_value", "sum"),
        n_payment_methods=("payment_type", "nunique"),
    ).reset_index()
    ptype = payments.groupby("order_id")["payment_type"].agg(mode_or_first).reset_index()
    ptype.columns = ["order_id", "payment_type"]
    agg = agg.merge(ptype, on="order_id", how="left")
    return agg


def build_geo_lookup(geo):
    # zip-prefix level average coordinates (a zip prefix can map to
    # several exact lat/lng pairs in the raw table, so we take the mean
    # and drop obvious outliers well outside Brazil's bounding box)
    g = geo[
        geo["geolocation_lat"].between(-34, 6) & geo["geolocation_lng"].between(-74, -32)
    ]
    lookup = g.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "mean"),
        lng=("geolocation_lng", "mean"),
    ).reset_index()
    lookup.columns = ["zip_code_prefix", "lat", "lng"]
    return lookup


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def build_master_table(raw_dir=RAW_DIR):
    orders, items, customers, sellers, products, translation, payments, geo = load_raw_tables(raw_dir)

    orders_all, orders_usable = clean_orders(orders)
    items_agg = aggregate_items(items, products, translation)
    payments_agg = aggregate_payments(payments)
    geo_lookup = build_geo_lookup(geo)

    df = orders_usable.merge(items_agg, on="order_id", how="inner")
    df = df.merge(customers, on="customer_id", how="left")
    df = df.merge(
        sellers.rename(columns={"seller_id": "primary_seller_id"}),
        on="primary_seller_id",
        how="left",
    )
    df = df.merge(payments_agg, on="order_id", how="left")

    # customer / seller coordinates -> distance feature
    df = df.merge(
        geo_lookup.rename(columns={"zip_code_prefix": "customer_zip_code_prefix", "lat": "cust_lat", "lng": "cust_lng"}),
        on="customer_zip_code_prefix",
        how="left",
    )
    df = df.merge(
        geo_lookup.rename(columns={"zip_code_prefix": "seller_zip_code_prefix", "lat": "sell_lat", "lng": "sell_lng"}),
        on="seller_zip_code_prefix",
        how="left",
    )
    df["seller_customer_distance_km"] = haversine_km(
        df["cust_lat"], df["cust_lng"], df["sell_lat"], df["sell_lng"]
    )

    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    return df, orders_all


if __name__ == "__main__":
    from data_validation import validate_raw_dataset, format_report

    gate = validate_raw_dataset(RAW_DIR)
    print(format_report(gate))
    if gate["verdict"] == "REJECT":
        raise SystemExit(
            "\nRaw dataset REJECTED -- preprocessing stopped before touching it. "
            "Fix the issues above (or point RAW_DIR at a valid Olist-shaped dataset) and re-run."
        )
    print()

    df, orders_all = build_master_table()
    print("Master table shape:", df.shape)
    print("Delay rate:", df["is_delayed"].mean().round(4))
    df.to_csv("../data/processed/ontime_joined_raw.csv", index=False)
