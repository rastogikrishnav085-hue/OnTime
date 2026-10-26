"""
features.py
------------
Turns the joined order-level table into a model-ready feature set.

The guiding rule throughout this file: a feature is only allowed in if it
would have been KNOWN at the moment the order was placed / approved and
before it shipped. Nothing derived from the delivery outcome itself
(delivered_customer_date, delay_days, review score, etc.) is used as an
input -- those columns exist purely to build the label or for later
evaluation.
"""

import numpy as np
import pandas as pd


def add_time_features(df):
    df = df.copy()
    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek  # 0=Mon
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["is_weekend_purchase"] = (df["purchase_dow"] >= 5).astype(int)

    df["approval_delay_hours"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600.0
    # a small number of orders have no recorded approval timestamp;
    # median-fill rather than drop the order
    df["approval_delay_hours"] = df["approval_delay_hours"].fillna(
        df["approval_delay_hours"].median()
    )
    df["approval_delay_hours"] = df["approval_delay_hours"].clip(lower=0)

    df["promised_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    return df


def add_historical_rates_by_purchase_order(df, key, new_col, target="is_delayed"):
    """
    FIRST-VERSION feature, kept only so the difference can be measured
    (see notebook 07). Averages the outcomes of every earlier-PURCHASED
    order in the group. Two flaws:
      1. orders bought a few days ago have not been delivered yet, so
         their outcome would not be known when a new order is placed;
      2. the fallback for a group with no history is the delay rate over
         ALL orders, including ones that happen later.
    Do not use for modelling.
    """
    df = df.sort_values("order_purchase_timestamp").copy()
    grp = df.groupby(key)[target]
    cum_sum = grp.cumsum() - df[target]
    cum_count = grp.cumcount()
    global_rate = df[target].mean()
    rate = cum_sum / cum_count.replace(0, np.nan)
    df[new_col] = rate.fillna(global_rate)
    return df


def add_historical_rates(df, key, new_col, target="is_delayed",
                         known_at="order_delivered_customer_date",
                         as_of="order_purchase_timestamp"):
    """
    Historical delay rate for a group (seller / category / state) using
    ONLY outcomes that were already observable when the order was placed.

    For an order placed at time t, the history is the set of orders in
    the same group whose delivery date is strictly before t -- because
    an order's is_delayed label does not exist until it is delivered.
    An order can never be part of its own history (it is delivered after
    it is placed), and no later order is ever used.

    When the group has no observable history yet, the fallback is the
    platform-wide rate computed the same way (outcomes known at time t,
    across all groups), and 0.0 only for the handful of orders placed
    before anything at all had been delivered. The fallback therefore
    also never looks at the future.
    """
    n = len(df)
    known = df[known_at].values.astype("datetime64[ns]")
    asof = df[as_of].values.astype("datetime64[ns]")
    y = df[target].values.astype(float)

    def rate_at(known_g, y_g, asof_g):
        order = np.argsort(known_g, kind="stable")
        cum = np.concatenate([[0.0], np.cumsum(y_g[order])])
        cnt = np.searchsorted(known_g[order], asof_g, side="left")  # delivered strictly before t
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(cnt > 0, cum[cnt] / np.maximum(cnt, 1), np.nan)
        return r

    global_rate_t = rate_at(known, y, asof)          # platform-wide, as known at each order's time
    global_rate_t = np.where(np.isnan(global_rate_t), 0.0, global_rate_t)

    out = np.full(n, np.nan)
    for _, idx in df.groupby(key).indices.items():
        out[idx] = rate_at(known[idx], y[idx], asof[idx])
    df = df.copy()
    df[new_col] = np.where(np.isnan(out), global_rate_t, out)
    return df


# columns build_feature_table computes itself (everything else in
# FEATURE_COLUMNS must already exist in the joined table it is given)
_DERIVED_HERE = {
    "purchase_dow", "purchase_month", "purchase_hour", "is_weekend_purchase",
    "approval_delay_hours", "promised_delivery_days",
    "seller_historical_delay_rate", "category_historical_delay_rate",
    "customer_state_historical_delay_rate",
}
def required_joined_columns():
    """Evaluated at call time because FEATURE_COLUMNS is defined below."""
    return sorted(
        (set(FEATURE_COLUMNS) - _DERIVED_HERE)
        | {"order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date",
           "order_delivered_customer_date", "is_delayed", "primary_seller_id"}
    )


def build_feature_table(df):
    missing = [c for c in required_joined_columns() if c not in df.columns]
    if missing:
        raise ValueError(
            f"build_feature_table: the joined table is missing {len(missing)} required "
            f"column(s): {missing}. Was it produced by preprocessing.build_master_table()?"
        )
    df = add_time_features(df)
    df = add_historical_rates(df, "primary_seller_id", "seller_historical_delay_rate")
    df = add_historical_rates(df, "product_category", "category_historical_delay_rate")
    df = add_historical_rates(df, "customer_state", "customer_state_historical_delay_rate")

    # a couple of numeric columns can have missing values (products with
    # no recorded weight/dimensions, orders with no matched geolocation)
    numeric_fill_cols = [
        "total_weight_g", "avg_weight_g", "total_volume_cm3",
        "seller_customer_distance_km", "total_payment_value",
        "n_payment_installments", "n_payment_methods",
        "avg_photos_qty", "avg_name_length", "avg_description_length",
    ]
    for c in numeric_fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    df["product_category"] = df["product_category"].fillna("unknown")
    df["payment_type"] = df["payment_type"].fillna("unknown")

    return df


# Columns that go INTO the model. Everything else (ids, raw dates,
# delivery outcome fields) is kept in the table for reference/reporting
# but excluded from X.
FEATURE_COLUMNS = [
    # order shape / value
    "n_items", "n_distinct_products", "n_distinct_sellers",
    "total_price", "avg_item_price", "max_item_price",
    "total_freight", "avg_freight", "total_order_value",
    # product
    "product_category", "total_weight_g", "avg_weight_g", "total_volume_cm3",
    "avg_photos_qty", "avg_name_length", "avg_description_length",
    # timing (known at purchase / approval time)
    "purchase_dow", "purchase_month", "purchase_hour", "is_weekend_purchase",
    "approval_delay_hours", "promised_delivery_days",
    # geography
    "customer_state", "seller_state", "same_state", "seller_customer_distance_km",
    # payment
    "payment_type", "n_payment_installments", "total_payment_value", "n_payment_methods",
    # historical risk signals (leak-free, time-ordered)
    "seller_historical_delay_rate", "category_historical_delay_rate",
    "customer_state_historical_delay_rate",
]

TARGET_COLUMN = "is_delayed"

LEAKAGE_COLUMNS_EXCLUDED = [
    "order_delivered_customer_date", "order_delivered_carrier_date",
    "delay_days", "order_status",
]

if __name__ == "__main__":
    raw = pd.read_csv("../data/processed/ontime_joined_raw.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ])
    feat = build_feature_table(raw)
    keep = ["order_id", TARGET_COLUMN, "order_purchase_timestamp"] + FEATURE_COLUMNS
    feat[keep].to_csv("../data/processed/ontime_model_ready.csv", index=False)
    print("Feature table shape:", feat[keep].shape)
    print(feat[FEATURE_COLUMNS].isna().sum()[lambda s: s > 0])
