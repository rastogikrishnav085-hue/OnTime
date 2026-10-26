"""Feature engineering utilities for OnTime v2.

This module adds new features to the Olist order dataset that improve model
performance while keeping the pipeline lightweight.

Features added:
- `seller_avg_delay` – average delivery delay per seller (in days).
- `seller_order_count` – total number of orders per seller.
- `order_dayofweek` – day of week of the order (0=Mon).
- `order_month` – month of the order (1‑12).
- `geo_distance_km` – haversine distance between seller and buyer using the
  geolocation CSVs.

All functions accept a pandas ``DataFrame`` and return the enriched ``DataFrame``.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Haversine distance helper
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def add_seller_aggregates(orders: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    """Add seller level aggregate columns.

    Parameters
    ----------
    orders: DataFrame with ``order_id`` and ``seller_id``.
    sellers: DataFrame with ``seller_id`` and ``delivery_delay`` (in days).
    """
    # Compute average delay per seller
    seller_delay = sellers.groupby('seller_id')['delivery_delay'].mean().rename('seller_avg_delay')
    seller_count = sellers.groupby('seller_id').size().rename('seller_order_count')
    agg = pd.concat([seller_delay, seller_count], axis=1)
    orders = orders.join(agg, on='seller_id')
    # Fill missing values with global averages
    orders['seller_avg_delay'].fillna(orders['seller_avg_delay'].mean(), inplace=True)
    orders['seller_order_count'].fillna(0, inplace=True)
    return orders


def add_temporal_features(orders: pd.DataFrame, date_col: str = 'order_purchase_timestamp') -> pd.DataFrame:
    """Extract day‑of‑week and month from the order purchase timestamp.
    """
    ts = pd.to_datetime(orders[date_col])
    orders['order_dayofweek'] = ts.dt.dayofweek
    orders['order_month'] = ts.dt.month
    return orders


def add_geo_distance(orders: pd.DataFrame,
                      customers_geo: pd.DataFrame,
                      sellers_geo: pd.DataFrame) -> pd.DataFrame:
    """Compute haversine distance between buyer and seller locations.

    The geography dataframes must contain ``customer_id``/``seller_id`` and
    ``geolocation_lat`` / ``geolocation_lng`` columns.
    """
    # Merge lat/lon for customers and sellers
    orders = orders.merge(customers_geo[['customer_id', 'geolocation_lat', 'geolocation_lng']]
                          .rename(columns={'geolocation_lat': 'cust_lat', 'geolocation_lng': 'cust_lng'}),
                          on='customer_id', how='left')
    orders = orders.merge(sellers_geo[['seller_id', 'geolocation_lat', 'geolocation_lng']]
                          .rename(columns={'geolocation_lat': 'sell_lat', 'geolocation_lng': 'sell_lng'}),
                          on='seller_id', how='left')
    # Compute distance row‑wise
    distances = []
    for _, row in orders.iterrows():
        if pd.isna(row['cust_lat']) or pd.isna(row['sell_lat']):
            distances.append(np.nan)
        else:
            distances.append(_haversine(row['cust_lat'], row['cust_lng'], row['sell_lat'], row['sell_lng']))
    orders['geo_distance_km'] = distances
    # Fill missing with median distance
    median_dist = orders['geo_distance_km'].median()
    orders['geo_distance_km'].fillna(median_dist, inplace=True)
    # Drop intermediate columns
    orders.drop(columns=['cust_lat', 'cust_lng', 'sell_lat', 'sell_lng'], inplace=True)
    return orders


def engineer_features(orders_path: str,
                       customers_path: str,
                       sellers_path: str,
                       customers_geo_path: str,
                       sellers_geo_path: str) -> pd.DataFrame:
    """High‑level helper that loads raw CSVs, adds all engineered features,
    and returns the final training matrix.
    """
    orders = pd.read_csv(orders_path)
    customers = pd.read_csv(customers_path)
    sellers = pd.read_csv(sellers_path)
    customers_geo = pd.read_csv(customers_geo_path)
    sellers_geo = pd.read_csv(sellers_geo_path)

    orders = add_seller_aggregates(orders, sellers)
    orders = add_temporal_features(orders)
    orders = add_geo_distance(orders, customers_geo, sellers_geo)
    return orders
