"""Adapting OnTime to other data: column matching, units, dates, provenance, outcomes, learning from own history."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import adapter as ad                                      # noqa: E402
from features import FEATURE_COLUMNS                      # noqa: E402
from scoring import required_input_columns                # noqa: E402

TYPICAL = {"n_items": 1.0, "n_distinct_products": 1.0, "n_distinct_sellers": 1.0, "total_price": 87.0, "avg_item_price": 79.0,
           "max_item_price": 80.0, "total_freight": 17.0, "avg_freight": 16.0, "total_order_value": 105.0, "total_weight_g": 750.0,
           "avg_weight_g": 700.0, "total_volume_cm3": 7224.0, "avg_photos_qty": 2.0, "avg_name_length": 52.0, "avg_description_length": 608.0,
           "purchase_dow": 3.0, "purchase_month": 6.0, "purchase_hour": 15.0, "is_weekend_purchase": 0.0, "approval_delay_hours": 0.34,
           "promised_delivery_days": 23.0, "same_state": 0.0, "seller_customer_distance_km": 434.0, "n_payment_installments": 2.0,
           "total_payment_value": 105.0, "n_payment_methods": 1.0, "product_category": "bed_bath_table", "customer_state": "SP",
           "seller_state": "SP", "payment_type": "credit_card"}


def foreign(n=3000, seed=0):
    rng = np.random.RandomState(seed)
    order = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.randint(0, 365 * 24 * 60, n), unit="min")
    miles = rng.uniform(3, 900, n)
    promised = rng.randint(3, 12, n)
    late_p = 1 / (1 + np.exp(-(miles / 150 - 2.2 + 1.2 * (order.month == 12))))
    actual = promised + (rng.rand(n) < late_p) * rng.randint(1, 5, n) + rng.randint(-2, 1, n)
    return pd.DataFrame({
        "Order ID": [f"A{i:05d}" for i in range(n)], "Order Date": order.strftime("%d/%m/%Y %H:%M"),
        "Promised Delivery": (order + pd.to_timedelta(promised, unit="D")).strftime("%d/%m/%Y"),
        "Actual Delivery": (order + pd.to_timedelta(actual, unit="D")).strftime("%d/%m/%Y"),
        "Destination State": rng.choice(["Texas", "Ohio", "Nevada"], n), "Warehouse State": rng.choice(["Texas", "Ohio"], n),
        "Qty": rng.randint(1, 4, n), "Item Price": rng.uniform(5, 200, n).round(2), "Shipping Cost": rng.uniform(2, 30, n).round(2),
        "Weight (kg)": rng.uniform(0.1, 12, n).round(2), "Category Name": rng.choice(["Home & Garden", "Electronics", "Toys"], n),
        "Payment Method": rng.choice(["Credit Card", "PayPal", "Debit card"], n), "Distance (miles)": miles.round(1),
        "Supplier": rng.choice(["S1", "S2", "S3", "S4"], n)})


def test_columns_are_matched_by_name_and_checked_against_values():
    df = foreign(200)
    m = ad.suggest_mapping(df)
    expect = {"order_id": "Order ID", "purchase_time": "Order Date", "promised_time": "Promised Delivery", "delivered_time": "Actual Delivery",
              "customer_state": "Destination State", "seller_state": "Warehouse State", "n_items": "Qty", "price": "Item Price",
              "freight": "Shipping Cost", "weight": "Weight (kg)", "category": "Category Name", "payment_type": "Payment Method",
              "distance": "Distance (miles)", "seller_id": "Supplier"}
    for k, col in expect.items():
        assert m[k]["source"] == col, (k, m[k])
    assert m["weight"]["unit"] == "kg" and m["distance"]["unit"] == "miles" and m["price"]["basis"] == "per_item"
    # one source column is never used for two concepts
    used = [v["source"] for v in m.values() if v["source"]]
    assert len(used) == len(set(used))
    # a name that looks like a date but holds text is not accepted as a date
    bad = pd.DataFrame({"order_date": ["hello"] * 50, "x": range(50)})
    assert ad.suggest_mapping(bad)["purchase_time"]["source"] is None


def test_dates_use_one_format_for_the_whole_column():
    s = pd.Series(["25/12/2024 10:00", "01/02/2024 08:30", "03/04/2024 12:00"])
    d = ad.to_datetime(s)
    assert list(d.dt.month) == [12, 2, 4] and list(d.dt.day) == [25, 1, 3]      # day-first, decided by the 25th
    us = ad.to_datetime(pd.Series(["12/25/2024", "02/01/2024", "04/03/2024"]))
    assert list(us.dt.month) == [12, 2, 4]                                       # the 25 forces month-first
    iso = ad.to_datetime(pd.Series(["2024-03-05T10:00:00Z", "2024-04-06T11:00:00Z"]))
    assert iso.dt.tz is None and iso.iloc[0].month == 3
    assert ad.to_datetime(pd.Series([1, 2, 3])).isna().all()


def test_apply_mapping_converts_units_and_records_where_each_input_came_from():
    df = foreign(300)
    m = ad.suggest_mapping(df)
    table, info = ad.apply_mapping(df, m, TYPICAL)
    assert set(required_input_columns()) <= set(table.columns) and {"order_id", "seller_id"} <= set(table.columns)
    r0 = df.iloc[0]
    assert table["total_weight_g"].iloc[0] == pytest.approx(r0["Weight (kg)"] * 1000)           # kg -> g
    assert table["seller_customer_distance_km"].iloc[0] == pytest.approx(r0["Distance (miles)"] * 1.609344)
    assert table["total_price"].iloc[0] == pytest.approx(r0["Item Price"] * r0["Qty"])          # per-item price x items
    assert table["customer_state"].iloc[0] == r0["Destination State"].upper() or len(table["customer_state"].iloc[0]) > 3
    assert (table["promised_delivery_days"] > 0).all() and (table["promised_delivery_days"] < 15).all()
    assert table["payment_type"].iloc[0] in ("credit_card", "debit_card", "paypal")
    p = info["provenance"]
    assert p["total_weight_g"] == "mapped" and p["purchase_dow"] == "derived" and p["same_state"] == "derived"
    assert p["avg_photos_qty"] == "default" and p["approval_delay_hours"] == "default" and p["max_item_price"] == "assumed"
    assert p["seller_historical_delay_rate"] == "lookup"
    assert 0 < info["importance_coverage"] < 1 and info["n_real"] < len(FEATURE_COLUMNS)
    assert not table[[c for c in required_input_columns() if c not in ("seller_id",)]].isna().any().any()


def test_coordinates_give_distance_and_blank_cells_are_filled_and_reported():
    df = pd.DataFrame({"order_date": pd.date_range("2024-01-01", periods=40).astype(str), "c_lat": [-23.55] * 40, "c_lng": [-46.63] * 40,
                       "s_lat": [-22.91] * 40, "s_lng": [-43.17] * 40, "weight_g": [500.0] * 39 + [np.nan]})
    m = {"purchase_time": {"source": "order_date"}, "customer_lat": {"source": "c_lat"}, "customer_lon": {"source": "c_lng"},
         "seller_lat": {"source": "s_lat"}, "seller_lon": {"source": "s_lng"}, "weight": {"source": "weight_g", "unit": "g"}}
    table, info = ad.apply_mapping(df, m, TYPICAL)
    assert table["seller_customer_distance_km"].iloc[0] == pytest.approx(357, abs=8)       # Sao Paulo - Rio de Janeiro
    assert info["provenance"]["seller_customer_distance_km"] == "derived"
    assert table["total_weight_g"].iloc[-1] == TYPICAL["total_weight_g"] and any("total_weight_g" in n for n in info["notes"])


def test_outcome_from_dates_status_text_and_inverted_flags():
    df = foreign(300)
    late, how = ad.detect_outcome(df, ad.suggest_mapping(df))
    assert "promised" in how and 0 < late.mean() < 1
    d = pd.DataFrame({"delivery_status": ["Late delivery"] * 40 + ["Shipping on time"] * 50 + ["Advance shipping"] * 20})
    late, _ = ad.detect_outcome(d, {"late_flag": {"source": "delivery_status"}})
    assert late.sum() == 40 and late.notna().sum() == 110
    f = pd.DataFrame({"on_time": [1] * 80 + [0] * 20})
    late, _ = ad.detect_outcome(f, {"late_flag": {"source": "on_time"}})
    assert late.sum() == 20                                                                 # on_time=0 means late
    none, why = ad.detect_outcome(pd.DataFrame({"a": range(10)}), {})
    assert none is None and "No outcome" in why


def test_learning_from_own_history_beats_chance_and_survives_save_load(tmp_path):
    df = foreign(4000)
    m = ad.suggest_mapping(df)
    table, info = ad.apply_mapping(df, m, TYPICAL)
    late, _ = ad.detect_outcome(df, m)
    pt, dt = ad.to_datetime(df["Order Date"]), ad.to_datetime(df["Actual Delivery"])
    pipe, spec, rep = ad.train_adaptive(table, info, late, pt, dt)
    assert rep["split"] == "time" and rep["roc_auc"] > 0.65 and rep["n_test"] == pytest.approx(0.2 * rep["n_train"] / 0.8, rel=0.2)
    assert set(rep["features_used"]) <= set(FEATURE_COLUMNS) and "avg_photos_qty" not in rep["features_used"]
    assert rep["own_history_rates"]                                                         # leak-free rates from the user's own history
    rows, y_te, p_te = rep["_test_rows"]
    assert len(rows) == len(y_te) == len(p_te) and rows.max() < len(table)
    ad.save_adapted(str(tmp_path / "m"), pipe, spec, m, "test")
    pipe2, meta = ad.load_adapted(str(tmp_path / "m"))
    assert np.allclose(ad.predict_adaptive(pipe, spec, table), ad.predict_adaptive(pipe2, meta["spec"], table))


def test_learning_refuses_when_there_is_too_little_to_learn_from():
    df = foreign(400)
    m = ad.suggest_mapping(df)
    table, info = ad.apply_mapping(df, m, TYPICAL)
    late = pd.Series(np.nan, index=table.index)
    late.iloc[:100] = 0.0
    late.iloc[:10] = 1.0
    with pytest.raises(ValueError, match="Not enough labelled"):
        ad.train_adaptive(table, info, late, ad.to_datetime(df["Order Date"]), ad.to_datetime(df["Actual Delivery"]))


def test_historical_rates_never_use_the_future():
    df = foreign(2500, seed=3)
    m = ad.suggest_mapping(df)
    table, info = ad.apply_mapping(df, m, TYPICAL)
    late, _ = ad.detect_outcome(df, m)
    pt, dt = ad.to_datetime(df["Order Date"]), ad.to_datetime(df["Actual Delivery"])
    t, added = ad.add_own_history_rates(table, info["provenance"], pt, dt, late)
    assert "seller_historical_delay_rate" in added
    # for one order, recompute by hand: mean outcome of same-seller orders DELIVERED strictly before it was placed
    row = t.iloc[1500]
    prior = t[(t["seller_id"] == row["seller_id"]) & (t["_delivered"] < row["_purchase"])]
    if len(prior):
        assert row["seller_historical_delay_rate"] == pytest.approx(prior[ad.TARGET_COLUMN].mean())
