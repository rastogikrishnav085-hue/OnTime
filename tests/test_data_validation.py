"""The raw-dataset gate: it must accept a valid dataset and REJECT (not
crash on) every kind of broken one."""
import os

import pandas as pd

from data_validation import validate_raw_dataset


def _verdict(d):
    return validate_raw_dataset(d)["verdict"]


def _edit(d, filename, fn):
    path = os.path.join(d, filename)
    df = pd.read_csv(path)
    fn(df).to_csv(path, index=False)


def test_valid_dataset_is_accepted(good_dir):
    r = validate_raw_dataset(good_dir)
    assert r["verdict"] == "ACCEPT", r["checks"]


def test_nonexistent_directory_is_rejected_not_crashed(tmp_path):
    assert _verdict(str(tmp_path / "does_not_exist")) == "REJECT"


def test_empty_directory_is_rejected(tmp_path):
    assert _verdict(str(tmp_path)) == "REJECT"


def test_missing_file_is_rejected(good_dir):
    os.remove(os.path.join(good_dir, "olist_sellers_dataset.csv"))
    assert _verdict(good_dir) == "REJECT"


def test_zero_byte_file_is_rejected(good_dir):
    open(os.path.join(good_dir, "olist_customers_dataset.csv"), "w").close()
    assert _verdict(good_dir) == "REJECT"


def test_missing_required_column_is_rejected(good_dir):
    _edit(good_dir, "olist_orders_dataset.csv", lambda d: d.drop(columns=["order_estimated_delivery_date"]))
    r = validate_raw_dataset(good_dir)
    assert r["verdict"] == "REJECT"
    assert any("order_estimated_delivery_date" in c["message"] for c in r["checks"])


def test_duplicate_order_ids_are_rejected(good_dir):
    _edit(good_dir, "olist_orders_dataset.csv", lambda d: pd.concat([d, d.head(30)]))
    assert _verdict(good_dir) == "REJECT"


def test_orphaned_items_are_rejected(good_dir):
    def orphan(d):
        d["order_id"] = "ghost_" + d.index.astype(str)
        return d
    _edit(good_dir, "olist_order_items_dataset.csv", orphan)
    assert _verdict(good_dir) == "REJECT"


def test_too_few_rows_is_rejected(good_dir):
    _edit(good_dir, "olist_order_items_dataset.csv", lambda d: d.head(50))
    assert _verdict(good_dir) == "REJECT"


def test_no_deliverable_orders_is_rejected(good_dir):
    def cancel_all(d):
        d["order_status"] = "canceled"
        return d
    _edit(good_dir, "olist_orders_dataset.csv", cancel_all)
    r = validate_raw_dataset(good_dir)
    assert r["verdict"] == "REJECT"
    assert any(c["check"] == "label:buildable" and c["status"] == "FAIL" for c in r["checks"])


def test_text_in_numeric_column_does_not_crash_and_is_flagged(good_dir):
    """Regression: a text value in `price` used to raise a raw pandas TypeError."""
    def poison(d):
        d["price"] = d["price"].astype(object)
        d.loc[d.index[:60], "price"] = "not_a_number"   # ~3.8% of rows
        return d
    _edit(good_dir, "olist_order_items_dataset.csv", poison)
    r = validate_raw_dataset(good_dir)          # must not raise
    assert r["verdict"] == "ACCEPT_WITH_WARNINGS"
    assert any(c["check"].startswith("types:") for c in r["checks"])


def test_mostly_text_numeric_column_is_rejected(good_dir):
    def poison(d):
        d["price"] = d["price"].astype(object)
        d.loc[d.index[:800], "price"] = "oops"      # ~50%
        return d
    _edit(good_dir, "olist_order_items_dataset.csv", poison)
    assert _verdict(good_dir) == "REJECT"


def test_negative_prices_are_warned(good_dir):
    _edit(good_dir, "olist_order_items_dataset.csv", lambda d: d.assign(price=-d["price"]))
    r = validate_raw_dataset(good_dir)
    assert any(c["check"] == "values:items_negative" and c["status"] == "WARN" for c in r["checks"])
