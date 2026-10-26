"""
adapter.py
-----------
Makes OnTime work on data that is NOT the Olist export.

A logistics team's own file will have different column names, units and
shapes ("Order Date", "Ship Cost", weight in kilograms, a delivery status
text instead of dates). This module:

  1. suggest_mapping()   guesses which of the file's columns is which
                         OnTime concept (order time, promised date, weight,
                         state ...), from the name AND by checking the
                         values (a "date" column must parse as dates);
  2. apply_mapping()     turns the file into the exact table the model
                         expects, deriving what can be derived (day of week,
                         promised days, distance from coordinates), and
                         recording for EVERY model input whether it was
                         taken from the file, derived, assumed or filled
                         with a typical value - nothing is silent;
  3. detect_outcome()    finds out whether the file says which orders were
                         really late (a flag, a status, or delivered vs
                         promised dates), so the model can be checked on it;
  4. train_adaptive()    learns a model from the user's own history using
                         only the inputs the file really supplies, with
                         leak-free past-delay rates computed from that
                         history (same rule as the Olist model);
  5. save/load_adapted() keep that model so more files can be scored.

The Olist model is the starting point; adaptive training is what you do
when its transfer accuracy on your data is not good enough.
"""
import json
import os
import re
import unicodedata
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, TARGET_COLUMN, add_historical_rates
from modeling import CATEGORICAL, NUMERIC

# --------------------------------------------------------------------------- concepts
# kind: id | text | datetime | number | flag
CONCEPTS = [
    dict(key="order_id", label="Order ID", kind="id", required=False,
         names=["order_id", "orderid", "order_no", "order_number", "order_ref", "shipment_id", "tracking_id", "awb", "id"]),
    dict(key="seller_id", label="Seller / vendor ID", kind="id", required=False,
         names=["seller_id", "vendor_id", "merchant_id", "supplier_id", "shop_id", "store_id", "seller", "vendor", "merchant", "supplier"]),
    dict(key="purchase_time", label="Order date / time", kind="datetime", required=True,
         names=["order_purchase_timestamp", "purchase_timestamp", "order_date", "order_time", "order_datetime", "ordered_at",
                "created_at", "purchase_date", "order_placed", "date_ordered", "order_created", "placed_at", "order_date_dateorders"]),
    dict(key="approved_time", label="Payment / approval time", kind="datetime", required=False,
         names=["order_approved_at", "approved_at", "approval_date", "payment_time", "payment_date", "confirmed_at", "paid_at"]),
    dict(key="promised_time", label="Promised delivery date", kind="datetime", required=False,
         names=["order_estimated_delivery_date", "estimated_delivery_date", "promised_date", "promised_delivery", "expected_delivery",
                "expected_delivery_date", "eta", "estimated_arrival", "scheduled_delivery", "due_date", "sla_date", "delivery_deadline",
                "estimated_delivery", "promise_date", "promised", "promise", "promised_by", "promised_on", "expected", "estimated", "target_date", "committed_date", "commit_date"]),
    dict(key="promised_days", label="Promised delivery window (days)", kind="number", required=False,
         names=["promised_days", "promised_delivery_days", "sla_days", "lead_time", "delivery_window", "estimated_days",
                "days_for_shipment_scheduled", "scheduled_days", "promised_lead_time", "delivery_days_promised"]),
    dict(key="delivered_time", label="Actual delivery date", kind="datetime", required=False,
         names=["order_delivered_customer_date", "delivered_at", "delivery_date", "actual_delivery", "actual_delivery_date",
                "delivered_date", "received_at", "arrival_date", "date_delivered", "shipping_date_dateorders", "delivered_on", "delivered", "delivery_completed"]),
    dict(key="late_flag", label="Was it late? (flag or status)", kind="flag", required=False,
         names=["is_delayed", "is_late", "late", "delayed", "late_delivery_risk", "delivery_status", "late_flag", "on_time", "ontime",
                "is_on_time", "delay_flag", "delivery_delay", "status"]),
    dict(key="customer_state", label="Delivery state / region", kind="text", required=False,
         names=["customer_state", "customer_region", "destination_state", "ship_state", "delivery_state", "customer_province",
                "order_state", "order_region", "dest_state", "ship_to", "deliver_to", "destination", "state", "region", "province"]),
    dict(key="seller_state", label="Origin state / region", kind="text", required=False,
         names=["seller_state", "origin_state", "vendor_state", "warehouse_state", "ship_from_state", "origin_region",
                "seller_region", "dispatch_state", "source_state", "ship_from", "origin", "dispatch_from"]),
    dict(key="customer_lat", label="Delivery latitude", kind="number", required=False,
         names=["customer_lat", "customer_latitude", "dest_lat", "destination_lat", "delivery_lat", "latitude", "lat"]),
    dict(key="customer_lon", label="Delivery longitude", kind="number", required=False,
         names=["customer_lng", "customer_lon", "customer_long", "customer_longitude", "dest_lng", "dest_lon", "destination_lng",
                "delivery_lng", "longitude", "lng", "lon", "long"]),
    dict(key="seller_lat", label="Origin latitude", kind="number", required=False,
         names=["seller_lat", "seller_latitude", "origin_lat", "warehouse_lat", "store_lat", "origin_latitude", "source_lat"]),
    dict(key="seller_lon", label="Origin longitude", kind="number", required=False,
         names=["seller_lng", "seller_lon", "seller_longitude", "origin_lng", "origin_lon", "origin_longitude", "warehouse_lng",
                "warehouse_lon", "store_lng", "source_lng"]),
    dict(key="distance", label="Shipping distance", kind="number", required=False, unit_options=["km", "miles", "m"],
         names=["distance", "distance_km", "shipping_distance", "haversine_distance", "route_distance", "distance_miles", "delivery_distance"]),
    dict(key="category", label="Product category", kind="text", required=False,
         names=["product_category", "category", "product_category_name_english", "product_category_name", "department", "product_type",
                "item_category", "category_name", "department_name"]),
    dict(key="n_items", label="Items in the order", kind="number", required=False,
         names=["n_items", "quantity", "qty", "item_count", "items", "order_item_quantity", "units", "number_of_items", "order_quantity"]),
    dict(key="price", label="Order / item price", kind="number", required=False, basis_options=["total", "per_item"],
         names=["total_price", "price", "order_value", "item_price", "product_price", "unit_price", "amount", "sales", "order_total",
                "subtotal", "sale_price", "order_item_product_price"]),
    dict(key="freight", label="Shipping cost", kind="number", required=False,
         names=["freight", "freight_value", "total_freight", "shipping_cost", "shipping_fee", "delivery_fee", "delivery_charge",
                "shipping_charge", "carrier_cost", "ship_cost", "postage"]),
    dict(key="weight", label="Weight", kind="number", required=False, unit_options=["g", "kg", "lb"], basis_options=["total", "per_item"],
         names=["weight", "weight_g", "product_weight_g", "weight_kg", "item_weight", "package_weight", "gross_weight", "total_weight_g"]),
    dict(key="volume", label="Package volume (cm³)", kind="number", required=False,
         names=["volume", "volume_cm3", "total_volume_cm3", "package_volume", "cubic_cm"]),
    dict(key="length", label="Length (cm)", kind="number", required=False, names=["length", "length_cm", "product_length_cm", "pkg_length"]),
    dict(key="height", label="Height (cm)", kind="number", required=False, names=["height", "height_cm", "product_height_cm", "pkg_height"]),
    dict(key="width", label="Width (cm)", kind="number", required=False, names=["width", "width_cm", "product_width_cm", "pkg_width"]),
    dict(key="payment_type", label="Payment method", kind="text", required=False,
         names=["payment_type", "payment_method", "pay_type", "pay_method", "payment_mode", "type", "payment"]),
    dict(key="installments", label="Payment instalments", kind="number", required=False,
         names=["installments", "payment_installments", "n_installments", "emi_months", "instalments", "number_of_installments"]),
    dict(key="payment_value", label="Amount paid", kind="number", required=False,
         names=["payment_value", "amount_paid", "total_paid", "payment_amount", "paid_amount"]),
]
CONCEPT_BY_KEY = {c["key"]: c for c in CONCEPTS}

STATE_CODES = {
    "acre": "AC", "alagoas": "AL", "amapa": "AP", "amazonas": "AM", "bahia": "BA", "ceara": "CE", "distrito federal": "DF",
    "espirito santo": "ES", "goias": "GO", "maranhao": "MA", "mato grosso do sul": "MS", "mato grosso": "MT",
    "minas gerais": "MG", "para": "PA", "paraiba": "PB", "parana": "PR", "pernambuco": "PE", "piaui": "PI",
    "rio de janeiro": "RJ", "rio grande do norte": "RN", "rio grande do sul": "RS", "rondonia": "RO", "roraima": "RR",
    "santa catarina": "SC", "sao paulo": "SP", "sergipe": "SE", "tocantins": "TO",
}
PAYMENT_WORDS = [("credit", "credit_card"), ("visa", "credit_card"), ("master", "credit_card"), ("cc", "credit_card"),
                 ("debit", "debit_card"), ("boleto", "boleto"), ("bank slip", "boleto"), ("voucher", "voucher"),
                 ("gift", "voucher"), ("coupon", "voucher")]

# Inputs the file can never supply in practice (Olist product-page statistics).
NEVER_FROM_FILE = ["avg_photos_qty", "avg_name_length", "avg_description_length"]
RATE_FEATURES = ["seller_historical_delay_rate", "category_historical_delay_rate", "customer_state_historical_delay_rate"]


# --------------------------------------------------------------------------- helpers
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _strip_accents(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y",
    "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y %H:%M", "%d.%m.%Y",
    "%d %b %Y", "%d-%b-%Y", "%d-%b-%y", "%b %d, %Y", "%d %B %Y",
]


def to_datetime(series):
    """Parse a column as dates, timezone-naive.

    Day/month order is the classic trap ("03/04/2024"), and pandas' flexible
    parser decides it row by row, which silently scrambles a column. So this
    picks ONE format for the whole column: ISO first, otherwise the format
    from the list above that reads the most values (a column with any day
    above 12 settles day-first versus month-first; on a true tie it goes
    day-first, as in Brazil, India and Europe)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    s = series.where(series.notna(), None)
    s = s.astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)
    nn = s.dropna()
    if len(nn) == 0:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    sample = nn.iloc[:: max(1, len(nn) // 2000)]

    def finish(parsed):
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert(None)
        return parsed

    try:
        iso = pd.to_datetime(sample, errors="coerce", format="ISO8601", utc=True)
        if iso.notna().mean() >= 0.95:
            return finish(pd.to_datetime(s, errors="coerce", format="ISO8601", utc=True))
    except (ValueError, TypeError):
        pass
    best_fmt, best_n = None, 0
    for fmt in _DATE_FORMATS:
        n_ok = int(pd.to_datetime(sample, errors="coerce", format=fmt).notna().sum())
        if n_ok > best_n:
            best_fmt, best_n = fmt, n_ok
    if best_fmt is not None and best_n / len(sample) >= 0.8:
        return pd.to_datetime(s, errors="coerce", format=best_fmt)
    try:  # last resort: let pandas infer one format from the data
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return finish(pd.to_datetime(s, errors="coerce"))
    except (ValueError, TypeError):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")


def _parse_rate(series, kind):
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    if kind == "datetime":
        return float(to_datetime(series).notna().sum() / len(s))
    if kind == "number":
        return float(pd.to_numeric(s, errors="coerce").notna().mean())
    return 1.0


def _name_score(col_norm, names):
    """1.0 exact; 0.88 a known name appears as whole words; 0.7 similar spelling."""
    import difflib
    best = 0.0
    for n in names:
        if col_norm == n:
            return 1.0
        if re.search(rf"(^|_){re.escape(n)}($|_)", col_norm) and len(n) >= 4:
            best = max(best, 0.88)
        elif len(n) >= 5 and difflib.SequenceMatcher(None, col_norm, n).ratio() >= 0.84:
            best = max(best, 0.7)
    return best


def _unit_hint(col_norm, concept):
    if concept == "weight":
        if re.search(r"(^|_)(kg|kgs|kilo|kilograms?)($|_)", col_norm):
            return "kg"
        if re.search(r"(^|_)(lb|lbs|pounds?)($|_)", col_norm):
            return "lb"
        return "g"
    if concept == "distance":
        if re.search(r"(^|_)(mi|miles?)($|_)", col_norm):
            return "miles"
        if col_norm.endswith("_m") or "meters" in col_norm:
            return "m"
        return "km"
    return None


def _basis_hint(col_norm):
    return "per_item" if re.search(r"(^|_)(unit|per_item|each|item_price|per_unit)($|_)", col_norm) else "total"


def suggest_mapping(df):
    """Guess which source column plays each concept. Returns
    {concept: {source, confidence, how, unit?, basis?}} for every concept
    (source None when nothing fits)."""
    cols = {c: _norm(c) for c in df.columns}
    cands = []
    for concept in CONCEPTS:
        for c, cn in cols.items():
            sc = _name_score(cn, concept["names"])
            if sc == 0:
                continue
            rate = _parse_rate(df[c], concept["kind"])
            if concept["kind"] in ("datetime", "number") and rate < 0.8:
                continue
            if concept["key"] == "late_flag" and not _looks_like_flag(df[c]):
                continue
            # bare words like "state" or "type" are weak evidence
            weak = cn in ("id", "state", "region", "province", "type", "payment", "items", "status", "seller", "vendor", "lat", "lon",
                          "long", "lng", "latitude", "longitude")
            cands.append((sc * (0.85 if weak else 1.0) * (0.6 + 0.4 * rate), concept["key"], c))
    cands.sort(key=lambda t: -t[0])
    used_cols, got = set(), {}
    for score, key, col in cands:
        if key in got or col in used_cols:
            continue
        got[key] = (col, score)
        used_cols.add(col)
    out = {}
    for concept in CONCEPTS:
        k = concept["key"]
        if k in got:
            col, sc = got[k]
            how = "exact name" if sc >= 0.95 else "known name" if sc >= 0.75 else "similar name"
            entry = {"source": col, "confidence": round(min(sc, 1.0), 2), "how": how}
            u = _unit_hint(cols[col], k)
            if u:
                entry["unit"] = u
            if "basis_options" in concept:
                entry["basis"] = _basis_hint(cols[col])
            out[k] = entry
        else:
            out[k] = {"source": None, "confidence": 0.0, "how": "not found"}
    return out


def _looks_like_flag(s):
    v = s.dropna().astype(str).str.strip().str.lower()
    if len(v) == 0:
        return False
    u = v.unique()
    if len(u) > 12:
        return False
    return True


# --------------------------------------------------------------------------- outcome
_LATE_WORDS = ("late", "delay", "overdue", "breach", "missed")
_OK_WORDS = ("on time", "ontime", "on_time", "early", "advance", "shipping on time", "not late", "not delayed")


def _flag_to_late(series, col_name):
    """Turn a flag/status column into 1 = late, 0 = on time, NaN = unknown."""
    cn = _norm(col_name)
    inverse = bool(re.search(r"(^|_)(on_time|ontime|is_on_time)($|_)", cn))
    out = pd.Series(np.nan, index=series.index)
    s = series.astype(str).str.strip().str.lower()
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.9 and set(num.dropna().unique()) <= {0, 1, 0.0, 1.0}:
        out = num.astype(float)
        return (1 - out) if inverse else out
    truthy = {"true", "yes", "y", "1", "t"}
    falsy = {"false", "no", "n", "0", "f"}
    is_t, is_f = s.isin(truthy), s.isin(falsy)
    if (is_t | is_f).mean() > 0.9:
        out = pd.Series(np.where(is_t, 1.0, np.where(is_f, 0.0, np.nan)), index=series.index)
        return (1 - out) if inverse else out
    late = s.apply(lambda x: any(w in x for w in _LATE_WORDS) and not any(w in x for w in _OK_WORDS))
    ok = s.apply(lambda x: any(w in x for w in _OK_WORDS))
    delivered_words = s.apply(lambda x: "deliver" in x or "advance" in x or "on" in x)
    out = pd.Series(np.where(late, 1.0, np.where(ok | delivered_words, 0.0, np.nan)), index=series.index)
    return out


def detect_outcome(df, mapping):
    """Returns (late Series of 1/0/NaN, how-text) or (None, reason)."""
    m = lambda k: (mapping.get(k) or {}).get("source")          # noqa: E731
    if m("delivered_time") and (m("promised_time")):
        d, p = to_datetime(df[m("delivered_time")]), to_datetime(df[m("promised_time")])
        late = pd.Series(np.where(d.notna() & p.notna(), (d.dt.normalize() > p.dt.normalize()).astype(float), np.nan), index=df.index)
        if late.notna().sum() >= 50:
            return late, "delivered date after the promised date"
    if m("late_flag"):
        late = _flag_to_late(df[m("late_flag")], m("late_flag"))
        if late.notna().sum() >= 50:
            return late, f"the '{m('late_flag')}' column"
    return None, ("No outcome found. Map the actual delivery date together with the promised date, or a column saying whether "
                  "the order was late, if you want OnTime to be checked or retrained on your data.")


# --------------------------------------------------------------------------- applying a mapping
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _clean_state(v):
    if pd.isna(v) or str(v).strip() == "":
        return np.nan
    s = str(v).strip()
    key = _strip_accents(s).lower()
    if key in STATE_CODES:
        return STATE_CODES[key]
    return s.upper() if len(s) <= 3 else s


def _clean_category(v):
    if pd.isna(v) or str(v).strip() == "":
        return "unknown"
    return re.sub(r"[^a-z0-9]+", "_", _strip_accents(v).lower()).strip("_") or "unknown"


def _clean_payment(v):
    if pd.isna(v) or str(v).strip() == "":
        return "unknown"
    s = _strip_accents(v).lower().strip()
    for word, canon in PAYMENT_WORDS:
        if word in s:
            return canon
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_") or "unknown"


def apply_mapping(df, mapping, typical, weights=None):
    """Build the model's input table from `df` using `mapping`.

    Returns (table, info). `table` has order_id, seller_id and every column
    in scoring.required_input_columns(). `info` holds `provenance`
    (feature -> mapped | derived | assumed | default), `coverage` and notes.
    """
    n = len(df)
    ix = df.index
    src = lambda k: (mapping.get(k) or {}).get("source")                     # noqa: E731
    opt = lambda k, o, d=None: (mapping.get(k) or {}).get(o, d)             # noqa: E731
    col = lambda k: df[src(k)] if src(k) and src(k) in df.columns else None  # noqa: E731
    num = lambda k: pd.to_numeric(col(k), errors="coerce") if col(k) is not None else None  # noqa: E731
    prov, notes, out = {}, [], {}

    def put(feature, values, how):
        out[feature] = values
        prov[feature] = how

    # --- time
    pt = to_datetime(col("purchase_time")) if col("purchase_time") is not None else None
    if pt is None:
        notes.append("No order date/time was mapped, so timing inputs (weekday, month, hour) use a typical order.")
    else:
        put("purchase_dow", pt.dt.dayofweek.astype(float), "derived")
        put("purchase_month", pt.dt.month.astype(float), "derived")
        put("purchase_hour", pt.dt.hour.astype(float), "derived")
        put("is_weekend_purchase", (pt.dt.dayofweek >= 5).astype(float), "derived")
    if pt is not None and col("approved_time") is not None:
        ap = to_datetime(col("approved_time"))
        put("approval_delay_hours", ((ap - pt).dt.total_seconds() / 3600.0).clip(lower=0), "derived")
    if pt is not None and col("promised_time") is not None:
        pr = to_datetime(col("promised_time"))
        put("promised_delivery_days", (pr - pt).dt.total_seconds() / 86400.0, "derived")
    elif num("promised_days") is not None:
        put("promised_delivery_days", num("promised_days"), "mapped")

    # --- basket and money
    n_items = num("n_items")
    if n_items is not None:
        n_items = n_items.clip(lower=1)
        put("n_items", n_items, "mapped")
    ni = n_items if n_items is not None else pd.Series(1.0, index=ix)
    if num("price") is not None:
        tp = num("price") * (ni if opt("price", "basis", "total") == "per_item" else 1.0)
        put("total_price", tp, "mapped")
        put("avg_item_price", tp / ni, "assumed")
        put("max_item_price", tp / ni, "assumed")
    if num("freight") is not None:
        put("total_freight", num("freight"), "mapped")
        put("avg_freight", num("freight") / ni, "derived")
    if "total_price" in out and "total_freight" in out:
        put("total_order_value", out["total_price"] + out["total_freight"], "derived")
    elif "total_price" in out:
        put("total_order_value", out["total_price"], "assumed")
    if num("payment_value") is not None:
        put("total_payment_value", num("payment_value"), "mapped")
    elif "total_order_value" in out:
        put("total_payment_value", out["total_order_value"], "assumed")
    if num("installments") is not None:
        put("n_payment_installments", num("installments").clip(lower=1), "mapped")
    put("n_payment_methods", pd.Series(1.0, index=ix), "assumed")
    put("n_distinct_sellers", pd.Series(1.0, index=ix), "assumed")
    put("n_distinct_products", ni.clip(lower=1), "assumed")

    # --- weight and volume
    if num("weight") is not None:
        f = {"g": 1.0, "kg": 1000.0, "lb": 453.592}.get(opt("weight", "unit", "g"), 1.0)
        tw = num("weight") * f * (ni if opt("weight", "basis", "total") == "per_item" else 1.0)
        put("total_weight_g", tw, "mapped")
        put("avg_weight_g", tw / ni, "derived")
    if num("volume") is not None:
        put("total_volume_cm3", num("volume"), "mapped")
    elif all(num(k) is not None for k in ("length", "height", "width")):
        put("total_volume_cm3", num("length") * num("height") * num("width"), "derived")

    # --- geography
    cs = col("customer_state")
    ss = col("seller_state")
    if cs is not None:
        put("customer_state", cs.map(_clean_state), "mapped")
    if ss is not None:
        put("seller_state", ss.map(_clean_state), "mapped")
    if "customer_state" in out and "seller_state" in out:
        put("same_state", (out["customer_state"] == out["seller_state"]).astype(float), "derived")
    if num("distance") is not None:
        f = {"km": 1.0, "miles": 1.609344, "m": 0.001}.get(opt("distance", "unit", "km"), 1.0)
        put("seller_customer_distance_km", num("distance") * f, "mapped")
    elif all(num(k) is not None for k in ("customer_lat", "customer_lon", "seller_lat", "seller_lon")):
        put("seller_customer_distance_km",
            pd.Series(haversine_km(num("customer_lat"), num("customer_lon"), num("seller_lat"), num("seller_lon")), index=ix), "derived")

    # --- product and payment
    if col("category") is not None:
        put("product_category", col("category").map(_clean_category), "mapped")
    if col("payment_type") is not None:
        put("payment_type", col("payment_type").map(_clean_payment), "mapped")

    # --- anything still missing comes from a typical order, and says so
    for feat in FEATURE_COLUMNS:
        if feat in RATE_FEATURES or feat in out:
            continue
        v = typical.get(feat)
        if feat in CATEGORICAL:
            v = "unknown" if feat in ("product_category", "payment_type") else v
        out[feat] = pd.Series(v, index=ix)
        prov[feat] = "default"
    # a mapped column with some blank cells: fill the blanks with the typical value, per cell
    for feat, how in list(prov.items()):
        if how == "default":
            continue
        s = out[feat]
        if feat in NUMERIC:
            s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            blanks = s.isna()
            if blanks.any():
                s = s.fillna(typical.get(feat))
                notes.append(f"{int(blanks.sum()):,} blank or unreadable value(s) in '{feat}' were filled with a typical value.")
            out[feat] = s.astype(float)
        else:
            blanks = s.isna()
            if blanks.any():
                s = s.fillna(typical.get(feat, "unknown"))
                notes.append(f"{int(blanks.sum()):,} blank value(s) in '{feat}' were filled with a typical value.")
            out[feat] = s

    table = pd.DataFrame({k: out[k].values for k in out if k not in RATE_FEATURES}, index=ix)
    table.insert(0, "order_id", col("order_id").astype(str).values if col("order_id") is not None else [f"row{i + 1}" for i in range(n)])
    table.insert(1, "seller_id", col("seller_id").astype(str).values if col("seller_id") is not None else [None] * n)

    real = [f for f in FEATURE_COLUMNS if prov.get(f) in ("mapped", "derived")]
    w = weights or {f: 1.0 for f in FEATURE_COLUMNS}
    tot = sum(w.get(f, 0.0) for f in FEATURE_COLUMNS) or 1.0
    # rates are filled from seller / category / state history, so they count when their key was supplied
    rate_key = {"seller_historical_delay_rate": "seller_id", "category_historical_delay_rate": "product_category",
                "customer_state_historical_delay_rate": "customer_state"}
    for rf, key in rate_key.items():
        prov[rf] = "lookup" if (key == "seller_id" and src("seller_id")) or prov.get(key) == "mapped" else "default"
    real_w = sum(w.get(f, 0.0) for f in real)
    info = {
        "provenance": prov, "notes": notes[:12],
        "n_real": len(real), "n_features": len(FEATURE_COLUMNS),
        "importance_coverage": round(real_w / tot, 3),
        "defaulted": [f for f in FEATURE_COLUMNS if prov.get(f) == "default"],
    }
    return table, info


# --------------------------------------------------------------------------- adaptive training
def _preprocessor(numeric, categorical):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    parts = []
    if numeric:
        parts.append(("num", StandardScaler(), numeric))
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(parts)


def _xgb_pipeline(numeric, categorical, spw):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9,
                        min_child_weight=5, scale_pos_weight=spw, eval_metric="auc", tree_method="hist",
                        random_state=42, n_jobs=1)
    return Pipeline([("pre", _preprocessor(numeric, categorical)), ("clf", clf)])


def add_own_history_rates(table, provenance, purchase_time, delivered_time, late):
    """Seller / category / state past-delay rates computed from the user's own
    history with the same leak-free rule as the Olist model (only orders
    already delivered before this one was placed). Returns (table, added)."""
    t = table.copy()
    t["_purchase"], t["_delivered"], t[TARGET_COLUMN] = purchase_time.values, delivered_time.values, late.values
    t = t[t["_purchase"].notna() & t["_delivered"].notna() & t[TARGET_COLUMN].notna()].copy()
    added = []
    keys = {"seller_historical_delay_rate": "seller_id", "category_historical_delay_rate": "product_category",
            "customer_state_historical_delay_rate": "customer_state"}
    for feat, key in keys.items():
        if key == "seller_id" and t["seller_id"].isna().all():
            continue
        if key != "seller_id" and provenance.get(key) != "mapped":
            continue
        t[key] = t[key].fillna("unknown") if key != "seller_id" else t[key].fillna("unknown")
        t = add_historical_rates(t, key, feat, known_at="_delivered", as_of="_purchase")
        added.append(feat)
    return t, added


def train_adaptive(table, info, late, purchase_time=None, delivered_time=None, max_rows=60000, log=print):
    """Train an XGBoost model on the user's own labelled history.

    Only inputs the file really supplies are used (mapped or derived, plus
    leak-free past-delay rates when the file has delivered dates). The split
    is by time when the file has order dates (the last 20% of orders are the
    test set), otherwise stratified random.
    Returns (pipeline, spec, report).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from modeling import find_best_threshold

    prov = info["provenance"]
    labelled = late.notna().values
    t = table.loc[labelled].copy()
    t["_row"] = np.flatnonzero(labelled)
    y = late[labelled].astype(int).reset_index(drop=True)
    pt = purchase_time[labelled] if purchase_time is not None else None
    dt = delivered_time[labelled] if delivered_time is not None else None
    if len(t) < 300 or y.nunique() < 2 or y.sum() < 30:
        raise ValueError(f"Not enough labelled orders to learn from: {len(t):,} with an outcome and {int(y.sum())} late. "
                         "At least 300 orders with 30 late ones are needed.")

    rate_added = []
    if pt is not None and dt is not None and pt.notna().any() and dt.notna().any():
        rt, rate_added = add_own_history_rates(t.assign(_y=y.values), prov, pt, dt, y)
        if len(rt) >= 300:
            t, y = rt.drop(columns=["_purchase", "_delivered", TARGET_COLUMN]), rt[TARGET_COLUMN].astype(int).reset_index(drop=True)
            pt = rt["_purchase"]
    real = [f for f in FEATURE_COLUMNS if prov.get(f) in ("mapped", "derived") or f in rate_added]
    numeric = [f for f in real if f in NUMERIC or f in RATE_FEATURES]
    categorical = [f for f in real if f in CATEGORICAL]
    # a column with a single value carries no information
    numeric = [f for f in numeric if t[f].nunique() > 1]
    categorical = [f for f in categorical if t[f].nunique() > 1]
    if not numeric and not categorical:
        raise ValueError("None of the mapped columns varies between orders, so there is nothing to learn from. Check the mapping.")
    t = t.reset_index(drop=True)
    if len(t) > max_rows:
        keep = np.sort(np.random.RandomState(42).choice(len(t), max_rows, replace=False))
        t, y = t.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)
        pt = pt.iloc[keep].reset_index(drop=True) if pt is not None else None

    if pt is not None and pd.Series(pt).notna().all():
        split = "time"
        order = np.argsort(pd.Series(pt).values, kind="stable")
        cut = int(len(order) * 0.8)
        tr_idx, te_idx = order[:cut], order[cut:]
    else:
        split = "random"
        tr_idx, te_idx = train_test_split(np.arange(len(t)), test_size=0.2, stratify=y, random_state=42)
    cols = numeric + categorical
    Xtr, ytr, Xte, yte = t.iloc[tr_idx][cols], y.iloc[tr_idx], t.iloc[te_idx][cols], y.iloc[te_idx]
    if yte.nunique() < 2:
        raise ValueError("The most recent orders contain no late deliveries, so the model cannot be tested. Use a longer history.")

    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    # threshold from a validation slice of the TRAIN rows (never the test rows)
    cut_v = int(len(Xtr) * 0.85)
    pipe_v = _xgb_pipeline(numeric, categorical, spw).fit(Xtr.iloc[:cut_v], ytr.iloc[:cut_v])
    if ytr.iloc[cut_v:].nunique() == 2:
        thr, _ = find_best_threshold(ytr.iloc[cut_v:].values, pipe_v.predict_proba(Xtr.iloc[cut_v:])[:, 1])
    else:
        thr = 0.5
    pipe = _xgb_pipeline(numeric, categorical, spw).fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    pred = p >= thr
    tp = int((pred & (yte.values == 1)).sum())
    report = {
        "split": split, "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "late_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(yte, p)), 4), "pr_auc": round(float(average_precision_score(yte, p)), 4),
        "threshold": round(float(thr), 4), "precision": round(tp / max(int(pred.sum()), 1), 4),
        "recall": round(tp / max(int(yte.sum()), 1), 4), "flagged_share": round(float(pred.mean()), 4),
        "features_used": cols, "own_history_rates": rate_added,
        "test_index": [int(i) for i in te_idx[:0]],  # kept small; the caller re-scores what it needs
    }
    # per-feature importance (one-hot columns summed back to their feature)
    from sklearn.pipeline import Pipeline  # noqa: F401
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    imp = clf.feature_importances_
    names = []
    if numeric:
        names += numeric
    if categorical:
        for c, cats in zip(categorical, pre.named_transformers_["cat"].categories_):
            names += [c] * len(cats)
    agg = {}
    for nme, v in zip(names, imp):
        agg[nme] = agg.get(nme, 0.0) + float(v)
    report["importance"] = sorted(({"feature": k, "importance": round(v, 4)} for k, v in agg.items()), key=lambda r: -r["importance"])[:10]
    # lookups so later files (no outcomes) can get the same rate features
    lookup = {"global_rate": float(y.mean())}
    full = t.assign(_y=y.values)
    for feat, key in {"seller_historical_delay_rate": "seller_id", "category_historical_delay_rate": "product_category",
                      "customer_state_historical_delay_rate": "customer_state"}.items():
        if feat in rate_added and key in full.columns:
            g = full.groupby(key)["_y"].agg(["mean", "count"])
            lookup[feat] = {str(k): float(r["mean"]) for k, r in g.iterrows() if r["count"] >= 5}
    spec = {"numeric": numeric, "categorical": categorical, "threshold": float(thr), "rate_lookup": lookup,
            "split": split, "metrics": {k: report[k] for k in ("roc_auc", "pr_auc", "precision", "recall", "flagged_share", "n_train", "n_test")}}
    report.pop("test_index")
    report["_test_rows"] = (t.iloc[te_idx]["_row"].values, y.iloc[te_idx].values, p)
    return pipe, spec, report


def predict_adaptive(pipe, spec, table):
    """Probabilities from an adapted model for a mapped table."""
    t = table.copy()
    for feat, key in {"seller_historical_delay_rate": "seller_id", "category_historical_delay_rate": "product_category",
                      "customer_state_historical_delay_rate": "customer_state"}.items():
        if feat in spec["numeric"]:
            look = spec["rate_lookup"].get(feat, {})
            g = spec["rate_lookup"]["global_rate"]
            t[feat] = t[key].astype(str).map(look).fillna(g).astype(float)
    cols = spec["numeric"] + spec["categorical"]
    return pipe.predict_proba(t[cols])[:, 1]


def save_adapted(directory, pipe, spec, mapping, name):
    os.makedirs(directory, exist_ok=True)
    joblib.dump(pipe.named_steps["pre"], os.path.join(directory, "pre.joblib"))
    pipe.named_steps["clf"].save_model(os.path.join(directory, "booster.json"))
    meta = {"name": name, "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "spec": spec, "mapping": mapping}
    with open(os.path.join(directory, "adapted.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_adapted(directory):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    with open(os.path.join(directory, "adapted.json"), encoding="utf-8") as f:
        meta = json.load(f)
    clf = XGBClassifier()
    clf.load_model(os.path.join(directory, "booster.json"))
    pre = joblib.load(os.path.join(directory, "pre.joblib"))
    return Pipeline([("pre", pre), ("clf", clf)]), meta


def risk_band(p):
    from modeling import risk_bucket
    return risk_bucket(float(p))
