"""
enriched_features.py
---------------------
37 extra CHECKOUT-time features computed from the REAL Olist data only (plus a real public-holiday calendar from
data/external/holidays_brazil.csv). Nothing here is simulated.

The rule for every feature, for an order bought at time t:
    it may use only what was KNOWN at t:
      * outcomes of OTHER orders that had already happened strictly before t (handed to the carrier / delivered before t),
      * the state of the network at t: which orders were bought before t and were not yet delivered at t, and which of those
        were already past their promised date (est < t),
      * the order's own checkout data (promise date, seller shipping deadline, destination, coordinates),
      * a public calendar.
    It never uses the order's own delivery outcome, and never an outcome that happened at or after t.
audit_leakage.py proves this empirically (scramble every outcome from a cut-off T onward: no feature of an order bought
before T may change).

Known simplification: the joined table holds DELIVERED orders only, so "open orders" never counts cancelled or lost
orders. That is the same in training and in the queue, and is stated in the README.

Groups (used by the ablation in train_enriched.py / evaluate_chronological.py):
    seller history, lane history, network state, place, calendar and promise
"""
import json
import os

import numpy as np
import pandas as pd

SELLER = ["seller_n_hist", "seller_handling_h_hist", "seller_transit_d_hist",
          "seller_open", "seller_overdue_open", "seller_overdue_share"]
LANE = ["lane_n_hist", "lane_late_hist", "lane_transit_d_hist", "lane_open", "lane_overdue_open", "lane_overdue_share",
        "promise_minus_lane_transit", "promise_minus_expected"]
NETWORK = ["net_open", "net_overdue_open", "net_overdue_share", "net_new_7d", "net_late_30d",
           "cstate_open", "cstate_overdue_open", "cstate_overdue_share"]
PLACE = ["cust_zip3_late_hist", "cust_zip3_n_hist", "cust_city_late_hist", "cust_city_n_hist",
         "cust_lat", "cust_lng", "sell_lat", "sell_lng"]
CALENDAR = ["shipping_limit_days", "holidays_in_promise", "promise_business_days", "days_from_black_friday",
            "promise_end_dow", "purchase_day_of_month", "promise_minus_limit"]

GROUPS = {"seller history": SELLER, "lane history": LANE, "network state": NETWORK, "place": PLACE, "calendar and promise": CALENDAR}
EXTRA_COLUMNS = SELLER + LANE + NETWORK + PLACE + CALENDAR

LABELS = {
    "seller_n_hist": "Seller's past orders on record", "seller_handling_h_hist": "Seller's usual hand-over time (hours)",
    "seller_transit_d_hist": "Seller's usual transit time (days)", "seller_open": "Seller's orders in progress now",
    "seller_overdue_open": "Seller's orders already overdue now", "seller_overdue_share": "Share of seller's open orders overdue",
    "lane_n_hist": "Past orders on this route", "lane_late_hist": "Route's past late rate",
    "lane_transit_d_hist": "Route's usual transit time (days)", "lane_open": "Orders in progress on this route",
    "lane_overdue_open": "Overdue orders on this route now", "lane_overdue_share": "Share of route's open orders overdue",
    "promise_minus_lane_transit": "Promise minus route's usual transit (days)",
    "promise_minus_expected": "Promise minus expected hand-over and transit (days)",
    "net_open": "Orders in progress, whole network", "net_overdue_open": "Overdue orders, whole network",
    "net_overdue_share": "Share of network's open orders overdue", "net_new_7d": "New orders in the last 7 days",
    "net_late_30d": "Network late rate, last 30 days", "cstate_open": "Orders in progress to this state",
    "cstate_overdue_open": "Overdue orders to this state", "cstate_overdue_share": "Share of this state's open orders overdue",
    "cust_zip3_late_hist": "Late rate of this postcode area", "cust_zip3_n_hist": "Past orders to this postcode area",
    "cust_city_late_hist": "Late rate of this city", "cust_city_n_hist": "Past orders to this city",
    "cust_lat": "Customer latitude", "cust_lng": "Customer longitude", "sell_lat": "Seller latitude", "sell_lng": "Seller longitude",
    "shipping_limit_days": "Seller's shipping deadline (days after purchase)", "holidays_in_promise": "Public holidays inside the promise window",
    "promise_business_days": "Business days inside the promise window", "days_from_black_friday": "Days from Black Friday",
    "promise_end_dow": "Weekday of the promised date", "purchase_day_of_month": "Day of the month of purchase",
    "promise_minus_limit": "Promise minus seller's shipping deadline (days)",
}

SHRINK = 10.0                       # pseudo-orders pulling a thin late-rate history towards the network rate at that time
BLACK_FRIDAYS = pd.to_datetime(["2016-11-25", "2017-11-24", "2018-11-23"])
NS_DAY = 86_400 * 1_000_000_000


def _ns(s):
    """datetime Series -> int64 ns, NaT -> None-mask."""
    v = pd.to_datetime(s).values.astype("datetime64[ns]")
    return v.astype("int64"), ~np.isnat(v)


def _groups(keys, n):
    if keys is None:
        yield np.arange(n)
        return
    for _, idx in pd.Series(keys).groupby(keys, dropna=True).indices.items():
        yield idx


def _past_stats(keys, known, ok, values, asof):
    """For every order: sum and count of `values` of OTHER orders in its group whose `known` time is strictly before its
    `asof` time. Orders with a missing known time / value are ignored."""
    n = len(asof)
    s, c = np.full(n, np.nan), np.zeros(n)
    valid = ok & ~np.isnan(values)
    for idx in _groups(keys, n):
        v = idx[valid[idx]]
        if len(v) == 0:
            c[idx] = 0
            s[idx] = 0.0
            continue
        o = np.argsort(known[v], kind="stable")
        ks, cs = known[v][o], np.concatenate([[0.0], np.cumsum(values[v][o])])
        pos = np.searchsorted(ks, asof[idx], side="left")
        s[idx], c[idx] = cs[pos], pos
    return s, c


def _mean(s, c):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _open_counts(keys, buy, deliv, est, asof):
    """Orders of the group bought before t and not delivered at t (open), and of those the ones already past their promise
    (overdue). Overdue = promise < t <= delivery; only LATE orders (delivery > promise) can ever be overdue."""
    n = len(asof)
    op, ov = np.zeros(n), np.zeros(n)
    late = deliv > est
    for idx in _groups(keys, n):
        b, d = np.sort(buy[idx]), np.sort(deliv[idx])
        t = asof[idx]
        op[idx] = np.searchsorted(b, t, side="left") - np.searchsorted(d, t, side="left")
        l = idx[late[idx]]
        if len(l):
            ov[idx] = np.searchsorted(np.sort(est[l]), t, side="left") - np.searchsorted(np.sort(deliv[l]), t, side="left")
    return op, ov


def _share(overdue, open_):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(open_ > 0, overdue / np.maximum(open_, 1), np.nan)


def _late_rate(keys, deliv, est, asof, net_rate):
    """Shrunk late rate of the group's orders delivered strictly before t, and how many there were."""
    late = (deliv > est).astype(float)
    s, c = _past_stats(keys, deliv, np.ones(len(asof), bool), late, asof)
    return (s + SHRINK * net_rate) / (c + SHRINK), c


def load_holidays(path):
    h = pd.read_csv(path, parse_dates=["date"])
    if "simulated" in h.columns:
        h = h[h["simulated"].astype(str).isin(["0", "False", "false"])]
    return pd.DatetimeIndex(sorted(h["date"].dt.normalize().unique()))


def build_extra_features(core, holidays_path):
    """core: the feature table of build_feature_table (still holding the joined columns). Row order is preserved."""
    d = core.reset_index(drop=True)
    n = len(d)
    buy, _ = _ns(d["order_purchase_timestamp"])
    deliv, _ = _ns(d["order_delivered_customer_date"])
    est, _ = _ns(d["order_estimated_delivery_date"])
    car, car_ok = _ns(d["order_delivered_carrier_date"])
    asof = buy
    seller = d["primary_seller_id"].astype(str).to_numpy()
    lane = (d["seller_state"].astype(str) + ">" + d["customer_state"].astype(str)).to_numpy()
    cstate = d["customer_state"].astype(str).to_numpy()
    zip3 = (pd.to_numeric(d["customer_zip_code_prefix"], errors="coerce") // 100).to_numpy()
    zip3 = np.where(np.isnan(zip3), "na", zip3.astype(str))
    city = d["customer_city"].astype(str).to_numpy()
    out = pd.DataFrame(index=d.index)

    # network late rate known at t (all deliveries strictly before t): the prior thin groups are shrunk towards
    late_all = (deliv > est).astype(float)
    s, c = _past_stats(None, deliv, np.ones(n, bool), late_all, asof)
    # before anything has been delivered there is no history at all: use a fixed, outcome-free constant
    net_rate = np.where(c > 0, s / np.maximum(c, 1), 0.08)

    # ---- seller history
    hand_h = (car - buy) / (3600 * 1e9)
    car_ok = car_ok & (hand_h >= 0) & (hand_h <= 24 * 60)      # a hand-over before purchase / after 60 days is a recording error
    s, c = _past_stats(seller, car, car_ok, np.where(car_ok, hand_h, np.nan), asof)
    out["seller_handling_h_hist"] = _mean(s, c)
    transit_d = np.where(car_ok, (deliv - car) / (NS_DAY), np.nan)
    transit_d = np.where((transit_d >= 0) & (transit_d <= 120), transit_d, np.nan)
    s, c = _past_stats(seller, deliv, np.ones(n, bool), transit_d, asof)
    out["seller_transit_d_hist"] = _mean(s, c)
    out["seller_n_hist"] = _past_stats(seller, deliv, np.ones(n, bool), np.zeros(n), asof)[1]
    op, ov = _open_counts(seller, buy, deliv, est, asof)
    out["seller_open"], out["seller_overdue_open"], out["seller_overdue_share"] = op, ov, _share(ov, op)

    # ---- lane (seller state -> customer state)
    out["lane_late_hist"], out["lane_n_hist"] = _late_rate(lane, deliv, est, asof, net_rate)
    s, c = _past_stats(lane, deliv, np.ones(n, bool), transit_d, asof)
    out["lane_transit_d_hist"] = _mean(s, c)
    op, ov = _open_counts(lane, buy, deliv, est, asof)
    out["lane_open"], out["lane_overdue_open"], out["lane_overdue_share"] = op, ov, _share(ov, op)
    promised = ((est - buy) / NS_DAY).astype(float)
    out["promise_minus_lane_transit"] = promised - out["lane_transit_d_hist"]
    out["promise_minus_expected"] = promised - (out["seller_handling_h_hist"] / 24.0 + out["lane_transit_d_hist"])   # blank if either history is blank (no data-wide fill: that would leak)

    # ---- network
    op, ov = _open_counts(None, buy, deliv, est, asof)
    out["net_open"], out["net_overdue_open"], out["net_overdue_share"] = op, ov, _share(ov, op)
    sb = np.sort(buy)
    out["net_new_7d"] = np.searchsorted(sb, asof, side="left") - np.searchsorted(sb, asof - 7 * NS_DAY, side="left")
    o = np.argsort(deliv, kind="stable")
    ds, cl = deliv[o], np.concatenate([[0.0], np.cumsum(late_all[o])])
    hi, lo = np.searchsorted(ds, asof, side="left"), np.searchsorted(ds, asof - 30 * NS_DAY, side="left")
    cnt = hi - lo
    out["net_late_30d"] = np.where(cnt >= 30, (cl[hi] - cl[lo]) / np.maximum(cnt, 1), np.nan)
    op, ov = _open_counts(cstate, buy, deliv, est, asof)
    out["cstate_open"], out["cstate_overdue_open"], out["cstate_overdue_share"] = op, ov, _share(ov, op)

    # ---- place
    out["cust_zip3_late_hist"], out["cust_zip3_n_hist"] = _late_rate(zip3, deliv, est, asof, net_rate)
    out["cust_city_late_hist"], out["cust_city_n_hist"] = _late_rate(city, deliv, est, asof, net_rate)
    for c_ in ("cust_lat", "cust_lng", "sell_lat", "sell_lng"):
        out[c_] = pd.to_numeric(d[c_], errors="coerce").astype(float)

    # ---- calendar and promise
    pt, et = pd.to_datetime(d["order_purchase_timestamp"]), pd.to_datetime(d["order_estimated_delivery_date"])
    hol = load_holidays(holidays_path)
    lim = pd.to_datetime(d["shipping_limit_min"]) if "shipping_limit_min" in d.columns else pd.Series(pd.NaT, index=d.index)
    out["shipping_limit_days"] = ((lim - pt).dt.total_seconds() / 86400.0).astype(float)
    p0, e0 = pt.dt.normalize().values.astype("datetime64[D]"), et.dt.normalize().values.astype("datetime64[D]")
    hol_d = hol.values.astype("datetime64[D]")
    out["holidays_in_promise"] = (np.searchsorted(hol_d, e0, side="right") - np.searchsorted(hol_d, p0, side="left")).astype(float)
    out["promise_business_days"] = np.busday_count(p0, e0, holidays=hol_d).astype(float)
    day = pt.dt.normalize().values.astype("datetime64[D]")
    bf = BLACK_FRIDAYS.values.astype("datetime64[D]")
    diff = (day[:, None] - bf[None, :]).astype(int)
    nearest = diff[np.arange(n), np.abs(diff).argmin(axis=1)]
    out["days_from_black_friday"] = np.clip(nearest, -60, 60).astype(float)
    out["promise_end_dow"] = et.dt.dayofweek.astype(float)
    out["purchase_day_of_month"] = pt.dt.day.astype(float)
    out["promise_minus_limit"] = promised - out["shipping_limit_days"]
    return out[EXTRA_COLUMNS].astype(float)


# ---------------------------------------------------------------- serving context (hand-typed orders, uploaded CSVs)
CONTEXT_DAYS = 60


def build_serving_context(core, extra):
    """Snapshot of REAL history as of the end of the data: per seller, lane and customer state, the long-run history
    features and the typical (last 60 days) network state. Lets a hand-typed order be scored without the raw files."""
    d = pd.concat([core.reset_index(drop=True)[["primary_seller_id", "seller_state", "customer_state", "order_purchase_timestamp"]],
                   extra.reset_index(drop=True)], axis=1)
    d["lane"] = d["seller_state"].astype(str) + ">" + d["customer_state"].astype(str)
    recent = d[d["order_purchase_timestamp"] >= d["order_purchase_timestamp"].max() - pd.Timedelta(days=CONTEXT_DAYS)]

    def last(df, key, cols):
        g = df.sort_values("order_purchase_timestamp").groupby(key)[cols].last()
        return {str(k): {c: (None if pd.isna(v) else round(float(v), 4)) for c, v in row.items()} for k, row in g.iterrows()}

    def mean(df, key, cols):
        g = df.groupby(key)[cols].mean()
        return {str(k): {c: (None if pd.isna(v) else round(float(v), 4)) for c, v in row.items()} for k, row in g.iterrows()}
    s_hist = ["seller_n_hist", "seller_handling_h_hist", "seller_transit_d_hist"]
    s_net = ["seller_open", "seller_overdue_open", "seller_overdue_share"]
    l_hist = ["lane_n_hist", "lane_late_hist", "lane_transit_d_hist"]
    l_net = ["lane_open", "lane_overdue_open", "lane_overdue_share"]
    c_net = ["cstate_open", "cstate_overdue_open", "cstate_overdue_share"]
    n_net = ["net_open", "net_overdue_open", "net_overdue_share", "net_new_7d", "net_late_30d"]
    # Compute each recent-window aggregate once. Calling mean(...) inside the dict comprehension
    # repeated the same groupby thousands of times and made the reproducible build appear to hang.
    seller_hist = last(d, "primary_seller_id", s_hist)
    seller_recent = mean(recent, "primary_seller_id", s_net)
    seller = {k: {**v, **seller_recent.get(k, {})} for k, v in seller_hist.items()}
    lane_hist = last(d, "lane", l_hist)
    lane_recent = mean(recent, "lane", l_net)
    lane = {k: {**v, **lane_recent.get(k, {})} for k, v in lane_hist.items()}
    return {"as_of": str(d["order_purchase_timestamp"].max()), "context_days": CONTEXT_DAYS, "seller": seller, "lane": lane,
            "cstate": mean(recent, "customer_state", c_net),
            "network": {c: round(float(recent[c].mean()), 4) for c in n_net},
            "typical": {c: round(float(d[c].median()), 4) for c in ("shipping_limit_days", "seller_handling_h_hist", "lane_transit_d_hist")}}


def save_context(ctx, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ctx, f)


class ServingContext:
    """Fills the 37 extra columns for an order that has no precomputed values, from the real-history snapshot."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.c = json.load(f)

    def fill(self, row):
        c = self.c
        seller = str(row.get("primary_seller_id", row.get("seller_id", "")))
        lane = f"{row.get('seller_state')}>{row.get('customer_state')}"
        s, l = c["seller"].get(seller), c["lane"].get(lane)
        v = {k: np.nan for k in EXTRA_COLUMNS}
        v.update({k: (np.nan if x is None else x) for k, x in (s or {}).items()})
        v.update({k: (np.nan if x is None else x) for k, x in (l or {}).items()})
        v.update({k: (np.nan if x is None else x) for k, x in c["cstate"].get(str(row.get("customer_state")), {}).items()})
        v.update(c["network"])
        promised = pd.to_numeric(row.get("promised_delivery_days"), errors="coerce")
        v["shipping_limit_days"] = c["typical"]["shipping_limit_days"]
        v["promise_minus_lane_transit"] = promised - v["lane_transit_d_hist"]
        hand = v["seller_handling_h_hist"] if pd.notna(v["seller_handling_h_hist"]) else c["typical"]["seller_handling_h_hist"]
        v["promise_minus_expected"] = promised - (hand / 24.0 + v["lane_transit_d_hist"])
        v["promise_minus_limit"] = promised - v["shipping_limit_days"]
        if pd.notna(pd.to_numeric(row.get("purchase_dow"), errors="coerce")) and pd.notna(promised):
            v["promise_end_dow"] = float((int(row["purchase_dow"]) + int(round(promised))) % 7)
        return v, {"seller_known": s is not None, "lane_known": l is not None}
