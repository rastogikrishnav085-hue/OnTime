"""
tracking_features.py - ONE feature builder for the in-transit "tracking score".
Used by train_tracking_model.py (training) and backend/app/tracking.py (the app), so they cannot drift apart.

Rules that keep it honest:
  * only events at or before `as_of` are used;
  * the DELIVERED event is never a feature (it is the outcome);
  * the tracking data behind it is SIMULATED (see data/synthetic/DATA_DICTIONARY.md).
"""
import numpy as np
import pandas as pd

DAY0 = pd.Timestamp("2016-09-01")
TIME_FEATURES = ["days_since_purchase", "days_left", "frac_promise_left", "picked_up", "hours_since_pickup",
                 "handling_hours", "km_per_day_needed"]
TRACK_FEATURES = ["n_scans", "n_hubs", "hours_since_last_scan", "max_gap_hours", "n_failed_attempts",
                  "out_for_delivery", "carrier_code"]
WEATHER_FEATURES = ["dest_weather_3d_max", "dest_weather_7d_mean", "origin_weather_3d_max",
                    "dest_strike_now", "origin_strike_now", "dest_strike_days_7d",
                    "dest_congestion_3d_mean", "origin_congestion_3d_mean", "dest_road_block_now", "origin_road_block_now",
                    "holidays_left"]
LABELS = {
    "carrier_code": "Which carrier has the parcel",
    "dest_congestion_3d_mean": "Hub congestion at the destination (last 3 days)",
    "origin_congestion_3d_mean": "Hub congestion at the seller's state (last 3 days)",
    "dest_road_block_now": "Road disruption in the destination state",
    "origin_road_block_now": "Road disruption in the seller's state",
    "holidays_left": "Public holidays left before the promised date",
    "max_gap_hours": "Longest silence between scans",
    "hours_since_last_scan": "Hours since the last scan",
    "n_failed_attempts": "Failed delivery attempts",
    "n_scans": "Hub scans so far",
    "n_hubs": "Hubs passed so far",
    "out_for_delivery": "Out for delivery",
    "dest_weather_3d_max": "Bad weather at the destination (last 3 days)",
    "dest_weather_7d_mean": "Weather at the destination (last 7 days)",
    "origin_weather_3d_max": "Bad weather at the seller's state (last 3 days)",
    "dest_strike_now": "Strike active in the destination state",
    "origin_strike_now": "Strike active in the seller's state",
    "dest_strike_days_7d": "Strike days at the destination (last 7 days)",
    "days_left": "Days of the promise left",
    "frac_promise_left": "Share of the promised window left",
    "km_per_day_needed": "Distance per day still to cover",
    "handling_hours": "Time the seller took to hand the parcel over",
    "hours_since_pickup": "Hours since the carrier collected it",
    "days_since_purchase": "Days since the order was placed",
}


class Context:
    """Weather and strike calendars as plain arrays, for fast lookups."""

    def __init__(self, weather: pd.DataFrame, strikes: pd.DataFrame, congestion=None, roads=None, holidays=None):
        w = weather.copy()
        w["date"] = pd.to_datetime(w["date"])
        self.states = sorted(w["state"].unique())
        self.sidx = {s: i for i, s in enumerate(self.states)}
        self.n_days = int((w["date"].max() - DAY0).days) + 1
        self.weather = np.zeros((len(self.states), self.n_days), dtype=np.float32)
        d = (w["date"] - DAY0).dt.days.values
        self.weather[w["state"].map(self.sidx).values, d] = w["severity"].values
        self.strike = np.zeros((len(self.states), self.n_days), dtype=np.float32)
        for r in strikes.itertuples():
            if r.state not in self.sidx:
                continue
            a, b = (pd.Timestamp(r.start_date) - DAY0).days, (pd.Timestamp(r.end_date) - DAY0).days
            self.strike[self.sidx[r.state], max(a, 0):max(b + 1, 0)] = 1.0
        self.congestion = np.zeros_like(self.weather)
        if congestion is not None:
            c = congestion.copy(); c["date"] = pd.to_datetime(c["date"])
            self.congestion[c["state"].map(self.sidx).values, (c["date"] - DAY0).dt.days.values] = c["index"].values
        self.roads = np.zeros_like(self.weather)
        if roads is not None:
            for r in roads.itertuples():
                if r.state in self.sidx:
                    a, b = (pd.Timestamp(r.start_date) - DAY0).days, (pd.Timestamp(r.end_date) - DAY0).days
                    self.roads[self.sidx[r.state], max(a, 0):max(b + 1, 0)] = 1.0
        self.holidays = np.sort(((pd.to_datetime(holidays["date"]) - DAY0).dt.days.values)) if holidays is not None else np.array([], dtype=int)

    def _look(self, arr, states, day, back):
        si = pd.Series(states).map(self.sidx).fillna(-1).astype(int).values
        out = np.zeros((back, len(si)), dtype=np.float32)
        for k in range(back):
            di = np.clip(day - k, 0, self.n_days - 1)
            out[k] = np.where(si >= 0, arr[np.clip(si, 0, None), di], 0.0)
        return out


def build_features(orders: pd.DataFrame, events: pd.DataFrame, ctx: Context, as_of) -> pd.DataFrame:
    """
    orders: one row per order with order_id, purchased_at, promised_by, approval_delay_hours,
            promised_delivery_days, seller_customer_distance_km, customer_state, seller_state.
    events: tracking events (order_id, event_time, event_type). as_of: timestamps aligned with `orders`.
    Returns a frame with TIME_FEATURES + TRACK_FEATURES + WEATHER_FEATURES, same index as `orders`.
    """
    o = pd.DataFrame({"order_id": orders["order_id"].values, "as_of": pd.to_datetime(np.asarray(as_of)),
                      "pos": np.arange(len(orders))})
    purchased = pd.to_datetime(orders["purchased_at"]).values
    promised = pd.to_datetime(orders["promised_by"]).values
    asof = o["as_of"].values
    f = pd.DataFrame(index=np.arange(len(orders)))
    days_left = (promised - asof) / np.timedelta64(1, "D")
    f["days_since_purchase"] = (asof - purchased) / np.timedelta64(1, "D")
    f["days_left"] = days_left
    f["frac_promise_left"] = days_left / np.clip(orders["promised_delivery_days"].values.astype(float), 1, None)
    f["km_per_day_needed"] = orders["seller_customer_distance_km"].values / np.clip(days_left, 0.5, None)

    e = events[events["order_id"].isin(set(o["order_id"]))].merge(o[["order_id", "as_of", "pos"]], on="order_id")
    e = e[(e["event_time"] <= e["as_of"]) & (e["event_type"] != "DELIVERED")]
    e = e.sort_values(["pos", "event_time"])
    pk_ev = e[e["event_type"] == "PICKED_UP"].drop_duplicates("pos").set_index("pos")
    pick = pk_ev["event_time"]
    f["picked_up"] = f.index.isin(pick.index).astype(float)
    pk = pick.reindex(f.index)
    f["hours_since_pickup"] = (o["as_of"] - pk.values).dt.total_seconds().values / 3600
    approved = purchased + (orders["approval_delay_hours"].fillna(0).values * 3600e9).astype("timedelta64[ns]")
    f["handling_hours"] = np.clip(((pk.values - approved) / np.timedelta64(1, "h")).astype(float), -24, 24 * 30)

    g = e.groupby("pos")
    f["carrier_code"] = pk_ev["carrier"].astype(str).str[-1].astype(float).reindex(f.index).fillna(-1).values
    f["n_scans"] = e[e["event_type"].isin(["ARRIVED_HUB", "DEPARTED_HUB"])].groupby("pos").size().reindex(f.index)
    f["n_hubs"] = e[e["event_type"] == "ARRIVED_HUB"].groupby("pos").size().reindex(f.index)
    f["n_failed_attempts"] = e[e["event_type"] == "DELIVERY_ATTEMPT_FAILED"].groupby("pos").size().reindex(f.index)
    f["out_for_delivery"] = e[e["event_type"] == "OUT_FOR_DELIVERY"].groupby("pos").size().reindex(f.index).gt(0)
    e["gap"] = e.groupby("pos")["event_time"].diff().dt.total_seconds() / 3600
    last = g["event_time"].max().reindex(f.index)
    trailing = (o["as_of"] - last.values).dt.total_seconds().values / 3600
    f["hours_since_last_scan"] = trailing
    f["max_gap_hours"] = np.fmax(e.groupby("pos")["gap"].max().reindex(f.index).values, trailing)
    for c in ("n_scans", "n_hubs", "n_failed_attempts"):
        f.loc[f["picked_up"] == 1, c] = f.loc[f["picked_up"] == 1, c].fillna(0)
    f["out_for_delivery"] = f["out_for_delivery"].fillna(False).astype(float)
    f.loc[f["picked_up"] == 0, ["n_scans", "n_hubs", "n_failed_attempts"]] = 0.0

    day = ((o["as_of"] - DAY0).dt.days).values
    dw = ctx._look(ctx.weather, orders["customer_state"].values, day, 7)
    ow = ctx._look(ctx.weather, orders["seller_state"].values, day, 3)
    f["dest_weather_3d_max"] = dw[:3].max(0)
    f["dest_weather_7d_mean"] = dw.mean(0)
    f["origin_weather_3d_max"] = ow.max(0)
    ds = ctx._look(ctx.strike, orders["customer_state"].values, day, 7)
    f["dest_strike_now"] = ds[0]
    f["dest_strike_days_7d"] = ds.sum(0)
    f["origin_strike_now"] = ctx._look(ctx.strike, orders["seller_state"].values, day, 1)[0]
    f["dest_congestion_3d_mean"] = ctx._look(ctx.congestion, orders["customer_state"].values, day, 3).mean(0)
    f["origin_congestion_3d_mean"] = ctx._look(ctx.congestion, orders["seller_state"].values, day, 3).mean(0)
    f["dest_road_block_now"] = ctx._look(ctx.roads, orders["customer_state"].values, day, 1)[0]
    f["origin_road_block_now"] = ctx._look(ctx.roads, orders["seller_state"].values, day, 1)[0]
    pday = np.floor((promised - np.datetime64(DAY0)) / np.timedelta64(1, "D")).astype(int)
    h = ctx.holidays
    f["holidays_left"] = (np.searchsorted(h, pday, side="right") - np.searchsorted(h, day, side="right")).clip(min=0)
    f.index = orders.index
    return f[TIME_FEATURES + TRACK_FEATURES + WEATHER_FEATURES].astype("float32")


def load_context(folder: str) -> Context:
    """Read the calendars from a folder holding the simulated files (same names in data/ and in the app copy)."""
    import os
    rd = lambda n: pd.read_csv(os.path.join(folder, n))
    return Context(rd("weather_daily_simulated.csv"), rd("strikes_simulated.csv"), rd("hub_congestion_daily_simulated.csv"),
                   rd("road_disruptions_simulated.csv"), rd("holidays_brazil.csv"))
