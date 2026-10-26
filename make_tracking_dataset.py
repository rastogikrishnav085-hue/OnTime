"""
make_tracking_dataset.py - builds a SIMULATED companion data set for the Olist orders.

REAL (from Olist): order ids, purchase time, carrier pickup time, delivery time, promised date, states, distance;
plus the real Brazilian public-holiday calendar.
INVENTED (seeded, reproducible): every scan between pickup and delivery, failed delivery attempts, the weather
calendar, hub congestion, road disruptions, carriers and the strike calendar. Rules are in DATA_DICTIONARY.md. Version 1
was written before any model was trained; version 2 raised the signal strengths ONCE, at the owner's request, to test
the ceiling (target: ROC-AUC above 0.91). So the scores on this data are a ceiling, not evidence.
Late orders are made MORE LIKELY to stall, fail an attempt, or sit in a bad-weather or strike week. A model that finds
those signals is finding what was put there, so every result on this data is labelled SIMULATED.

Run:  python make_tracking_dataset.py            (needs olist_orders_dataset.csv in data/raw)
      python make_tracking_dataset.py --export-app-only   (rebuilds the small copy the app reads)
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "synthetic")
APP = os.path.join(ROOT, "backend", "app_data", "tracking")
SEED = 2026

DICTIONARY = """# Simulated tracking data set (companion to Olist)

**SIMULATED.** Real Olist does not publish carrier scans, failed delivery attempts, weather or strikes.
This folder is invented data built for the same orders, to show what OnTime could do if such data existed.
Never quote a result on it as a result on real Olist.

## What is real and what is invented
| Real (from Olist) | Invented (seed 2026) |
|---|---|
| order_id, purchase time, carrier pickup time (PICKED_UP), delivery time (DELIVERED), promised date, states | hub scans, out-for-delivery, failed attempts, weather calendar, strike calendar |

## Rules (simulation version 2: strengthened once, at the owner's request, after version 1 gave ROC-AUC 0.87-0.89)
* Scan times lie between the real pickup and the real delivery. Number of hub scans grows with distance (1 to 5 hubs).
* **Stall:** 70% of late orders and 8% of on-time orders get their scans bunched early, leaving a long silent gap.
* **Failed attempt:** 30% of late and 3% of on-time orders get one failed delivery attempt (a second one for 25% of those late ones).
* **Dropped scans:** each hub scan is lost with probability 8% (real feeds have gaps).
* **Weather:** per state and day, severity 0 to 3. Bad weather is more likely in Nov-Mar and in state-weeks whose real late rate was above average.
* **Carrier:** every order gets one of 5 carriers (C1 to C5). Share of late orders by carrier C1..C5: 5/10/20/30/35%; of on-time orders: 35/28/20/12/5%.
* **Hub congestion:** per state and day, an index from 0 to 1 that is higher on Mondays and Tuesdays, in Nov-Dec, and in state-weeks with an above-average real late rate.
* **Road disruptions:** per state, 1 to 4 days (landslide, flooded road, bridge closed, truck accident), more likely in state-weeks with an above-average real late rate.
* **Holidays:** the real Brazilian national holidays and carnival days for 2016 to 2018 (this file is REAL, not simulated).
* **Strikes:** per state and week, more likely when the real late rate of that state-week was above average; last 2 to 5 days.

## Files
* `tracking_events_simulated.csv`: order_id, event_time, event_type (PICKED_UP, ARRIVED_HUB, DEPARTED_HUB, OUT_FOR_DELIVERY, DELIVERY_ATTEMPT_FAILED, DELIVERED), hub_state, carrier, simulated.
* `weather_daily_simulated.csv`: date, state, severity (0 none, 1 rain, 2 heavy rain, 3 storm/flood), rain_mm, simulated.
* `strikes_simulated.csv`: strike_id, state, start_date, end_date, kind, simulated.
* `hub_congestion_daily_simulated.csv`: date, state, index (0 to 1), simulated.
* `road_disruptions_simulated.csv`: state, start_date, end_date, kind, simulated.
* `holidays_brazil.csv`: date, name, simulated (0 = real).
"""


def load_orders():
    d = pd.read_csv(os.path.join(ROOT, "data", "processed", "ontime_model_ready.csv"),
                    usecols=["order_id", "is_delayed", "customer_state", "seller_state", "seller_customer_distance_km"])
    o = pd.read_csv(os.path.join(ROOT, "data", "raw", "olist_orders_dataset.csv"),
                    parse_dates=["order_purchase_timestamp", "order_delivered_carrier_date",
                                 "order_delivered_customer_date", "order_estimated_delivery_date"])
    d = d.merge(o, on="order_id").dropna(subset=["order_delivered_carrier_date", "order_delivered_customer_date"])
    P, D = d["order_delivered_carrier_date"], d["order_delivered_customer_date"]
    ok = (P >= d["order_purchase_timestamp"]) & (P <= d["order_purchase_timestamp"] + pd.Timedelta(days=120)) \
        & (D > P + pd.Timedelta(hours=1))
    return d[ok].reset_index(drop=True)


HOLIDAYS = ["2016-09-07", "2016-10-12", "2016-11-02", "2016-11-15", "2016-12-25", "2017-01-01", "2017-02-27", "2017-02-28",
            "2017-04-14", "2017-04-21", "2017-05-01", "2017-06-15", "2017-09-07", "2017-10-12", "2017-11-02", "2017-11-15",
            "2017-12-25", "2018-01-01", "2018-02-12", "2018-02-13", "2018-03-30", "2018-04-21", "2018-05-01", "2018-05-31",
            "2018-09-07", "2018-10-12"]
CARRIERS = ["C1", "C2", "C3", "C4", "C5"]
CARRIER_P_LATE = [0.05, 0.10, 0.20, 0.30, 0.35]
CARRIER_P_ONTIME = [0.35, 0.28, 0.20, 0.12, 0.05]


def make_calendars(d, rng):
    days = pd.date_range("2016-09-01", "2018-10-31")
    states = sorted(set(d["customer_state"]) | set(d["seller_state"]))
    late = d.assign(week=d["order_delivered_customer_date"].dt.to_period("W").dt.start_time)
    grp = late.groupby(["customer_state", "week"])["is_delayed"].agg(["sum", "count"])
    overall = d["is_delayed"].mean()
    grp["rate"] = (grp["sum"] + 8 * overall) / (grp["count"] + 8)
    excess = (grp["rate"] - overall).to_dict()
    rows = []
    for s in states:
        for day in days:
            wk = day.to_period("W").start_time
            ex = excess.get((s, wk), 0.0)
            winter = 1.0 if day.month in (11, 12, 1, 2, 3) else 0.0
            p = float(np.clip(0.05 + 0.07 * winter + 1.2 * ex, 0.02, 0.6))
            sev = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])) if rng.random() < p else 0
            rain = round(float(rng.gamma(2.0, 4.0 * sev)), 1) if sev else 0.0
            rows.append((day.date(), s, sev, rain, 1))
    weather = pd.DataFrame(rows, columns=["date", "state", "severity", "rain_mm", "simulated"])
    srows = []
    for (s, wk), ex in excess.items():
        p = float(np.clip(0.02 + 1.0 * max(ex, 0), 0, 0.5))
        if rng.random() < p:
            start = wk + pd.Timedelta(days=int(rng.integers(0, 7)))
            srows.append((s, start.date(), (start + pd.Timedelta(days=int(rng.integers(2, 6)))).date(),
                          str(rng.choice(["carrier staff strike", "truckers strike", "port/airport strike"]))))
    strikes = pd.DataFrame(srows, columns=["state", "start_date", "end_date", "kind"])
    crow, rrows = [], []
    for s_ in states:
        for day in days:
            wk = day.to_period("W").start_time
            ex = excess.get((s_, wk), 0.0)
            wd = 0.08 if day.dayofweek in (0, 1) else 0.0            # Monday/Tuesday backlog after the weekend
            peak = 0.12 if day.month in (11, 12) else 0.0
            idx = float(np.clip(0.30 + wd + peak + 3.5 * ex + rng.normal(0, 0.08), 0, 1))
            crow.append((day.date(), s_, round(idx, 3), 1))
    congestion = pd.DataFrame(crow, columns=["date", "state", "index", "simulated"])
    for (s_, wk), ex in excess.items():
        if rng.random() < float(np.clip(0.03 + 1.2 * max(ex, 0), 0, 0.6)):
            start = wk + pd.Timedelta(days=int(rng.integers(0, 7)))
            rrows.append((s_, start.date(), (start + pd.Timedelta(days=int(rng.integers(1, 5)))).date(),
                          str(rng.choice(["landslide", "flooded road", "bridge closed", "truck accident"]))))
    roads = pd.DataFrame(rrows, columns=["state", "start_date", "end_date", "kind"])
    roads["simulated"] = 1
    strikes.insert(0, "strike_id", [f"S{i + 1:04d}" for i in range(len(strikes))])
    strikes["simulated"] = 1
    return weather, strikes, congestion, roads


def make_events(d, rng):
    n = len(d)
    P = d["order_delivered_carrier_date"].values.astype("datetime64[s]").astype("int64")
    D = d["order_delivered_customer_date"].values.astype("datetime64[s]").astype("int64")
    late = d["is_delayed"].astype(bool).values
    km = d["seller_customer_distance_km"].fillna(500).values
    n_hubs = np.clip(1 + (km // 700).astype(int), 1, 5)
    stall = rng.random(n) < np.where(late, 0.70, 0.08)
    failed = rng.random(n) < np.where(late, 0.30, 0.03)
    second = failed & late & (rng.random(n) < 0.25)
    sstate, cstate = d["seller_state"].values, d["customer_state"].values
    ids = d["order_id"].values
    carr = np.where(late, rng.choice(CARRIERS, n, p=CARRIER_P_LATE), rng.choice(CARRIERS, n, p=CARRIER_P_ONTIME))
    ts, ty, st, oid, cc = [], [], [], [], []

    def add(i, t, kind, state):
        oid.append(ids[i]); ts.append(t); ty.append(kind); st.append(state); cc.append(carr[i])

    for i in range(n):
        T = D[i] - P[i]
        add(i, P[i], "PICKED_UP", sstate[i])
        k = int(n_hubs[i])
        fr = np.sort(rng.beta(1.0, 3.0, k) * 0.55 + 0.03) if stall[i] else np.sort(rng.uniform(0.05, 0.85, k))
        for j, x in enumerate(fr):
            state = sstate[i] if x < 0.5 else cstate[i]
            t = int(P[i] + x * T)
            if rng.random() >= 0.08:
                add(i, t, "ARRIVED_HUB", state)
            if rng.random() >= 0.08:
                add(i, min(t + int(rng.integers(1800, 21600)), D[i] - 60), "DEPARTED_HUB", state)
        if failed[i]:
            add(i, max(P[i] + 60, D[i] - int(rng.uniform(0.4, 2.5) * 86400)), "DELIVERY_ATTEMPT_FAILED", cstate[i])
        if second[i]:
            add(i, max(P[i] + 60, D[i] - int(rng.uniform(0.1, 0.35) * 86400)), "DELIVERY_ATTEMPT_FAILED", cstate[i])
        add(i, max(P[i] + 60, D[i] - int(rng.uniform(2, 10) * 3600)), "OUT_FOR_DELIVERY", cstate[i])
        add(i, D[i], "DELIVERED", cstate[i])
    ev = pd.DataFrame({"order_id": oid, "event_time": pd.to_datetime(np.array(ts, dtype="int64"), unit="s"),
                       "event_type": ty, "hub_state": st, "carrier": cc, "simulated": 1})
    return ev.sort_values(["order_id", "event_time"]).reset_index(drop=True)


def export_app():
    os.makedirs(APP, exist_ok=True)
    q = set(pd.read_csv(os.path.join(ROOT, "backend", "app_data", "queue.csv"), usecols=["order_id"])["order_id"])
    ev = pd.read_csv(os.path.join(OUT, "tracking_events_simulated.csv"))
    ev = ev[ev["order_id"].isin(q)]
    ev.to_csv(os.path.join(APP, "tracking_events_simulated.csv"), index=False)
    for f in ("weather_daily_simulated.csv", "strikes_simulated.csv", "hub_congestion_daily_simulated.csv",
              "road_disruptions_simulated.csv", "holidays_brazil.csv"):
        pd.read_csv(os.path.join(OUT, f)).to_csv(os.path.join(APP, f), index=False)
    print(f"app copy: {len(ev):,} events for {ev['order_id'].nunique():,} queue orders")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = load_orders()
    print(f"{len(d):,} orders with real pickup and delivery times")
    weather, strikes, congestion, roads = make_calendars(d, rng)
    ev = make_events(d, rng)
    ev.to_csv(os.path.join(OUT, "tracking_events_simulated.csv"), index=False)
    weather.to_csv(os.path.join(OUT, "weather_daily_simulated.csv"), index=False)
    congestion.to_csv(os.path.join(OUT, "hub_congestion_daily_simulated.csv"), index=False)
    roads.to_csv(os.path.join(OUT, "road_disruptions_simulated.csv"), index=False)
    pd.DataFrame({"date": HOLIDAYS, "name": "national holiday or carnival", "simulated": 0}).to_csv(
        os.path.join(OUT, "holidays_brazil.csv"), index=False)
    strikes.to_csv(os.path.join(OUT, "strikes_simulated.csv"), index=False)
    open(os.path.join(OUT, "DATA_DICTIONARY.md"), "w", encoding="utf-8").write(DICTIONARY)
    print(f"{len(ev):,} events, {len(weather):,} weather rows, {len(strikes)} strikes")
    export_app()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-app-only", action="store_true")
    a = ap.parse_args()
    export_app() if a.export_app_only else main()
