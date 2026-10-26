"""
clustering.py
--------------
Groups LATE orders into patterns that share a cause, so the team can fix a
pattern instead of chasing orders one by one (mentor's guidance: "form
different clusters and understand why problems are happening for these
types of deliveries, products or areas").

Method
  * Only orders that were actually late are clustered (7.8k of 96k).
  * Features are things a logistics manager can act on, all known at
    checkout: route distance, promised window, the seller's past late
    rate, peak-season timing, parcel weight, order value, freight share,
    same-state flag, payment approval delay, and the destination region.
  * K-Means on standardised features; k chosen by silhouette score
    between 4 and 7 (more than 7 patterns is too many to act on).

Profiling, done carefully
  * A cluster is described by what makes it DIFFERENT from late orders in
    general (z-scores of cluster means against all late orders), not by
    what is common everywhere. A first version said "concentrated in RJ"
    for almost every cluster simply because RJ and SP dominate the whole
    dataset; states and categories are therefore reported by LIFT (share
    in the cluster / share among all late orders), with a minimum share.
  * Every distinguishing feature has its own recommendation branch, so no
    feature silently falls through to a generic message (an earlier bug:
    total order value had no branch).
"""
import math

import numpy as np
import pandas as pd

from simulation import REGIONS, HUB_OF_REGION

PEAK_MONTHS = {11, 12, 2, 3}
NUMERIC_FEATURES = {
    # name in profile      : (column, transform for clustering)
    "distance_km": ("seller_customer_distance_km", np.log1p),
    "promised_days": ("promised_delivery_days", None),
    "seller_late_rate": ("seller_historical_delay_rate", None),
    "peak_season": ("_peak", None),
    "weight_kg": ("_weight_kg", np.log1p),
    "order_value": ("total_order_value", np.log1p),
    "freight_share": ("_freight_share", None),
    "same_state": ("same_state", None),
    "approval_hours": ("approval_delay_hours", np.log1p),
}
LABELS = {
    "distance_km": "route distance", "promised_days": "promised window", "seller_late_rate": "seller's past late rate",
    "peak_season": "peak-season share", "weight_kg": "parcel weight", "order_value": "order value",
    "freight_share": "freight share of the price", "same_state": "same-state share", "approval_hours": "payment approval time",
}


def _prepare(df):
    d = df.copy()
    d["_peak"] = d["purchase_month"].isin(PEAK_MONTHS).astype(float)
    d["_weight_kg"] = d["total_weight_g"].fillna(0) / 1000
    d["_freight_share"] = (d["total_freight"] / d["total_order_value"].replace(0, np.nan)).fillna(0).clip(0, 1)
    d["seller_customer_distance_km"] = d["seller_customer_distance_km"].fillna(d["seller_customer_distance_km"].median())
    d["approval_delay_hours"] = d["approval_delay_hours"].fillna(0).clip(lower=0)
    return d


def _matrix(d, mean=None, std=None):
    cols = []
    for name, (col, tf) in NUMERIC_FEATURES.items():
        v = d[col].astype(float).values
        cols.append(tf(v) if tf else v)
    X = np.column_stack(cols)
    if mean is None:
        mean, std = X.mean(0), X.std(0)
        std[std == 0] = 1
    Xs = (X - mean) / std
    reg = np.column_stack([(d["customer_region"] == r).astype(float).values * 0.8 for r in REGIONS])
    return np.hstack([Xs, reg]), mean, std


def fit_clusters(late, k_range=range(4, 8), random_state=42):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    d = _prepare(late)
    X, mean, std = _matrix(d)
    rng = np.random.RandomState(random_state)
    sample = rng.choice(len(X), size=min(4000, len(X)), replace=False)
    best = None
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        s = silhouette_score(X[sample], km.labels_[sample])
        scores[k] = round(float(s), 4)
        if best is None or s > best[0]:
            best = (s, k, km)
    _, k, km = best
    params = {"k": int(k), "mean": mean.tolist(), "std": std.tolist(), "centers": km.cluster_centers_.tolist(),
              "silhouette_by_k": scores}
    return km.labels_, params


def assign(df, params):
    d = _prepare(df)
    X, _, _ = _matrix(d, np.array(params["mean"]), np.array(params["std"]))
    C = np.array(params["centers"])
    dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return dist.argmin(1)


def _lift_table(cluster_s, all_s, min_share=0.06, top=3):
    c = cluster_s.value_counts(normalize=True)
    a = all_s.value_counts(normalize=True)
    rows = []
    for key, share in c.items():
        if share < min_share:
            continue
        lift = share / a.get(key, np.nan)
        rows.append({"key": str(key), "share": round(float(share) * 100, 1), "lift": round(float(lift), 2)})
    rows.sort(key=lambda r: (r["lift"], r["share"]), reverse=True)
    return rows[:top]


def profile_clusters(late, labels, params):
    d = _prepare(late).assign(cluster=labels)
    overall = {name: (d[col].mean(), d[col].std()) for name, (col, _) in NUMERIC_FEATURES.items()}
    total = len(d)
    profiles = []
    for c in range(params["k"]):
        g = d[d.cluster == c]
        z = {}
        for name, (col, _) in NUMERIC_FEATURES.items():
            mu, sd = overall[name]
            z[name] = round(float((g[col].mean() - mu) / (sd or 1)), 2)
        dist_feats = sorted(((n, v) for n, v in z.items() if abs(v) >= 0.35), key=lambda t: abs(t[1]), reverse=True)
        p = {
            "id": int(c), "n": int(len(g)), "share_of_late": round(len(g) / total * 100, 1),
            "median_distance_km": round(float(g["seller_customer_distance_km"].median())),
            "median_promised_days": round(float(g["promised_delivery_days"].median()), 1),
            "median_actual_days": round(float(g["actual_days"].median()), 1),
            "p75_actual_days": round(float(g["actual_days"].quantile(0.75)), 1),
            "median_days_late": round(float(g["delay_days"].median()), 1),
            "peak_share": round(float(g["_peak"].mean()) * 100, 1),
            "median_weight_kg": round(float(g["_weight_kg"].median()), 2),
            "median_order_value": round(float(g["total_order_value"].median()), 2),
            "median_seller_late_rate": round(float(g["seller_historical_delay_rate"].median()) * 100, 1),
            "median_approval_hours": round(float(g["approval_delay_hours"].median()), 1),
            "same_state_share": round(float(g["same_state"].mean()) * 100, 1),
            "z": z,
            "distinguishing": [{"feature": n, "label": LABELS[n], "z": v} for n, v in dist_feats[:4]],
            "regions": _lift_table(g["customer_region"], d["customer_region"], 0.08),
            "states": _lift_table(g["customer_state"], d["customer_state"], 0.06),
            "categories": _lift_table(g["product_category"], d["product_category"], 0.04),
            "top_sellers": _seller_concentration(g),
            "logistics": {  # simulated
                "delivery_model": {k: round(v * 100, 1) for k, v in g["sim_delivery_model"].value_counts(normalize=True).items()},
                "median_transfers": float(g["sim_hub_transfers"].median()),
                "shipping_mode": {k: round(v * 100, 1) for k, v in g["sim_shipping_mode"].value_counts(normalize=True).items()},
                "main_last_mile_hub": str(g["sim_last_mile_hub"].mode().iloc[0]),
                "main_first_mile_hub": str(g["sim_first_mile_hub"].mode().iloc[0]),
            },
            "top_city": str(g["customer_city"].fillna("").str.title().mode().iloc[0]) if g["customer_city"].notna().any() else "",
        }
        p["name"] = _name(p)
        p["recommendations"] = recommendations(p)
        profiles.append(p)
    # unique names
    seen = {}
    for p in profiles:
        if p["name"] in seen:
            seen[p["name"]] += 1
            p["name"] = f'{p["name"]} ({seen[p["name"]]})'
        else:
            seen[p["name"]] = 1
    profiles.sort(key=lambda p: p["n"], reverse=True)
    for rank, p in enumerate(profiles, 1):
        p["rank"] = rank
    return profiles


def _seller_concentration(g):
    vc = g["seller_id"].value_counts()
    top10 = vc.head(10).sum() / max(len(g), 1)
    return {"n_sellers": int(vc.size), "top10_share": round(float(top10) * 100, 1)}


def _region_phrase(p):
    if p["regions"] and p["regions"][0]["lift"] >= 1.3:
        return f'to the {p["regions"][0]["key"]}'
    return ""


def _name(p):
    z = p["z"]
    feats = [d["feature"] for d in p["distinguishing"]]
    lead = feats[0] if feats else ""
    words = {
        "distance_km": "Long-haul" if z["distance_km"] > 0 else "Short-route",
        "promised_days": "Tight-promise" if z["promised_days"] < 0 else "Long-promise",
        "seller_late_rate": "Slow-seller" if z["seller_late_rate"] > 0 else "Reliable-seller",
        "peak_season": "Peak-season" if z["peak_season"] > 0 else "Off-season",
        "weight_kg": "Bulky-parcel" if z["weight_kg"] > 0 else "Light-parcel",
        "order_value": "High-value" if z["order_value"] > 0 else "Low-value",
        "freight_share": "Freight-heavy" if z["freight_share"] > 0 else "Low-freight",
        "same_state": "Local last-mile" if z["same_state"] > 0 else "Interstate",
        "approval_hours": "Slow-approval" if z["approval_hours"] > 0 else "Fast-approval",
    }
    head = words.get(lead, "Mixed")
    second = words.get(feats[1], "") if len(feats) > 1 else ""
    name = head if not second or second.split("-")[0] in head else f"{head}, {second.lower()}"
    reg = _region_phrase(p)
    return f"{name} orders {reg}".strip()


def recommendations(p):
    """Concrete, cluster-specific actions. One branch per feature."""
    z, recs = p["z"], []
    hub_to = HUB_OF_REGION.get(p["regions"][0]["key"], "regional") if p["regions"] else "regional"
    region_txt = p["regions"][0]["key"] if p["regions"] else "these destinations"
    for d in p["distinguishing"]:
        f, v = d["feature"], d["z"]
        if f == "distance_km" and v > 0:
            recs.append(f"Pre-position fast-moving stock at the {hub_to} hub so orders to the {region_txt} ship regionally "
                        f"instead of travelling a median {p['median_distance_km']:,} km; book long-haul express for the rest.")
        elif f == "distance_km":
            recs.append(f"These are short routes (median {p['median_distance_km']:,} km) that still arrive late, so the delay is "
                        "not transport distance: look at seller dispatch and last-mile delivery instead.")
        elif f == "promised_days" and v < 0:
            floor = math.ceil(p["p75_actual_days"])
            recs.append(f"Stop promising about {p['median_promised_days']:.0f}-day delivery for orders like these: they actually took "
                        f"{p['median_actual_days']:.0f} days (median) and {p['p75_actual_days']:.0f} days for 1 in 4. "
                        f"Set the minimum promise to {floor} days on these routes.")
        elif f == "promised_days":
            recs.append(f"The promise was already generous (median {p['median_promised_days']:.0f} days) and orders still ran "
                        f"{p['median_days_late']:.0f} days over, which points at stalled shipments: chase any order with no "
                        "carrier scan after 5 days.")
        elif f == "seller_late_rate" and v > 0:
            recs.append(f"Seller reliability is the lever: the median seller here was already late {p['median_seller_late_rate']:.0f}% "
                        f"of the time before these orders. Put sellers above 15% on a 48-hour dispatch SLA with a weekly scorecard"
                        + (f"; the top 10 sellers alone account for {p['top_sellers']['top10_share']:.0f}% of this pattern." if p['top_sellers']['top10_share'] >= 20 else "."))
        elif f == "seller_late_rate":
            recs.append("Sellers here have good track records, so look beyond the seller: route, season or carrier.")
        elif f == "peak_season" and v > 0:
            recs.append(f"{p['peak_share']:.0f}% of these orders were placed in Nov or Feb-Mar. Add 3-5 buffer days to promises in "
                        "those months and book extra carrier capacity two weeks before the peak starts.")
        elif f == "peak_season":
            recs.append("These delays happen outside peak months, so they are structural (route or seller) rather than capacity-driven.")
        elif f == "weight_kg" and v > 0:
            recs.append(f"Parcels are heavy (median {p['median_weight_kg']:.1f} kg). Route them to a bulky-freight service and book "
                        "delivery slots with the customer to avoid failed attempts.")
        elif f == "weight_kg":
            recs.append("Light parcels: these suit a faster small-parcel service if the route allows.")
        elif f == "order_value" and v > 0:
            recs.append(f"High-value orders (median R$ {p['median_order_value']:,.0f}): give them tracked, insured shipping and a "
                        "proactive call when a delay is predicted; they cost the most in refunds.")
        elif f == "order_value":
            recs.append(f"Low-value orders (median R$ {p['median_order_value']:,.0f}): batch them with the seller's other "
                        "shipments rather than paying for express.")
        elif f == "freight_share" and v > 0:
            recs.append("Freight is a large share of the price, a sign of hard-to-serve routes: negotiate a regional carrier "
                        "rate or set a minimum order value for free shipping on these routes.")
        elif f == "freight_share":
            recs.append("Freight is cheap relative to price here, so paying for a faster service is affordable.")
        elif f == "same_state" and v > 0:
            city = f" in {p['top_city']}" if p["top_city"] else ""
            recs.append(f"{p['same_state_share']:.0f}% are same-state deliveries that still ran late, which points at last-mile "
                        f"delivery{city}. Review the local courier's on-time rate there.")
        elif f == "same_state":
            recs.append("Almost all of these cross state lines; agree interstate transfer cut-off times with the carrier.")
        elif f == "approval_hours" and v > 0:
            recs.append(f"Payment approval takes a median {p['median_approval_hours']:.0f} h here. Auto-approve low-risk payments "
                        "or start picking before approval completes.")
        elif f == "approval_hours":
            recs.append("Approval is fast here, so the delay happens after dispatch.")
    # observations (negative-direction branches) go after actionable advice
    observational = ("These delays happen outside", "Almost all of these cross", "Light parcels", "Sellers here have good",
                     "Approval is fast", "Freight is cheap", "These are short routes")
    recs.sort(key=lambda r: r.startswith(observational))
    if p["categories"] and p["categories"][0]["lift"] >= 2:
        cats = [c for c in p["categories"] if c["lift"] >= 2][:2]
        names = " and ".join(c["key"].replace("_", " ") for c in cats)
        recs.insert(1, f"{names[0].upper() + names[1:]} {'are' if len(cats) > 1 else 'is'} {cats[0]['lift']:.1f}x over-represented "
                       "here: a category-specific fix (packaging, carrier choice, seller review) reaches a large part of this pattern.")
    if p["states"] and p["states"][0]["lift"] >= 1.5:
        s = p["states"][0]
        recs.append(f"{s['key']} is over-represented: {s['share']:.0f}% of this pattern, {s['lift']:.1f}x its share of late "
                    f"orders overall. A regional partner carrier for {s['key']} is worth testing.")
    if not recs:
        recs.append("No single factor stands out; monitor this pattern month by month and review it again after the next retraining.")
    return recs[:4]
