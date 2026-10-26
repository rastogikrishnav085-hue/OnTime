"""
simulation.py
--------------
SIMULATED logistics fields. Olist does not publish how an order actually
moved: no carrier, no shipping mode, no hubs. The mentor's instruction
(review of 2 Oct 2026) was to simulate such data rather than stop at
"the dataset doesn't have it", so the business discussion about delivery
models (hub-and-spoke vs direct) has something concrete to work with.

Every field here is derived by documented rules from REAL fields
(seller/customer state, distance, promised window, freight, weight), plus
a small, reproducible amount of noise keyed on the order ID. Consequences
that the app states wherever these fields appear:

  * They are labelled "simulated" everywhere in the UI and reports.
  * They are NOT used by the risk model. The sensitivity test below shows
    why that is the right call: because they are built from real fields
    the model already has, adding them gives no real lift. Real carrier
    and route data would be needed for that.
  * They ARE useful for explaining and planning: "this pattern of late
    orders mostly moves hub-and-spoke with 3 transfers to the North-East"
    is a concrete starting point for a logistics discussion.
"""
import hashlib

import numpy as np
import pandas as pd

REGION_OF_STATE = {
    **{s: "North" for s in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
    **{s: "North-East" for s in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
    **{s: "Centre-West" for s in ["DF", "GO", "MT", "MS"]},
    **{s: "South-East" for s in ["ES", "MG", "RJ", "SP"]},
    **{s: "South" for s in ["PR", "RS", "SC"]},
}
REGIONS = ["North", "North-East", "Centre-West", "South-East", "South"]

# One (simulated) sorting hub per region, placed in a real logistics city.
HUB_OF_REGION = {
    "South-East": "Cajamar SP",
    "South": "Curitiba PR",
    "Centre-West": "Brasilia DF",
    "North-East": "Recife PE",
    "North": "Belem PA",
}
CARRIERS = ["Carrier A (national express)", "Carrier B (regional)", "Carrier C (economy)"]
DELIVERY_MODELS = ["Direct (point-to-point)", "Regional hub", "Hub-and-spoke (national)"]
SHIPPING_MODES = ["Express", "Standard", "Economy"]

SIMULATED_COLUMNS = ["sim_delivery_model", "sim_hub_transfers", "sim_shipping_mode", "sim_carrier",
                     "sim_first_mile_hub", "sim_last_mile_hub", "sim_route_km"]


def _unit_hash(key, salt):
    """Reproducible pseudo-random number in [0, 1) for a given key."""
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def region(state):
    return REGION_OF_STATE.get(str(state), "South-East")


def simulate_logistics(df, id_col="order_id", seller_col="seller_id"):
    """Return a DataFrame of simulated fields aligned with df's index.

    Rules (all documented, all derived from real columns):
      delivery model  - direct if same state and < 150 km; regional hub if
                        same region; otherwise national hub-and-spoke
      hub transfers   - 0 direct, 1 regional, 2 national, +1 for long-haul
                        into the North / North-East from another region
      shipping mode   - Express when the promise is <= 8 days or freight per
                        kg is in the top 20%; Economy when the promise is
                        >= 30 days and freight per kg is in the bottom 30%;
                        otherwise Standard; 5% of orders flipped at random
      carrier         - Express -> A, Economy -> C, regional/direct -> B,
                        with the seller's habitual carrier used 30% of the time
      route km        - straight-line distance x 1.25 (typical road circuity)
    """
    s_reg = df["seller_state"].map(region)
    c_reg = df["customer_state"].map(region)
    dist = df["seller_customer_distance_km"].fillna(df["seller_customer_distance_km"].median())
    same_state = df["seller_state"] == df["customer_state"]

    model = np.where(same_state & (dist < 150), DELIVERY_MODELS[0],
                     np.where(s_reg == c_reg, DELIVERY_MODELS[1], DELIVERY_MODELS[2]))
    transfers = np.select([model == DELIVERY_MODELS[0], model == DELIVERY_MODELS[1]], [0, 1], 2)
    long_haul = (c_reg.isin(["North", "North-East"])) & (s_reg != c_reg)
    transfers = transfers + long_haul.astype(int).values

    kg = (df["total_weight_g"].fillna(0) / 1000).clip(lower=0.1)
    fpk = df["total_freight"].fillna(0) / kg
    hi, lo = fpk.quantile(0.80), fpk.quantile(0.30)
    promise = df["promised_delivery_days"].fillna(df["promised_delivery_days"].median())
    mode = np.where((promise <= 8) | (fpk >= hi), "Express",
                    np.where((promise >= 30) & (fpk <= lo), "Economy", "Standard"))
    ids = df[id_col].astype(str).values
    flip = np.array([_unit_hash(i, "mode") < 0.05 for i in ids])
    alt = np.array([SHIPPING_MODES[int(_unit_hash(i, "alt") * 3)] for i in ids])
    mode = np.where(flip, alt, mode)

    sellers = df[seller_col].astype(str).values if seller_col in df.columns else ids
    habitual = np.array([CARRIERS[int(_unit_hash(s, "carrier") * 3)] for s in sellers])
    use_habit = np.array([_unit_hash(i, "habit") < 0.30 for i in ids])
    by_rule = np.where(mode == "Express", CARRIERS[0],
                       np.where(mode == "Economy", CARRIERS[2],
                                np.where(model == DELIVERY_MODELS[2], CARRIERS[0], CARRIERS[1])))
    carrier = np.where(use_habit, habitual, by_rule)

    return pd.DataFrame({
        "sim_delivery_model": model,
        "sim_hub_transfers": transfers.astype(int),
        "sim_shipping_mode": mode,
        "sim_carrier": carrier,
        "sim_first_mile_hub": s_reg.map(HUB_OF_REGION).values,
        "sim_last_mile_hub": c_reg.map(HUB_OF_REGION).values,
        "sim_route_km": (dist * 1.25).round(0).values,
    }, index=df.index)


def sensitivity_test(ready, sim, feature_columns, target, xgb_params, split):
    """Does adding the simulated fields to the model improve it?
    Same split, same XGBoost settings, with and without. Returns AUCs."""
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier
    from modeling import NUMERIC, CATEGORICAL

    data = pd.concat([ready.reset_index(drop=True), sim.reset_index(drop=True)], axis=1)
    train, test = split(data)
    out = {}
    for name, extra_num, extra_cat in [("without simulated fields", [], []),
                                       ("with simulated fields", ["sim_hub_transfers", "sim_route_km"],
                                        ["sim_delivery_model", "sim_shipping_mode", "sim_carrier"])]:
        num, cat = NUMERIC + extra_num, CATEGORICAL + extra_cat
        pre = ColumnTransformer([("num", StandardScaler(), num),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
        pipe = Pipeline([("pre", pre), ("clf", XGBClassifier(**xgb_params))])
        cols = num + cat
        pipe.fit(train[cols], train[target])
        out[name] = round(float(roc_auc_score(test[target], pipe.predict_proba(test[cols])[:, 1])), 4)
    out["lift"] = round(out["with simulated fields"] - out["without simulated fields"], 4)
    return out
