"""Simulated logistics, delay patterns (clustering), narratives and focus categories."""
import numpy as np
import pandas as pd
import pytest


def _orders(n=400, seed=0):
    rng = np.random.RandomState(seed)
    states = np.array(["SP", "RJ", "BA", "AL", "PR", "AM"])
    cs = states[rng.randint(0, len(states), n)]
    ss = np.where(rng.rand(n) < 0.7, "SP", states[rng.randint(0, len(states), n)])
    return pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)], "seller_id": [f"s{i % 25}" for i in range(n)],
        "customer_state": cs, "seller_state": ss,
        "seller_customer_distance_km": np.where(cs == ss, rng.uniform(5, 300, n), rng.uniform(300, 3000, n)),
        "promised_delivery_days": rng.uniform(3, 45, n), "total_freight": rng.uniform(5, 80, n),
        "total_weight_g": rng.uniform(100, 20000, n), "total_order_value": rng.uniform(20, 900, n),
        "purchase_month": rng.randint(1, 13, n), "seller_historical_delay_rate": rng.beta(1, 10, n),
        "approval_delay_hours": rng.exponential(5, n), "product_category": rng.choice(["baby", "toys", "auto"], n),
        "customer_city": rng.choice(["sao paulo", "salvador"], n),
    }).assign(same_state=lambda d: (d.customer_state == d.seller_state).astype(int))


# ---------------- simulation ----------------
def test_simulation_rules_and_labels():
    from simulation import simulate_logistics, DELIVERY_MODELS, SHIPPING_MODES, CARRIERS
    d = _orders()
    sim = simulate_logistics(d)
    assert len(sim) == len(d) and list(sim.index) == list(d.index)
    assert set(sim.sim_delivery_model) <= set(DELIVERY_MODELS)
    assert set(sim.sim_shipping_mode) <= set(SHIPPING_MODES) and set(sim.sim_carrier) <= set(CARRIERS)
    direct = (d.customer_state == d.seller_state) & (d.seller_customer_distance_km < 150)
    assert (sim.loc[direct, "sim_delivery_model"] == DELIVERY_MODELS[0]).all()
    assert (sim.loc[direct, "sim_hub_transfers"] == 0).all()
    longhaul = d.customer_state.isin(["BA", "AL", "AM"]) & (d.seller_state == "SP")
    assert (sim.loc[longhaul, "sim_hub_transfers"] == 3).all()


def test_simulation_is_reproducible():
    from simulation import simulate_logistics
    d = _orders()
    pd.testing.assert_frame_equal(simulate_logistics(d), simulate_logistics(d.copy()))


# ---------------- clustering ----------------
@pytest.fixture(scope="module")
def clustered():
    from simulation import simulate_logistics, region
    from clustering import fit_clusters, profile_clusters
    d = _orders(600)
    d["customer_region"] = d.customer_state.map(region)
    d["actual_days"] = d.promised_delivery_days + 5
    d["delay_days"] = 5.0
    d = pd.concat([d, simulate_logistics(d)], axis=1)
    labels, params = fit_clusters(d, k_range=range(3, 5))
    return d, labels, params, profile_clusters(d, labels, params)


def test_clusters_are_profiled_with_names_and_recommendations(clustered):
    d, labels, params, profiles = clustered
    assert 3 <= params["k"] <= 4 and len(profiles) == params["k"]
    assert sum(p["n"] for p in profiles) == len(d)
    assert [p["rank"] for p in profiles] == list(range(1, params["k"] + 1))
    assert len({p["name"] for p in profiles}) == len(profiles), "names must be unique"
    for p in profiles:
        assert p["recommendations"] and all(isinstance(r, str) and len(r) > 20 for r in p["recommendations"])


def test_assign_maps_new_orders_to_a_known_pattern(clustered):
    from clustering import assign
    d, labels, params, _ = clustered
    got = assign(d.head(50), params)
    assert set(got) <= set(range(params["k"]))
    assert (got == labels[:50]).mean() > 0.95


def test_lift_ignores_states_that_are_merely_big():
    """The 'concentrated in RJ' bug: a state that is big everywhere must not be called concentrated."""
    from clustering import _lift_table
    allv = pd.Series(["RJ"] * 50 + ["SP"] * 40 + ["BA"] * 10)
    cluster = pd.Series(["RJ"] * 25 + ["SP"] * 15 + ["BA"] * 10)
    rows = _lift_table(cluster, allv)
    assert rows[0]["key"] == "BA" and rows[0]["lift"] == 2.0


@pytest.mark.parametrize("feature", ["distance_km", "promised_days", "seller_late_rate", "peak_season", "weight_kg",
                                     "order_value", "freight_share", "same_state", "approval_hours"])
@pytest.mark.parametrize("sign", [1, -1])
def test_every_feature_has_its_own_recommendation(feature, sign):
    """No distinguishing feature may silently fall through to the generic message
    (a real bug once: total order value had no branch)."""
    from clustering import recommendations
    z = {f: 0.0 for f in ["distance_km", "promised_days", "seller_late_rate", "peak_season", "weight_kg", "order_value",
                          "freight_share", "same_state", "approval_hours"]}
    z[feature] = 1.2 * sign
    p = {"z": z, "distinguishing": [{"feature": feature, "label": feature, "z": z[feature]}], "regions": [], "states": [],
         "categories": [], "median_distance_km": 900, "median_promised_days": 10, "median_actual_days": 16, "p75_actual_days": 21,
         "median_days_late": 6, "median_seller_late_rate": 30, "top_sellers": {"top10_share": 40}, "peak_share": 80,
         "median_weight_kg": 12, "median_order_value": 400, "same_state_share": 90, "top_city": "Sao Paulo",
         "median_approval_hours": 30}
    recs = recommendations(p)
    assert recs and not recs[0].startswith("No single factor")


# ---------------- narratives & categories ----------------
def test_order_story_uses_the_orders_numbers():
    from narrative import order_story, customer_message
    o = {"order_id": "abc", "product_category": "baby", "seller_state": "SP", "customer_state": "BA", "seller_city": "Sao Paulo",
         "customer_city": "Salvador", "purchased_at": "2018-03-01 10:00:00", "promised_by": "2018-03-10", "probability": 0.91,
         "risk_level": "CRITICAL", "suggested_action": "URGENT: Book an express carrier.", "flagged": True,
         "reasons": [{"label": "Long shipping distance", "contribution": 0.8}],
         "features": {"total_order_value": 120.5, "seller_customer_distance_km": 1950, "promised_delivery_days": 9,
                      "seller_historical_delay_rate": 0.2, "customer_state_historical_delay_rate": 0.14, "purchase_month": 3},
         "outcome": {"was_late": True, "days_vs_promise": 4.2}}
    text = " ".join(order_story(o))
    for must in ("91%", "1,950 km", "Salvador", "long shipping distance", "Suggested action (urgent)", "4 days after"):
        assert must in text, must
    msg = customer_message(o)
    assert "Salvador" in msg and "91%" not in msg and "seller" in msg.lower()


def test_focus_categories_are_six_with_playbooks():
    from categories import FOCUS_CATEGORIES, weight_of, tier_of
    assert len(FOCUS_CATEGORIES) == 6
    assert all(len(c["playbook"]) >= 3 and c["impact"] for c in FOCUS_CATEGORIES.values())
    assert weight_of("baby") > weight_of("toys") == 1.0 and tier_of("toys") == ""
