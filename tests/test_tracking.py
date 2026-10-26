"""Third score (in transit, SIMULATED tracking data): honest labelling, validation, no leakage, serve = train."""
import json
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import main  # noqa: F401  (puts src/ on the path)
from backend.app import tracking as trk
from tracking_features import TIME_FEATURES, TRACK_FEATURES, WEATHER_FEATURES, build_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    main.set_db_path(str(tmp_path_factory.mktemp("db") / "board.db"))
    return TestClient(main.app)


@pytest.fixture(scope="module")
def order(client):
    """A queue order that has scans and is not delivered within its first 3 days."""
    q = main.get_engine().queue
    ev = trk._load()["events"]
    ids = set(ev["order_id"])
    q = q[q.order_id.isin(ids)]
    for _, r in q.iterrows():
        if (pd.Timestamp(r["delivered_at"]) - pd.Timestamp(r["purchased_at"])).days > 10 and \
           (pd.Timestamp(r["promised_by"]) - pd.Timestamp(r["purchased_at"])).days > 10:
            return r
    pytest.skip("no suitable order")


def test_saved_metadata_is_labelled_simulated_and_real_baseline_is_kept():
    m = json.load(open(os.path.join(ROOT, "models", "tracking_meta.json")))
    assert m["simulated"] is True and "SIMULATED" in m["note"]
    assert m["with_tracking"]["roc_auc"] > m["without_tracking"]["roc_auc"]
    assert "DELIVERED" not in " ".join(m["feature_columns"]).upper()


def test_model_endpoint_says_simulated(client):
    j = client.get("/api/tracking/model").json()
    assert j["available"] and j["simulated"] is True


def test_timeline_never_leaks_the_delivery_event(client, order):
    r = client.get(f"/api/tracking/{order['order_id']}?days_after_purchase=60")
    assert r.status_code == 200
    types = {e["type"] for e in r.json()["events"]}
    assert "DELIVERED" not in types and r.json()["simulated"] is True


def test_events_after_as_of_are_ignored(order):
    c = trk._load()
    o = trk._frame(order)
    early = pd.Timestamp(order["purchased_at"]) + pd.Timedelta(days=1)
    late = pd.Timestamp(order["purchased_at"]) + pd.Timedelta(days=8)
    fe = build_features(o, c["events"], c["ctx"], [early]).iloc[0]
    fl = build_features(o, c["events"], c["ctx"], [late]).iloc[0]
    assert fl["n_scans"] >= fe["n_scans"] and fl["days_since_purchase"] > fe["days_since_purchase"]
    assert list(build_features(o, c["events"], c["ctx"], [late]).columns) == TIME_FEATURES + TRACK_FEATURES + WEATHER_FEATURES


def test_score_returns_probability_and_checkout_comparison(client, order):
    r = client.post("/api/score/tracking", json={"order_id": order["order_id"], "days_after_purchase": 5})
    assert r.status_code == 200, r.text
    j = r.json()
    assert 0 <= j["probability"] <= 1 and j["simulated"] is True
    assert j["checkout_probability"] == pytest.approx(float(order["probability"]))
    assert j["change"] == pytest.approx(j["probability"] - j["checkout_probability"])


def test_score_accepts_an_explicit_time(client, order):
    at = (pd.Timestamp(order["purchased_at"]) + pd.Timedelta(days=5)).isoformat()
    a = client.post("/api/score/tracking", json={"order_id": order["order_id"], "as_of": at}).json()
    b = client.post("/api/score/tracking", json={"order_id": order["order_id"], "days_after_purchase": 5}).json()
    assert a["probability"] == pytest.approx(b["probability"])


@pytest.mark.parametrize("body,code", [
    ({"as_of": "not a date"}, 400),
    ({"days_after_purchase": -1}, 400),
    ({"days_after_purchase": 500}, 400),
    ({}, 400),
    ({"days_after_purchase": 90}, 400),          # past delivery or promise: nothing left to forecast
])
def test_bad_times_are_rejected(client, order, body, code):
    r = client.post("/api/score/tracking", json={"order_id": order["order_id"], **body})
    assert r.status_code == code and r.json()["detail"]


def test_unknown_order_is_404(client):
    assert client.post("/api/score/tracking", json={"order_id": "nope", "days_after_purchase": 3}).status_code == 404
    assert client.get("/api/tracking/nope?days_after_purchase=3").status_code == 404


def test_serving_matches_the_training_pipeline_on_a_sample(order):
    """Same builder, same model: scoring a queue order twice through different paths gives the same number."""
    c = trk._load()
    t = pd.Timestamp(order["purchased_at"]) + pd.Timedelta(days=4)
    p1 = trk.score(order, t)["probability"]
    f = build_features(trk._frame(order), c["events"], c["ctx"], [t])
    X = pd.concat([pd.DataFrame([{k: order[k] for k in c["meta"]["feature_columns"] if k not in f.columns}]), f.reset_index(drop=True)], axis=1)
    for col, cats in c["meta"]["categorical"].items():
        X[col] = pd.Categorical(X[col].astype(str), categories=cats)
    p2 = float(c["model"].predict_proba(X[c["meta"]["feature_columns"]])[:, 1][0])
    assert p1 == pytest.approx(p2, abs=1e-6)
