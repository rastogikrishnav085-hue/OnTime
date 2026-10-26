"""Stage 2 (after carrier handoff): metrics are real, the endpoint validates input, and it matches training."""
import json
import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app import handoff as hand

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    main.set_db_path(str(tmp_path_factory.mktemp("db") / "board.db"))
    return TestClient(main.app)


@pytest.fixture(scope="module")
def order(client):
    r = main.get_engine().queue.iloc[0]
    return r


def test_saved_metrics_show_a_real_but_honest_gain():
    m = json.load(open(os.path.join(ROOT, "models", "handoff_meta.json")))
    assert m["random_split"]["roc_auc"] > m["checkout_same_orders"]["roc_auc"] + 0.03
    bt = m["monthly_backtest"]
    assert bt["handoff"]["mean_auc"] > bt["checkout"]["mean_auc"] + 0.05
    assert bt["handoff"]["mean_auc"] < 0.9          # the strict figure must not be dressed up
    assert not {"delivered_at", "review_score", "order_delivered_customer_date"} & set(m["feature_columns"])


def test_model_info_endpoint(client):
    j = client.get("/api/handoff/model").json()
    assert j["available"] and j["random_split"]["roc_auc"] > 0.8


def test_score_after_handoff(client, order):
    when = (pd.Timestamp(order["purchased_at"]) + pd.Timedelta(days=2)).isoformat()
    r = client.post("/api/score/handoff", json={"order_id": order["order_id"], "handed_over_at": when})
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["probability"] <= 1 and j["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert j["checkout_probability"] == pytest.approx(float(order["probability"]))
    assert j["change"] == pytest.approx(j["probability"] - j["checkout_probability"])


def test_a_later_handoff_is_never_less_risky_for_the_same_order(client, order):
    t0 = pd.Timestamp(order["purchased_at"])
    p = lambda days: client.post("/api/score/handoff", json={"order_id": order["order_id"],        # noqa: E731
                                                              "handed_over_at": (t0 + pd.Timedelta(days=days)).isoformat()}).json()["probability"]
    assert p(14) >= p(1) - 0.02          # less of the promise left cannot make the order safer


def test_rejects_impossible_or_unknown_input(client, order):
    early = (pd.Timestamp(order["purchased_at"]) - pd.Timedelta(days=1)).isoformat()
    assert client.post("/api/score/handoff", json={"order_id": order["order_id"], "handed_over_at": early}).status_code == 400
    assert client.post("/api/score/handoff", json={"order_id": order["order_id"], "handed_over_at": "not a date"}).status_code == 400
    assert client.post("/api/score/handoff", json={"order_id": "nope", "handed_over_at": "2018-01-01T10:00"}).status_code == 404
    far = (pd.Timestamp(order["purchased_at"]) + pd.Timedelta(days=400)).isoformat()
    assert client.post("/api/score/handoff", json={"order_id": order["order_id"], "handed_over_at": far}).status_code == 400
