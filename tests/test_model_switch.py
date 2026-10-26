"""Switching the scoring model in Settings must change every queue-based view, and say honestly what the numbers mean."""
import pytest
from fastapi.testclient import TestClient

from backend.app import main


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.ai, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    c = TestClient(main.app)
    yield c
    c.post("/api/settings", json={"scoring_model": "enriched"})   # leave the shared engine on the default view
    main.get_engine().set_view("enriched")


def _pick(c, key):
    assert c.post("/api/settings", json={"scoring_model": key}).status_code == 200


def test_default_view_is_the_chronological_model(client):
    s = client.get("/api/summary").json()
    assert s["n_flagged"] == 4642
    m = client.get("/api/meta").json()
    assert m["queue_model"] == "enriched" and m["queue_view"]["in_sample"] is False


def test_switching_changes_summary_queue_bands_and_thresholds(client):
    base = client.get("/api/summary").json()
    _pick(client, "xgboost")
    x = client.get("/api/summary").json()
    assert x["n_flagged"] != base["n_flagged"]
    assert x["risk"]["n_flagged"] == x["n_flagged"] == main.get_engine().queue.flagged.sum()
    assert client.get("/api/meta").json()["queue_model"] == "xgboost"
    top = client.get("/api/orders?limit=5").json()["items"]
    assert top[0]["probability"] >= top[-1]["probability"]
    th = client.get("/api/model/thresholds").json()
    assert th["model"] == "xgboost" and th["in_sample"] is True and "optimistic" in th["note"]
    _pick(client, "enriched")
    assert client.get("/api/summary").json()["n_flagged"] == base["n_flagged"]


def test_probabilities_come_from_the_selected_model(client):
    _pick(client, "ensemble")
    e = main.get_engine()
    q = e.queue
    assert (q["probability"] == q["p_ensemble"]).all()
    assert (q["flagged"] == (q["p_ensemble"] >= e.models["ensemble"][1])).all()
    assert set(q["risk_level"]) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    o = client.get(f"/api/orders/{q.order_id.iloc[0]}").json()
    assert o["probability"] == pytest.approx(float(q.probability.iloc[0]), abs=1e-3)
    assert o["reasons"] and o["main_driver"]


def test_evaluation_headline_follows_the_model_and_is_honest(client):
    ev = client.get("/api/model/evaluation").json()
    assert ev["active"]["key"] == "enriched" and ev["active"]["in_sample"] is False
    assert ev["active"]["roc_auc"] == pytest.approx(0.717, abs=1e-3)
    _pick(client, "xgboost")
    ev = client.get("/api/model/evaluation").json()
    a = ev["active"]
    assert a["key"] == "xgboost" and a["in_sample"] is True
    assert a["roc_auc"] == pytest.approx(0.804, abs=2e-3)            # its own random-split figure
    assert a["queue_roc_auc"] > 0.9                                  # in-sample, labelled as such
    assert any(r["key"] == "xgboost" and r["serving_queue"] for r in ev["rows"])
    assert client.get("/api/monitoring/summary?bootstrap=50").json()["in_sample"] is True


def test_dispatch_model_cannot_rescore_the_historical_queue(client):
    base = client.get("/api/summary").json()["n_flagged"]
    _pick(client, "synthetic_operational")
    assert client.get("/api/summary").json()["n_flagged"] == base
    ev = client.get("/api/model/evaluation").json()
    assert ev["dispatch_selected"] is True and ev["dispatch"]["roc_auc"] == pytest.approx(0.943, abs=1e-3)
    v = client.get("/api/meta").json()["queue_view"]
    assert v["limited_by"] == "synthetic_operational" and "dispatch" in v["note"].lower()
    r = client.post("/api/score", json={"fields": {}}).json()
    assert r["model"] == "synthetic_operational"


def test_banner_is_gone():
    html = open("frontend/index.html", encoding="utf-8").read()
    assert "Data provenance" not in html and "provenance-banner" not in html
