"""The web app's API, end to end, against the real model and queue.
The board uses a throwaway SQLite file so the tests never touch real data."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "backend", "app_data", "queue.csv")),
    reason="app data not built (run backend/prepare_app_data.py)",
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import sys
    sys.path.insert(0, ROOT)
    from fastapi.testclient import TestClient
    from backend.app import main
    main.set_db_path(str(tmp_path_factory.mktemp("db") / "board.db"))
    return TestClient(main.app)


@pytest.fixture(scope="module")
def critical_ids(client):
    r = client.get("/api/orders", params={"band": "CRITICAL", "limit": 5}).json()
    return [o["order_id"] for o in r["items"]]


def test_health_and_meta(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["orders_in_queue"] > 10000 and 0 < h["threshold"] < 1
    m = client.get("/api/meta").json()
    assert "SP" in m["states"] and m["typical_order"] and len(m["stages"]) == 4


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "OnTime" in r.text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/vendor/chart.umd.js").status_code == 200


def test_summary_numbers_are_consistent(client):
    s = client.get("/api/summary").json()
    assert sum(s["bands"].values()) == s["n_queue"]
    assert s["n_flagged"] <= s["n_queue"]
    assert len(s["needs_attention"]) <= 8
    rates = [b["late_rate"] for b in s["risk"]["bands"]]
    assert rates == sorted(rates), "late rate should rise from LOW to CRITICAL"


def test_queue_filters_sort_and_paging(client):
    r = client.get("/api/orders", params={"band": "HIGH,CRITICAL", "limit": 20}).json()
    assert r["total"] > 0 and len(r["items"]) == 20
    assert all(o["risk_level"] in ("HIGH", "CRITICAL") for o in r["items"])
    probs = [o["probability"] for o in r["items"]]
    assert probs == sorted(probs, reverse=True)
    p2 = client.get("/api/orders", params={"band": "HIGH,CRITICAL", "limit": 20, "offset": 20}).json()
    assert not {o["order_id"] for o in r["items"]} & {o["order_id"] for o in p2["items"]}
    sp = client.get("/api/orders", params={"state": "AL", "limit": 50}).json()
    assert all(o["customer_state"] == "AL" for o in sp["items"])
    assert client.get("/api/orders", params={"limit": 999}).status_code == 422


def test_order_detail_has_reasons_action_and_outcome(client, critical_ids):
    d = client.get(f"/api/orders/{critical_ids[0]}").json()
    assert d["risk_level"] == "CRITICAL" and d["suggested_action"].startswith("URGENT")
    assert d["reasons"] and all(r["contribution"] > 0 for r in d["reasons"])
    assert set(d["outcome"]) == {"was_late", "delivered_at", "days_vs_promise"}
    assert client.get("/api/orders/not-a-real-order").status_code == 404


def test_score_one_partial_input_and_derivations(client):
    r = client.post("/api/score", json={"fields": {"customer_state": "AL", "seller_state": "SP",
                                                   "seller_customer_distance_km": 2100, "promised_delivery_days": 8,
                                                   "purchase_month": 3, "total_price": 200, "total_freight": 40, "n_items": 2}, "model": "enriched"}).json()
    assert r["status"] == "SCORED" and 0 <= r["probability"] <= 1
    assert "same_state" in r["derived"] and "total_order_value" in r["derived"]
    assert "customer_state" not in r["filled_from_typical"]
    assert r["reasons"], "the simulator explains every scored order"


def test_score_one_rejects_impossible_input(client):
    r = client.post("/api/score", json={"fields": {"total_price": -10}, "model": "enriched"}).json()
    assert r["status"] == "REJECTED" and r["reject_reasons"] and r["probability"] is None
    r = client.post("/api/score", json={"fields": {"seller_customer_distance_km": "far"}, "model": "enriched"}).json()
    assert r["status"] == "REJECTED"


def test_riskier_inputs_score_higher(client):
    calm = client.post("/api/score", json={"fields": {"customer_state": "SP", "seller_state": "SP",
                                                      "seller_customer_distance_km": 15, "promised_delivery_days": 25, "purchase_month": 6}}).json()
    rough = client.post("/api/score", json={"fields": {"customer_state": "AL", "seller_state": "SP",
                                                       "seller_customer_distance_km": 2100, "promised_delivery_days": 6, "purchase_month": 3}}).json()
    assert rough["probability"] > calm["probability"]


def test_batch_scoring_roundtrip(client):
    tpl = client.get("/api/template.csv")
    assert tpl.status_code == 200 and tpl.text.startswith("order_id,seller_id")
    r = client.post("/api/score/batch", files={"file": ("t.csv", tpl.text, "text/csv")}).json()
    assert r["summary"]["total"] == 10 == len(r["rows"])
    assert r["summary"]["scored"] + r["summary"]["rejected"] == 10
    assert r["summary"]["flagged"] >= 1, "template includes some alerts"


def test_batch_rejects_bad_files(client):
    assert client.post("/api/score/batch", files={"file": ("e.csv", "", "text/csv")}).status_code == 400
    bad = client.post("/api/score/batch", files={"file": ("x.csv", "a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 400 and "missing" in bad.json()["detail"]


def test_board_lifecycle(client, critical_ids):
    oid = critical_ids[0]
    assert client.post("/api/board", json={"order_id": oid}).status_code == 201
    assert client.post("/api/board", json={"order_id": oid}).status_code == 409
    assert client.post("/api/board", json={"order_id": "nope"}).status_code == 404
    r = client.patch(f"/api/board/{oid}", json={"stage": "investigating", "owner": "Avash", "note": "Seller called"}).json()
    assert r["stage"] == "investigating" and r["owner"] == "Avash"
    assert client.patch(f"/api/board/{oid}", json={"stage": "flying"}).status_code == 400
    b = client.get("/api/board").json()
    assert b["counts"]["investigating"] == 1 and b["items"][0]["order"]["order_id"] == oid
    acts = client.get("/api/board/activity").json()["items"]
    assert any("Investigating" in a["detail"] for a in acts)
    s = client.get("/api/summary").json()
    assert oid not in [o["order_id"] for o in s["needs_attention"]]
    assert client.delete(f"/api/board/{oid}").status_code == 200
    assert client.delete(f"/api/board/{oid}").status_code == 404


def test_board_autofill_adds_only_new_critical(client):
    first = client.post("/api/board/autofill", params={"n": 3}).json()["added"]
    second = client.post("/api/board/autofill", params={"n": 3}).json()["added"]
    items = client.get("/api/board").json()["items"]
    assert first == 3 and second == 3 and len({i["order_id"] for i in items}) == len(items)
    assert all(i["order"]["risk_level"] == "CRITICAL" for i in items)


def test_insights_and_model(client):
    i = client.get("/api/insights").json()
    assert i["state"] and i["seller_watchlist"] and abs(i["base_rate"] - 8.1) < 0.2
    m = client.get("/api/model").json()
    assert m["live_scoring"]["key"] in ("synthetic_operational", "xgboost", "ensemble", "enriched")
    assert m["live_scoring"]["available"] is True
    assert m["serving_queue"] in ("xgboost", "ensemble", "enriched") and m["historical_queue_available"]


def test_seller_endpoint(client, critical_ids):
    d = client.get(f"/api/orders/{critical_ids[1]}").json()
    s = client.get(f"/api/sellers/{d['seller_id']}").json()
    assert s["seller_id"] == d["seller_id"] and s["n_in_queue"] >= 1
    assert client.get("/api/sellers/unknown-seller").status_code == 404


def test_batch_accepts_excel_style_csv(client):
    """Excel on Windows adds a byte-order mark; some locales use semicolons."""
    tpl = client.get("/api/template.csv").text
    bom = ("﻿" + tpl).encode("utf-8")
    r = client.post("/api/score/batch", files={"file": ("excel.csv", bom, "text/csv")}).json()
    assert r["summary"]["total"] == 10 and r["summary"]["rejected"] == 0
    semi = tpl.replace(",", ";").encode("cp1252")
    r = client.post("/api/score/batch", files={"file": ("semi.csv", semi, "text/csv")}).json()
    assert r["summary"]["total"] == 10 and r["summary"]["rejected"] == 0


# ---------------- v2 features ----------------
def test_patterns_and_detail(client):
    d = client.get("/api/patterns").json()
    assert d["k"] == len(d["patterns"]) >= 4 and sum(p["share_of_late"] for p in d["patterns"]) == pytest.approx(100, abs=1)
    p = client.get("/api/patterns/1").json()
    assert p["story"] and p["recommendations"] and "orders" in p
    assert client.get("/api/patterns/99").status_code == 404


def test_queue_filters_by_pattern_and_focus_and_priority(client):
    r = client.get("/api/orders", params={"pattern": 2, "limit": 30}).json()
    assert r["total"] > 0 and all(o["pattern"] == 2 for o in r["items"])
    r = client.get("/api/orders", params={"focus": True, "sort": "priority", "limit": 30}).json()
    assert all(o["focus_tier"] for o in r["items"])
    pr = [o["priority"] for o in r["items"]]
    assert pr == sorted(pr, reverse=True)


def test_order_story_and_customer_message(client, critical_ids):
    st = client.get(f"/api/orders/{critical_ids[0]}/story").json()
    assert len(st["paragraphs"]) >= 4 and "Hello" in st["customer_message"]
    o = client.get(f"/api/orders/{critical_ids[0]}").json()
    assert o["simulated"]["delivery_model"] and o["pattern_name"]


def test_focus_categories_network_briefing_planner(client):
    c = client.get("/api/categories").json()
    assert len(c["focus"]) == 6 and c["roadmap_note"]
    assert client.get("/api/categories/baby").json()["story"]
    assert client.get("/api/categories/toys").status_code == 404
    n = client.get("/api/network").json()
    assert abs(n["sensitivity"]["lift"]) < 0.02 and n["delivery_model"]
    ps = client.get("/api/briefing/periods").json()["periods"]
    b = client.get("/api/briefing", params={"period": ps[0]["key"]}).json()
    assert b["story"] and b["actions"] and b["n_orders"] > 0
    assert client.get("/api/briefing", params={"period": "1999-01"}).status_code == 404
    pl = client.get("/api/planner", params={"cost_check": 15, "cost_late": 120, "effectiveness": 0.5}).json()
    assert pl["best"]["net_saving"] >= pl["current"]["net_saving"]
    free = client.get("/api/planner", params={"cost_check": 0, "cost_late": 100}).json()
    assert free["best"]["threshold"] == 0.05, "checking for free: alert on everything"


def test_score_includes_pattern_and_simulated_view(client):
    r = client.post("/api/score", json={"fields": {"customer_state": "BA", "seller_state": "SP", "seller_customer_distance_km": 1500}, "model": "enriched"}).json()
    assert r["pattern"]["rank"] >= 1 and r["simulated"]["delivery_model"]


def test_assistant_endpoints(client, tmp_path, monkeypatch):
    from backend.app import assistant
    monkeypatch.setattr(assistant, "SETTINGS_PATH", str(tmp_path / "s.json"))
    st = client.get("/api/assistant/status").json()
    assert st["engines"]["builtin"]["available"] and st["examples"]
    r = client.post("/api/assistant/ask", json={"question": "What is pattern 1?", "engine": "builtin"}).json()
    assert r["topic"] == "pattern" and r["engine"] == "builtin"
    assert client.post("/api/assistant/ask", json={"question": "x", "engine": "nope"}).status_code == 400
    assert client.post("/api/settings", json={"engine": "nope"}).status_code == 400
    s = client.post("/api/settings", json={"timeout_s": 300}).json()
    assert s["timeout_s"] == 120
