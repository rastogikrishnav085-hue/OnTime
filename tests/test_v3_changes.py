"""Tests for the v3 changes: board stage guide, threshold table, auto assistant engine, trimmed pages."""
import os
import pytest
from fastapi.testclient import TestClient

from backend.app import main, assistant as ai

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    main.set_db_path(str(tmp_path_factory.mktemp("db") / "board.db"))
    return TestClient(main.app)


def test_board_stages_explain_themselves(client):
    st = client.get("/api/board").json()["stages"]
    assert [s["key"] for s in st] == ["to_review", "investigating", "customer_informed", "resolved"]
    for s in st:
        assert s["meaning"] and s["do"]
    assert all(s["next_label"] for s in st[:3]) and st[3]["next_label"] == ""


def test_board_walks_through_every_stage_and_sorts_by_risk(client):
    assert client.post("/api/board/autofill", params={"n": 3}).json()["added"] == 3
    items = client.get("/api/board").json()["items"]
    probs = [i["order"]["probability"] for i in items]
    assert probs == sorted(probs, reverse=True)
    oid = items[0]["order_id"]
    for stage in ("investigating", "customer_informed", "resolved", "to_review"):
        assert client.patch(f"/api/board/{oid}", json={"stage": stage}).json()["stage"] == stage


def test_threshold_table(client):
    r = client.get("/api/model/thresholds").json()
    rows = r["rows"]
    assert len(rows) >= 5
    recalls = [x["recall"] for x in rows]
    assert recalls == sorted(recalls)                      # alerting more orders never lowers recall
    assert all(0 <= x["precision"] <= 1 for x in rows)


def test_auto_engine_is_known_and_reports_status():
    assert "auto" in ai.ENGINES
    s = dict(ai.DEFAULTS, gemini_api_key="", ollama_url="http://127.0.0.1:9")
    st = ai.engine_status(s)
    assert st["auto"]["available"] and ai._auto_engine(s) == "builtin"


def test_auto_engine_answers_even_when_no_ai_engine_is_running(client):
    r = client.post("/api/assistant/ask", json={"question": "How reliable is the model?", "engine": "auto"})
    assert r.status_code == 200 and r.json()["paragraphs"]


def test_insights_and_planner_are_in_the_navigation():
    html = open(os.path.join(ROOT, "frontend", "index.html"), encoding="utf-8").read()
    assert "#/insights" in html and "#/planner" in html
    assert "#/board" in html and "#/assistant" in html


def test_template_values_from_env_example_are_not_treated_as_real_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    monkeypatch.setenv("SUPABASE_URL", "https://yourproject.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "your-supabase-anon-or-service-key")
    s = ai.load_settings(str(tmp_path / "none.json"))
    assert s["gemini_api_key"] == "" and s["supabase_url"] == "" and s["supabase_key"] == ""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyRealLookingKey1234567890abcdef")
    assert ai.load_settings(str(tmp_path / "none.json"))["gemini_api_key"].startswith("AIza")
