"""Compact end-to-end smoke test through FastAPI's actual routing and scoring stack."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "backend", "app_data", "queue.csv")),
    reason="prepared application data is missing",
)


@pytest.fixture(scope="module")
def live_client(tmp_path_factory):
    sys.path.insert(0, ROOT)
    from fastapi.testclient import TestClient
    from backend.app import main, assistant
    tmp = tmp_path_factory.mktemp("e2e-smoke")
    main.set_db_path(str(tmp / "board.db"))
    assistant.SETTINGS_PATH = str(tmp / "settings.json")
    return TestClient(main.app)


def test_full_application_smoke_path(live_client):
    home = live_client.get("/")
    assert home.status_code == 200 and "OnTime" in home.text
    assert live_client.get("/assets/app.js").status_code == 200

    health = live_client.get("/api/health")
    assert health.status_code == 200 and health.json()["status"] == "ok"
    diagnostics = live_client.get("/api/health/diagnostics")
    assert diagnostics.status_code == 200
    diag = diagnostics.json()
    assert diag["checks"]["real_world_validation"]["status"] == "not_configured"
    assert diag["checks"]["weather_provider"]["status"] == "not_checked"
    meta = live_client.get("/api/meta")
    assert meta.status_code == 200 and meta.json()["live_scoring_available"] is True
    assert live_client.get("/api/models").status_code == 200
    summary = live_client.get("/api/summary")
    assert summary.status_code == 200 and summary.json()["n_queue"] > 0

    order_page = live_client.get("/api/orders", params={"limit": 1})
    assert order_page.status_code == 200 and order_page.json()["items"]
    order_id = order_page.json()["items"][0]["order_id"]
    assert live_client.get(f"/api/orders/{order_id}").status_code == 200

    score = live_client.post("/api/score", json={
        "model": "synthetic_operational",
        "fields": {"order_id": "smoke-order-1", "route_km": 500, "weather_severity_forecast": 4, "congestion_forecast": 4,
                    "carrier_reliability_prior": 0.55, "capacity_utilization_at_dispatch": 1.1,
                    "remaining_sla_slack_hours": 2, "facility_backlog_at_dispatch": 50,
                    "route_disruption_alert_at_dispatch": 1, "last_scan_age_hours": 12,
                    "estimated_truck_delay_hours_at_dispatch": 10, "dispatch_lag_hours": 12},
    })
    assert score.status_code == 200 and score.json()["status"] == "SCORED"
    assert 0 <= score.json()["probability"] <= 1
    assert score.json()["confidence"] == "SYNTHETIC_BENCHMARK"
    assert score.json()["order_id"] == "smoke-order-1" and score.json()["scored_at"]

    template = live_client.get("/api/template.csv")
    assert template.status_code == 200 and "text/csv" in template.headers["content-type"]
    batch = live_client.post("/api/score/batch?model=synthetic_operational",
                             files={"file": ("smoke.csv", template.content, "text/csv")})
    assert batch.status_code == 200, batch.text
    payload = batch.json()
    assert payload["summary"]["total"] == 10
    assert payload["summary"]["scored"] + payload["summary"]["rejected"] == 10
    assert payload["scored_at"] and all(row["scored_at"] == payload["scored_at"] for row in payload["rows"])

    planner = live_client.get("/api/planner")
    assert planner.status_code == 200 and planner.json()
    assistant = live_client.post("/api/assistant/ask", json={"question": "What does the synthetic dispatch score mean?", "engine": "builtin"})
    assert assistant.status_code == 200 and assistant.json()
