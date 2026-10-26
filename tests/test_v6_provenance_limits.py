"""v6: provenance on every score, upload size caps, and request-field validation."""
import io
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "backend", "app_data", "queue.csv")), reason="app data not built")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    sys.path.insert(0, ROOT)
    from fastapi.testclient import TestClient
    from backend.app import adapt_service, assistant, main
    tmp = tmp_path_factory.mktemp("v6")
    main.set_db_path(str(tmp / "board.db"))
    adapt_service.ADAPT_DIR = str(tmp / "adapted")
    assistant.SETTINGS_PATH = str(tmp / "settings.json")
    return TestClient(main.app)


def test_single_score_carries_provenance(client):
    for key, synthetic in (("xgboost", False), ("synthetic_operational", True)):
        r = client.post("/api/score", json={"fields": {}, "model": key})
        assert r.status_code == 200, r.text
        p = r.json()["provenance"]
        assert p["model"] == key and p["synthetic"] is synthetic
        assert p["stage"] and p["data_origin"] and p["score_meaning"]
    assert "GENERATED" in client.post("/api/score", json={"fields": {}, "model": "synthetic_operational"}).json()["provenance"]["data_origin"]


def test_batch_score_carries_provenance(client):
    csv = client.get("/api/template.csv").text
    r = client.post("/api/score/batch?model=synthetic_operational", files={"file": ("t.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["provenance"]["synthetic"] is True


def test_oversized_batch_upload_is_refused(client, monkeypatch):
    from backend.app import main
    monkeypatch.setattr(main, "MAX_BATCH_BYTES", 1024)
    r = client.post("/api/score/batch", files={"file": ("big.csv", "a,b\n" + "1,2\n" * 2000, "text/csv")})
    assert r.status_code == 413


def test_oversized_adapt_upload_is_refused(client, monkeypatch):
    from backend.app import main
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1024)
    r = client.post("/api/adapt/preview", files={"file": ("big.csv", "a,b\n" + "1,2\n" * 2000, "text/csv")})
    assert r.status_code == 413


def test_score_rejects_nested_or_excessive_fields(client):
    assert client.post("/api/score", json={"fields": {"price": {"x": 1}}}).status_code == 422
    assert client.post("/api/score", json={"fields": {f"f{i}": 1 for i in range(300)}}).status_code == 422


def test_handoff_and_tracking_validate_inputs(client):
    assert client.post("/api/score/handoff", json={"order_id": "", "handed_over_at": "2018-04-02T14:30"}).status_code == 422
    assert client.post("/api/score/handoff", json={"order_id": "x", "handed_over_at": "x" * 500}).status_code == 422
    assert client.post("/api/score/tracking", json={"order_id": "x" * 500}).status_code == 422


def test_board_rejects_unknown_stage_and_survives_restart(client, tmp_path):
    from backend.app import main
    oid = str(main.get_engine().queue.order_id.iloc[0])
    assert client.post("/api/board", json={"order_id": oid, "owner": "A", "note": "n"}).status_code in (201, 409)
    assert client.patch(f"/api/board/{oid}", json={"stage": "not-a-stage"}).status_code == 400
    # a fresh store pointed at the same file still has the item and its activity log
    path = main.board().path if hasattr(main.board(), "path") else None
    if path:
        from backend.app.store import BoardStore
        again = BoardStore(path)
        assert again.get(oid) is not None
        assert again.activity(10)
