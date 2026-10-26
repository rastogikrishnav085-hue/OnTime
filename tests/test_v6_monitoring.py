import os, sys
import numpy as np, pandas as pd, pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from backend.app import monitoring as mon  # noqa: E402


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, .1, 3000); p = np.clip(.1 + .15 * y + rng.normal(0, .1, 3000), 0, 1)
    r = mon.bootstrap_ci(y, p, n_boot=100)
    for m in r["metrics"].values():
        assert m["low"] <= m["estimate"] <= m["high"]
    assert mon.bootstrap_ci(np.zeros(100), np.random.rand(100)) is None


def test_calibration_and_psi():
    rng = np.random.default_rng(1)
    p = rng.random(5000); y = (rng.random(5000) < p).astype(int)
    c = mon.calibration(y, p)
    assert c["ece"] < 0.05 and c["brier"] < c["brier_baseline"]
    a = rng.normal(0, 1, 2000)
    assert mon.psi(a, rng.normal(0, 1, 2000)) < 0.1
    assert mon.psi(a, rng.normal(2, 1, 2000)) > 0.25
    assert mon.psi([1, 2], [1, 2]) is None


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    if not os.path.exists(os.path.join(ROOT, "backend", "app_data", "queue.csv")):
        pytest.skip("app data not built")
    from fastapi.testclient import TestClient
    from backend.app import adapt_service, assistant, main
    tmp = tmp_path_factory.mktemp("mon")
    main.set_db_path(str(tmp / "b.db")); adapt_service.ADAPT_DIR = str(tmp / "a"); assistant.SETTINGS_PATH = str(tmp / "s.json")
    return TestClient(main.app)


def test_summary_endpoint(client):
    r = client.get("/api/monitoring/summary?bootstrap=60")
    assert r.status_code == 200, r.text
    j = r.json()
    ci = j["confidence_intervals"]["metrics"]["roc_auc"]
    assert ci["low"] < ci["estimate"] < ci["high"] and 0.65 < ci["estimate"] < 0.78
    assert j["calibration"]["served_score"]["bins"] and j["calibration"]["calibrated_ensemble"]["bins"] and j["monthly_stability"]["months"] and j["drift_early_vs_late"]["rows"]
    assert j["provenance"]["synthetic"] is False


def test_outcomes_endpoint(client):
    from backend.app import main
    q = main.get_engine().queue.head(500)
    csv = q[["order_id", "is_delayed"]].to_csv(index=False)
    r = client.post("/api/monitoring/outcomes", files={"file": ("o.csv", csv, "text/csv")})
    assert r.status_code == 200 and r.json()["matched_to_scored_orders"] == 500
    assert client.post("/api/monitoring/outcomes", files={"file": ("o.csv", "a,b\n1,2\n", "text/csv")}).status_code == 400


def test_drift_endpoint(client):
    from backend.app import main
    csv = main.get_engine().queue.head(300).to_csv(index=False)
    r = client.post("/api/monitoring/drift", files={"file": ("d.csv", csv, "text/csv")})
    assert r.status_code == 200 and r.json()["rows"]


def test_batch_returns_row_errors(client):
    import io
    df = pd.read_csv(io.StringIO(client.get("/api/template.csv").text))
    df.loc[0, "total_price"] = -50          # impossible value
    df.loc[1, "promised_delivery_days"] = -3
    r = client.post("/api/score/batch?model=xgboost", files={"file": ("b.csv", df.to_csv(index=False), "text/csv")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["summary"]["rejected"] >= 1
    assert j["row_errors"] and j["row_errors"][0]["reasons"] and j["error_reasons"]
    # a file with the wrong columns is refused with a message naming what is missing
    bad = client.post("/api/score/batch?model=xgboost", files={"file": ("b.csv", "order_id,price\nA,1\n", "text/csv")})
    assert bad.status_code == 400 and "missing" in bad.json()["detail"].lower()
