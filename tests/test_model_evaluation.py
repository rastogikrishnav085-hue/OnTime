"""The AUC and alert numbers the app shows must match what the saved models and the served queue really produce."""
import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import sys
    sys.path.insert(0, ROOT)
    from fastapi.testclient import TestClient
    from backend.app import adapt_service, assistant, main
    tmp = tmp_path_factory.mktemp("eval")
    main.set_db_path(str(tmp / "board.db"))
    adapt_service.ADAPT_DIR = str(tmp / "adapted")
    assistant.SETTINGS_PATH = str(tmp / "settings.json")
    return TestClient(main.app)


def test_alert_table_matches_the_queue_shown_on_the_overview(client):
    th = client.get("/api/model/thresholds").json()
    summary = client.get("/api/summary").json()
    assert th["n"] == summary["n_queue"]                         # same orders as the Overview, not an older test file
    q = pd.read_csv(os.path.join(ROOT, "backend", "app_data", "queue.csv"))
    assert th["base_rate"] == pytest.approx(q["is_delayed"].mean(), abs=1e-6)
    recalls = [r["recall"] for r in th["rows"]]
    assert recalls == sorted(recalls)


def test_evaluation_table_is_honest_about_real_versus_simulated(client):
    ev = client.get("/api/model/evaluation").json()
    rows = {r["key"]: r for r in ev["rows"]}
    assert {"enriched", "xgboost", "ensemble", "handoff", "tracking", "synthetic_operational"} <= set(rows)
    assert rows["enriched"]["real_data"] is True and rows["enriched"]["serving_queue"] is True
    assert rows["tracking"]["real_data"] is False and rows["synthetic_operational"]["real_data"] is False
    assert all(0.5 <= r["roc_auc"] <= 1.0 for r in rows.values())
    # the served queue's precision/recall are measured on the queue itself and agree with the Overview
    s = client.get("/api/summary").json()
    assert ev["queue"]["alerts"] == s["n_flagged"]


def test_served_queue_auc_matches_recorded_chronological_auc():
    q = pd.read_csv(os.path.join(ROOT, "backend", "app_data", "queue.csv"))
    meta = json.load(open(os.path.join(ROOT, "models", "enriched", "ensemble_meta.json"), encoding="utf-8"))
    assert roc_auc_score(q["is_delayed"], q["probability"]) == pytest.approx(meta["metrics_on_test"]["roc_auc"], abs=0.002)


def test_shipped_dispatch_model_reproduces_its_recorded_synthetic_auc():
    from xgboost import XGBClassifier
    rep = json.load(open(os.path.join(ROOT, "models", "synthetic_operational", "metrics.json"), encoding="utf-8"))
    d = pd.read_csv(os.path.join(ROOT, "data", "raw", "synthetic_operational_orders.csv"))
    d["dispatch_lag_hours"] = (pd.to_datetime(d["dispatch_timestamp"]) - pd.to_datetime(d["order_purchase_timestamp"])).dt.total_seconds() / 3600
    X = d[rep["features"]].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    cut = int(0.85 * len(d))
    m = XGBClassifier()
    m.load_model(os.path.join(ROOT, "models", "synthetic_operational", "xgboost_synthetic_operational.json"))
    auc = roc_auc_score(d["synthetic_late_delivery"].astype(int).to_numpy()[cut:], m.predict_proba(X.iloc[cut:])[:, 1])
    assert auc == pytest.approx(rep["heldout_chronological_test_roc_auc"], abs=0.001)


def test_assistant_reliability_answer_gives_real_and_simulated_figures(client):
    r = client.post("/api/assistant/ask", json={"question": "How reliable is the model?", "engine": "builtin"}).json()
    text = " ".join(r["paragraphs"])
    assert "real orders" in text and "SIMULATED" in text
    assert "0.717" in text and "0.943" in text


def test_ensemble_answer_does_not_claim_the_wrong_model_is_served(client):
    r = client.post("/api/assistant/ask", json={"question": "Is the ensemble better than XGBoost?", "engine": "builtin"}).json()
    text = " ".join(r["paragraphs"])
    assert "Served by default: the ensemble" not in text
    assert "enriched chronological model" in text
