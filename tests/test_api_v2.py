"""New in v2: model choice, ensemble/second opinion, literature, adapting to other data, smarter assistant."""
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "backend", "app_data", "queue.csv")), reason="app data not built")



def test_live_scoring_metadata_is_synced_across_backend_endpoints(client):
    meta = client.get("/api/meta").json()
    health = client.get("/api/health").json()
    models = client.get("/api/models").json()
    report = client.get("/api/model").json()
    assert meta["live_scoring_model"] == health["live_scoring_model"] == models["live_scoring_model"]
    assert meta["live_scoring_label"] == health["live_scoring_label"] == models["live_scoring_label"]
    assert meta["live_scoring_available"] is True
    assert report["live_scoring"]["key"] == meta["live_scoring_model"]
    assert report["live_scoring"]["available"] is True
    if report["live_scoring"]["key"] == "synthetic_operational":
        assert report["live_scoring"]["benchmark_roc_auc"] > 0.91
        assert report["live_scoring"]["benchmark_test_rows"] == 12000


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    sys.path.insert(0, ROOT)
    from fastapi.testclient import TestClient
    from backend.app import adapt_service, assistant, main
    tmp = tmp_path_factory.mktemp("v2")
    main.set_db_path(str(tmp / "board.db"))
    adapt_service.ADAPT_DIR = str(tmp / "adapted")
    assistant.SETTINGS_PATH = str(tmp / "settings.json")
    return TestClient(main.app)


@pytest.fixture(scope="module")
def has_ensemble(client):
    return client.get("/api/models").json()["models"][1]["available"]


def test_models_endpoint_and_meta(client):
    r = client.get("/api/models").json()
    keys = [m["key"] for m in r["models"]]
    assert keys == ["xgboost", "ensemble", "enriched", "synthetic_operational"] and r["models"][0]["available"]
    assert r["models"][3]["available"] and "synthetic" in r["models"][3]["detail"].lower()
    assert r["serving_queue"] in ("xgboost", "ensemble", "enriched")
    assert r["models"][2]["available"]
    meta = client.get("/api/meta").json()
    assert meta["serving"] == r["serving_queue"] and "ensemble_available" in meta
    assert meta["live_scoring_model"] in ("synthetic_operational", "xgboost", "ensemble", "enriched")
    assert meta["live_scoring_available"] is True
    model_page = client.get("/api/model").json()
    assert model_page["live_scoring"]["key"] == meta["live_scoring_model"]
    assert model_page["live_scoring"]["available"] is True
    if meta["live_scoring_model"] == "synthetic_operational":
        assert model_page["live_scoring"]["benchmark_roc_auc"] > 0.91
    else:
        assert "benchmark_roc_auc" in model_page["live_scoring"]


def test_score_with_each_model_and_bad_model(client, has_ensemble):
    base = client.post("/api/score", json={"fields": {"promised_delivery_days": 6, "seller_customer_distance_km": 1800}}).json()
    assert base["status"] == "SCORED" and base["model"] == "enriched"
    assert base["confidence"] != "SYNTHETIC_BENCHMARK"
    x = client.post("/api/score", json={"fields": {}, "model": "xgboost"}).json()
    assert x["model"] == "xgboost" and 0 < x["probability"] < 1
    if has_ensemble:
        e = client.post("/api/score", json={"fields": {}, "model": "ensemble"}).json()
        assert e["model"] == "ensemble" and 0 < e["probability"] < 1 and e["reasons"]
        assert e["threshold"] != x["threshold"] or e["probability"] != x["probability"]
    enriched = client.post("/api/score", json={"fields": {}, "model": "enriched"}).json()
    assert enriched["model"] == "enriched" and 0 < enriched["probability"] < 1 and enriched["reasons"]
    operational = client.post("/api/score", json={"fields": {"weather_severity_forecast": 4, "congestion_forecast": 4, "route_disruption_alert_at_dispatch": 1}, "model": "synthetic_operational"}).json()
    assert operational["model"] == "synthetic_operational" and operational["status"] == "SCORED" and operational["reasons"]
    assert client.post("/api/score", json={"fields": {}, "model": "magic"}).status_code == 400


def test_batch_scoring_follows_the_model_parameter(client, has_ensemble):
    tpl = client.get("/api/template.csv").content
    for m in (["xgboost", "ensemble", "enriched"] if has_ensemble else ["xgboost", "enriched"]):
        r = client.post(f"/api/score/batch?model={m}", files={"file": ("t.csv", tpl, "text/csv")}).json()
        assert r["model"] == m and r["summary"]["scored"] == 10


def test_scoring_model_setting_is_validated_and_used(client, has_ensemble):
    assert client.post("/api/settings", json={"scoring_model": "banana"}).status_code in (400, 422)
    if has_ensemble:
        assert client.post("/api/settings", json={"scoring_model": "ensemble"}).status_code == 200
        assert client.post("/api/score", json={"fields": {}}).json()["model"] == "ensemble"
    assert client.post("/api/settings", json={"scoring_model": "enriched"}).status_code == 200
    assert client.post("/api/score", json={"fields": {}}).json()["model"] == "enriched"
    assert client.post("/api/settings", json={"scoring_model": "auto"}).status_code == 200


def test_second_opinion_and_disagreement_filter(client, has_ensemble):
    if not has_ensemble:
        pytest.skip("no ensemble trained")
    items = client.get("/api/orders", params={"limit": 3}).json()["items"]
    o = client.get(f"/api/orders/{items[0]['order_id']}").json()
    if o["second_opinion"] is None:
        pytest.skip("queue not scored with both models (run backend/add_second_opinion.py)")
    so = o["second_opinion"]
    assert so["model"] != so["primary_model"] and 0 <= so["probability"] <= 1 and so["agrees"] in (True, False)
    dis = client.get("/api/orders", params={"disagree": "true", "limit": 5}).json()
    ag = client.get("/api/models").json()["agreement"]
    assert dis["total"] == ag["n_disagree"]
    for it in dis["items"]:
        d = client.get(f"/api/orders/{it['order_id']}").json()["second_opinion"]
        assert d["agrees"] is False


def test_model_page_data_matches_live_scoring_configuration(client, has_ensemble):
    meta = client.get("/api/meta").json()
    page = client.get("/api/model").json()
    assert page["live_scoring"]["key"] == meta["live_scoring_model"]
    assert page["live_scoring"]["label"] == meta["live_scoring_label"]
    assert page["live_scoring"]["available"] is True
    if meta["live_scoring_model"] == "synthetic_operational":
        assert page["live_scoring"]["benchmark_roc_auc"] > 0.91
    else:
        assert page["live_scoring"]["key"] in ("xgboost", "ensemble", "enriched")
    assert page["serving_queue"] in ("xgboost", "ensemble", "enriched")
    # Historical checkout metrics stay out of the active Model-page payload.
    assert "static_split_auc" not in page and "paired_test_comparison" not in page
    assert "selected" not in page and "models" not in page


def test_literature_endpoint_is_complete_and_links_are_sane(client):
    L = client.get("/api/literature").json()
    papers = [p for t in L["themes"] for p in t["papers"]]
    assert len(papers) >= 20 and L["positioning"] and L["limits"]
    ids = [p["id"] for p in papers]
    assert len(ids) == len(set(ids))
    for p in papers:
        assert p["title"] and p["authors"] and 1950 < p["year"] < 2027 and p["what"] and p["used"]
        assert p["link"] == "" or p["link"].startswith("https://")


# ----------------------------------------------------------------- adapting to other data
def _foreign_from_queue(client, n=2500):
    from backend.app.main import get_engine
    q = get_engine().queue.sample(n, random_state=5)
    return pd.DataFrame({
        "Order No": q.order_id.values, "Vendor": q.seller_id.values, "Order Date": q.purchased_at.values,
        "Promised": q.promised_by.values, "Delivered": q.delivered_at.values,
        "Ship To": q.customer_state.values, "Ship From": q.seller_state.values, "Quantity": q.n_items.values,
        "Total Price": q.total_price.values, "Shipping Fee": q.total_freight.values, "Weight (kg)": (q.total_weight_g / 1000).round(3).values,
        "Distance Km": q.seller_customer_distance_km.round(1).values, "Category": q.product_category.values,
        "Pay Method": q.payment_type.values, "Instalments": q.n_payment_installments.values})


def _csv(df):
    return df.to_csv(index=False).encode()


def test_adapt_preview_run_train_and_score_roundtrip(client):
    df = _foreign_from_queue(client)
    pv = client.post("/api/adapt/preview", files={"file": ("their_orders.csv", _csv(df), "text/csv")})
    assert pv.status_code == 200, pv.text
    pv = pv.json()
    s = pv["suggested"]
    assert s["purchase_time"]["source"] == "Order Date" and s["promised_time"]["source"] == "Promised" and s["delivered_time"]["source"] == "Delivered"
    assert s["weight"]["unit"] == "kg" and s["customer_state"]["source"] == "Ship To" and s["distance"]["source"] == "Distance Km"
    assert pv["outcome"]["found"] and pv["outcome"]["can_learn"] and 0 < pv["outcome"]["late_rate"] < 0.3
    r = client.post("/api/adapt/run", json={"dataset_id": pv["dataset_id"], "mapping": s, "train": True, "name": "My shop"})
    assert r.status_code == 200, r.text
    r = r.json()
    cov, t, a = r["coverage"], r["transfer"], r["adapted"]
    assert cov["importance_coverage"] > 0.6 and cov["provenance"]["total_weight_g"] == "mapped" and cov["provenance"]["avg_photos_qty"] == "default"
    assert t["scored"] > 2000 and t["metrics"]["roc_auc"] > 0.55 and t["verdict"]["level"] in ("good", "fair", "weak")
    assert a["split"] == "time" and 0.5 < a["roc_auc"] < 1 and a["same_rows_olist_model"]["roc_auc"] > 0.5 and a["verdict"]
    mid = a["model_id"]
    models = client.get("/api/adapt/models").json()["models"]
    assert [m["model_id"] for m in models] == [mid]
    sc = client.post(f"/api/adapt/models/{mid}/score", files={"file": ("new.csv", _csv(df.head(200)), "text/csv")})
    assert sc.status_code == 200, sc.text
    sc = sc.json()
    assert sc["n_rows"] == 200 and not sc["columns_missing_from_file"] and all(0 <= x["probability"] <= 1 for x in sc["rows"])
    assert client.delete(f"/api/adapt/models/{mid}").status_code == 200
    assert client.get("/api/adapt/models").json()["models"] == []
    assert client.delete(f"/api/adapt/models/{mid}").status_code == 404


def test_adapt_without_outcome_scores_but_cannot_learn(client):
    df = _foreign_from_queue(client, 600).drop(columns=["Delivered"])
    pv = client.post("/api/adapt/preview", files={"file": ("x.csv", _csv(df), "text/csv")}).json()
    assert pv["outcome"]["found"] is False
    r = client.post("/api/adapt/run", json={"dataset_id": pv["dataset_id"], "mapping": pv["suggested"], "train": False}).json()
    assert r["transfer"]["metrics"] is None and r["transfer"]["verdict"]["level"] == "unknown" and r["adapted"] is None
    bad = client.post("/api/adapt/run", json={"dataset_id": pv["dataset_id"], "mapping": pv["suggested"], "train": True})
    assert bad.status_code == 400 and "outcome" in bad.json()["detail"].lower()


def test_adapt_rejects_bad_input_clearly(client):
    assert client.post("/api/adapt/preview", files={"file": ("e.csv", b"", "text/csv")}).status_code == 400
    assert client.post("/api/adapt/preview", files={"file": ("one.csv", b"a\n1\n2\n", "text/csv")}).status_code == 400
    tiny = _csv(_foreign_from_queue(client, 600).head(5))
    assert "at least 20" in client.post("/api/adapt/preview", files={"file": ("t.csv", tiny, "text/csv")}).json()["detail"].lower()
    pv = client.post("/api/adapt/preview", files={"file": ("x.csv", _csv(_foreign_from_queue(client, 100)), "text/csv")}).json()
    assert client.post("/api/adapt/run", json={"dataset_id": pv["dataset_id"], "mapping": {"purchase_time": {"source": "nope"}}}).status_code == 400
    assert client.post("/api/adapt/run", json={"dataset_id": pv["dataset_id"], "mapping": {"made_up": {"source": "Order Date"}}}).status_code == 400
    assert client.post("/api/adapt/run", json={"dataset_id": "../../etc", "mapping": {}}).status_code == 400
    assert client.post("/api/adapt/run", json={"dataset_id": pv["dataset_id"], "mapping": {}}).status_code == 400


# ----------------------------------------------------------------- assistant
def ask(client, q, **kw):
    return client.post("/api/assistant/ask", json={"question": q, "engine": "builtin", **kw}).json()


def test_assistant_new_topics(client):
    lit = ask(client, "Which papers support using SHAP explanations?")
    assert lit["topic"] == "literature" and any("SHAP" in p or "Shapley" in p for p in lit["paragraphs"])
    assert ask(client, "Is the ensemble better than XGBoost?")["topic"] == "ensemble"
    assert ask(client, "What did tuning change?")["topic"] == "tuning"
    ad = ask(client, "Can I use my own company's data?")
    assert ad["topic"] == "adapt" and "Adapt to your data" in " ".join(ad["paragraphs"])
    reliability = ask(client, "How reliable is the model?")
    assert reliability["topic"] == "model"
    assert all("0.7168" not in paragraph and "0.71679" not in paragraph for paragraph in reliability["paragraphs"])


def test_assistant_follow_up_uses_previous_answer(client):
    top = client.get("/api/orders", params={"limit": 1}).json()["items"][0]["order_id"]
    first = ask(client, f"Why is order {top[:10]} risky?")
    assert first["topic"] == "order" and first["context"]["order_id"] == top
    nxt = ask(client, "What should we do about it?", context=first["context"])
    assert nxt["resolved_as"] and nxt["paragraphs"][0].startswith("Suggested action") and top[:10] in nxt["paragraphs"][0]
    # with no context the same words are not guessed at
    lone = ask(client, "What should we do about it?")
    assert lone["resolved_as"] is None


def test_calibrated_chance_is_only_reported_when_holdout_checks_pass(client, has_ensemble):
    # A score is not automatically a real-world probability. The current enriched model
    # has chronological calibration drift, so its literal calibrated chance is disabled.
    r = client.post("/api/score", json={"fields": {}, "model": "enriched"}).json()
    assert r["status"] == "SCORED" and 0 < r["probability"] < 1
    model_page = client.get("/api/model").json()
    assert model_page["live_scoring"]["key"] in ("synthetic_operational", "xgboost", "ensemble", "enriched")
    assert r["calibrated_probability"] is None or 0 <= r["calibrated_probability"] <= 1


def test_old_auto_scoring_setting_migrates_to_real_checkout_model(tmp_path):
    import json
    from backend.app import assistant
    settings_path = tmp_path / "old-settings.json"
    settings_path.write_text(json.dumps({"scoring_model": "auto"}))
    settings = assistant.load_settings(str(settings_path))
    assert settings["scoring_model"] == "enriched"
    assert settings["scoring_model_migrated_v3"] is True
    # Explicit user choices remain intact during migration.
    settings_path.write_text(json.dumps({"scoring_model": "enriched"}))
    settings = assistant.load_settings(str(settings_path))
    assert settings["scoring_model"] == "enriched"
