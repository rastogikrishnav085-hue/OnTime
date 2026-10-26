import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]

def test_synthetic_raw_csv_and_saved_model_reproduce_chronological_metrics():
    csv_path = ROOT / "data/raw/synthetic_operational_orders.csv"
    metrics_path = ROOT / "models/synthetic_operational/metrics.json"
    model_path = ROOT / "models/synthetic_operational/xgboost_synthetic_operational.json"
    assert csv_path.exists() and model_path.exists() and metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    frame = pd.read_csv(csv_path)
    assert len(frame) == 80000 and len(frame.columns) == 38
    assert frame.order_id.str.startswith("SYNTH_").all()
    assert metrics["features"] == ["route_km", "weather_severity_forecast", "congestion_forecast", "carrier_reliability_prior", "capacity_utilization_at_dispatch", "remaining_sla_slack_hours", "facility_backlog_at_dispatch", "route_disruption_alert_at_dispatch", "last_scan_age_hours", "estimated_truck_delay_hours_at_dispatch", "hour_at_dispatch", "weekday_at_dispatch", "price", "freight_value", "product_weight_g", "dispatch_lag_hours"]
    assert "synthetic_late_delivery" not in metrics["features"]
    assert not any("delivered_customer" in x or x.endswith("_id") for x in metrics["features"])
    model = xgb.Booster(); model.load_model(str(model_path))
    test_start = metrics["train_rows"] + metrics["validation_rows"]
    X = frame[metrics["features"]].astype(float).iloc[test_start:]
    y = frame.synthetic_late_delivery.astype(int).iloc[test_start:]
    pred = model.predict(xgb.DMatrix(X, feature_names=metrics["features"]))
    auc = roc_auc_score(y, pred); pr = average_precision_score(y, pred)
    assert auc > 0.91
    assert abs(auc - metrics["heldout_chronological_test_roc_auc"]) < 1e-9
    assert abs(pr - metrics["heldout_chronological_test_pr_auc"]) < 1e-9

def test_production_engine_loads_and_scores_with_operational_model():
    import sys
    sys.path.insert(0, str(ROOT / "backend"))
    from app.engine import Engine
    engine = Engine()
    assert engine.synthetic_model is not None
    info = next(m for m in engine.models_info() if m["key"] == "synthetic_operational")
    assert info["available"] and info["benchmark_roc_auc"] > 0.91
    result = engine.score_one({"seller_customer_distance_km": 800, "promised_delivery_days": 5,
                               "weather_severity_forecast": 4, "congestion_forecast": 4,
                               "carrier_reliability_prior": 0.55, "capacity_utilization_at_dispatch": 1.1,
                               "remaining_sla_slack_hours": 2, "facility_backlog_at_dispatch": 50,
                               "route_disruption_alert_at_dispatch": 1, "last_scan_age_hours": 12,
                               "estimated_truck_delay_hours_at_dispatch": 10, "dispatch_lag_hours": 24},
                              "synthetic_operational")
    assert result["status"] == "SCORED" and 0 <= result["probability"] <= 1
    assert result["confidence"] == "SYNTHETIC_BENCHMARK" and result["warnings"] and result["reasons"]
    rejected = engine.score_one({"total_price": -1}, "synthetic_operational")
    assert rejected["status"] == "REJECTED" and rejected["probability"] is None

def test_operational_model_can_score_synthetic_csv_rows_without_target_as_feature():
    import sys
    sys.path.insert(0, str(ROOT / "backend"))
    from app.engine import Engine, SYNTHETIC_OPERATIONAL_FEATURES
    engine = Engine()
    frame = pd.read_csv(ROOT / "data/raw/synthetic_operational_orders.csv", nrows=25)
    scored = engine.score_batch(frame, "synthetic_operational")
    assert len(scored) == 25 and (scored.status == "SCORED").all()
    assert scored.probability.between(0, 1).all()
    assert "synthetic_late_delivery" not in SYNTHETIC_OPERATIONAL_FEATURES
