import numpy as np
import pandas as pd
import pytest

from scripts.validate_operational_outcomes import validate


def _write_pair(tmp_path, n=250):
    rng = np.random.default_rng(1234)
    order_id = [f"real-order-{i:04d}" for i in range(n)]
    label = np.array([0, 1] * (n // 2) + ([0] if n % 2 else []))
    probability = np.clip(0.12 + 0.72 * label + rng.normal(0, 0.12, n), 0.01, 0.99)
    # The test data is deliberately generated only as a unit-test fixture. It is
    # never stored or represented as real operational evidence.
    predictions = pd.DataFrame({
        "order_id": order_id,
        "probability": probability,
        "scored_at": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC").astype(str),
    })
    outcomes = pd.DataFrame({"order_id": order_id, "is_delayed": label})
    pred_path, out_path = tmp_path / "predictions.csv", tmp_path / "outcomes.csv"
    predictions.to_csv(pred_path, index=False)
    outcomes.to_csv(out_path, index=False)
    return pred_path, out_path


def test_operational_validation_uses_joined_labels_and_chronological_holdout(tmp_path):
    pred_path, out_path = _write_pair(tmp_path)
    report = validate(pred_path, out_path, threshold=0.5, min_rows=200)
    assert report["status"] == "ELIGIBLE_FOR_HUMAN_REVIEW"
    assert report["operational_approval"] is False
    assert report["quality"]["matched_rows"] == 250
    assert report["chronological_validation"]["method"].startswith("chronological last-20-percent holdout")
    assert report["chronological_validation"]["holdout"]["metrics"]["n"] == 50
    assert 0 <= report["overall_metrics"]["roc_auc"] <= 1
    assert "interventions prevent delays" in report["limitations"][0]


def test_operational_validation_refuses_duplicate_order_ids(tmp_path):
    pred_path, out_path = _write_pair(tmp_path, n=40)
    p = pd.read_csv(pred_path)
    p.loc[1, "order_id"] = p.loc[0, "order_id"]
    p.to_csv(pred_path, index=False)
    with pytest.raises(ValueError, match="duplicate order_id"):
        validate(pred_path, out_path, min_rows=20)


def test_operational_validation_not_eligible_without_prediction_timestamps(tmp_path):
    pred_path, out_path = _write_pair(tmp_path, n=60)
    p = pd.read_csv(pred_path).drop(columns="scored_at")
    p.to_csv(pred_path, index=False)
    report = validate(pred_path, out_path, min_rows=20)
    assert report["status"] == "INSUFFICIENT_EVIDENCE"
    assert report["chronological_validation"] is None
    assert "missing scored_at" in report["quality"]["timestamp_check"]


def test_operational_validation_rejects_unlabelled_outcomes(tmp_path):
    pred_path, out_path = _write_pair(tmp_path, n=40)
    pd.DataFrame({"order_id": ["x"], "status": ["delivered"]}).to_csv(out_path, index=False)
    with pytest.raises(ValueError, match="Outcomes must include"):
        validate(pred_path, out_path, min_rows=20)
