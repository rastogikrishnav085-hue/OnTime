"""End-to-end behaviour of the saved model + guardrails + scorer.
Skipped automatically if the model artifacts / processed data are absent."""
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS, TARGET_COLUMN
from modeling import stratified_split, NUMERIC, CATEGORICAL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "ontime_final_model.joblib")
META = os.path.join(ROOT, "models", "model_metadata.json")
READY = os.path.join(ROOT, "data", "processed", "ontime_model_ready.csv")
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (MODEL, META, READY)),
    reason="model artifacts / processed data not built yet",
)


@pytest.fixture(scope="module")
def bundle():
    from artifacts import load_model, load_json
    from inference_validation import load_reference_stats
    model, meta = load_model(os.path.join(ROOT, "models"), FEATURE_COLUMNS, allow_rebuild=False)
    stats = load_reference_stats(os.path.join(ROOT, "models", "reference_stats.json"))
    lookup = load_json(os.path.join(ROOT, "models", "rate_lookup.json"))
    return model, meta, stats, lookup


@pytest.fixture(scope="module")
def heldout():
    df = pd.read_csv(READY, parse_dates=["order_purchase_timestamp"])
    return stratified_split(df)[1].reset_index(drop=True)


def _inputs(df, drop_rates=True):
    from scoring import RATE_FEATURES
    return df.drop(columns=list(RATE_FEATURES)) if drop_rates else df.copy()


# ---------- artifacts / integrity ----------
def test_model_loads_and_predicts_valid_probabilities(bundle, heldout):
    model = bundle[0]
    p = model.predict_proba(heldout[FEATURE_COLUMNS].head(200))[:, 1]
    assert np.all((p >= 0) & (p <= 1))


def test_tampered_model_file_is_refused(tmp_path):
    from artifacts import load_model_checked, ModelIntegrityError
    bad = tmp_path / "model.joblib"
    shutil.copy(MODEL, bad)
    with open(bad, "ab") as f:
        f.write(b"x")
    with pytest.raises(ModelIntegrityError, match="hash"):
        load_model_checked(str(bad), META, FEATURE_COLUMNS)


def test_changed_feature_list_is_refused():
    from artifacts import load_model_checked, ModelIntegrityError
    with pytest.raises(ModelIntegrityError, match="Feature list"):
        load_model_checked(MODEL, META, FEATURE_COLUMNS[:-1])


def test_threshold_in_metadata_is_sane(bundle):
    assert 0.05 < bundle[1]["threshold"] < 0.95


# ---------- scoring ----------
def test_every_input_row_gets_exactly_one_output_row(bundle, heldout):
    from scoring import score_orders
    model, meta, stats, lookup = bundle
    df = _inputs(heldout.head(50))
    out = score_orders(df, model, meta["threshold"], stats, lookup)
    assert len(out) == len(df) and list(out["row"]) == list(range(len(df)))
    assert set(out["status"]) <= {"SCORED", "REJECTED"}


def test_bad_rows_are_rejected_and_good_rows_still_scored(bundle, heldout):
    from scoring import score_orders
    model, meta, stats, lookup = bundle
    df = _inputs(heldout.head(6)).astype({"product_category": object, "total_price": object})
    df.loc[1, "product_category"] = np.nan          # blank category
    df.loc[2, "total_price"] = -20.0                 # impossible
    df.loc[3, "total_price"] = "twelve"              # text in a numeric field
    out = score_orders(df, model, meta["threshold"], stats, lookup)
    assert list(out["status"]) == ["SCORED", "REJECTED", "REJECTED", "REJECTED", "SCORED", "SCORED"]
    assert all(out.loc[[1, 2, 3], "reject_reasons"] != "")
    assert out.loc[[0, 4, 5], "probability"].notna().all()


def test_missing_required_columns_raise_a_clear_error(bundle, heldout):
    from scoring import score_orders
    model, meta, stats, lookup = bundle
    df = _inputs(heldout.head(5)).drop(columns=["total_price", "payment_type"])
    with pytest.raises(ValueError, match="total_price"):
        score_orders(df, model, meta["threshold"], stats, lookup)


def test_empty_input_raises(bundle):
    from scoring import score_orders
    model, meta, stats, lookup = bundle
    with pytest.raises(ValueError):
        score_orders(pd.DataFrame(), model, meta["threshold"], stats, lookup)


def test_unknown_seller_falls_back_to_platform_rate_with_a_warning(bundle, heldout):
    from scoring import score_orders
    model, meta, stats, lookup = bundle
    df = _inputs(heldout.head(3))
    df["seller_id"] = ["brand_new_seller", None, list(lookup["seller"])[0]]
    out = score_orders(df, model, meta["threshold"], stats, lookup)
    assert out.loc[0, "confidence"] == "LOW_CONFIDENCE" and "no history" in out.loc[0, "warnings"]
    assert out.loc[1, "confidence"] == "LOW_CONFIDENCE" and "no seller_id" in out.loc[1, "warnings"]


def test_supplied_rates_are_respected_over_the_lookup(bundle, heldout):
    from scoring import score_orders
    model, meta, stats, lookup = bundle
    df = heldout.head(20).copy()
    df["seller_id"] = list(lookup["seller"])[0]
    with_rates = score_orders(df, model, meta["threshold"], stats, lookup)
    df2 = _inputs(df)
    df2["seller_id"] = list(lookup["seller"])[0]
    from_lookup = score_orders(df2, model, meta["threshold"], stats, lookup)
    assert not np.allclose(with_rates["probability"], from_lookup["probability"])


def test_flagged_orders_get_reasons_and_unflagged_do_not(bundle, heldout):
    from scoring import score_orders
    model, meta, stats, lookup = bundle
    out = score_orders(_inputs(heldout.head(400)), model, meta["threshold"], stats, lookup)
    ok = out[out.status == "SCORED"]
    assert (ok[ok.flagged == True]["reasons"] != "").all()          # noqa: E712
    assert (ok[ok.flagged == False]["reasons"] == "").all()         # noqa: E712


def test_risk_level_is_monotonic_in_probability(bundle, heldout):
    from scoring import score_orders
    model, meta, stats, lookup = bundle
    out = score_orders(_inputs(heldout.head(500)), model, meta["threshold"], stats, lookup)
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    s = out[out.status == "SCORED"].sort_values("probability")
    assert list(s["risk_level"].map(order)) == sorted(s["risk_level"].map(order))


# ---------- guardrail calibration ----------
def test_ordinary_heldout_orders_are_almost_never_low_confidence(bundle, heldout):
    """A warning that fires on ordinary orders carries no information.
    The first version flagged ~14% of them."""
    from scoring import score_orders
    model, meta, stats, lookup = bundle
    out = score_orders(heldout.head(3000), model, meta["threshold"], stats, lookup)   # rates supplied
    assert (out.status == "SCORED").all()
    assert (out.confidence == "LOW_CONFIDENCE").mean() < 0.02


def test_absurd_values_are_flagged_low_confidence(bundle, heldout):
    from scoring import score_orders
    model, meta, stats, lookup = bundle
    df = heldout.head(5).copy()
    df["seller_customer_distance_km"] = 99999.0
    out = score_orders(df, model, meta["threshold"], stats, lookup)
    assert (out.confidence == "LOW_CONFIDENCE").all()


# ---------- explanations ----------
def test_explanations_reconstruct_the_models_own_probability(bundle, heldout):
    import xgboost as xgb
    model = bundle[0]
    X = heldout[FEATURE_COLUMNS].head(300)
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    c = clf.get_booster().predict(xgb.DMatrix(pre.transform(X)), pred_contribs=True)
    recon = 1 / (1 + np.exp(-c.sum(axis=1)))
    assert np.abs(recon - model.predict_proba(X)[:, 1]).max() < 1e-4


def test_explanations_differ_between_orders(bundle, heldout):
    from prediction import explain_orders
    model = bundle[0]
    X = heldout[FEATURE_COLUMNS]
    top = np.argsort(-model.predict_proba(X)[:, 1])[:40]
    sets = {tuple(f for f, _, _ in e) for e in explain_orders(model, X.iloc[top], top_n=3)}
    assert len(sets) > 3


def test_explanations_are_json_serialisable(bundle, heldout):
    import json
    from prediction import explain_orders
    json.dumps(explain_orders(bundle[0], heldout[FEATURE_COLUMNS].head(5)))


# ---------- suggested actions ----------
def test_suggested_action_rules():
    from scoring import suggested_action, ACTION_BY_DRIVER, DEFAULT_ACTION
    assert suggested_action(None, False) == ""
    assert suggested_action("LOW", False) == "No action needed."
    assert suggested_action("MEDIUM", False).startswith("Monitor")
    assert suggested_action("HIGH", False).startswith("Close to the alert line")
    seller = suggested_action("HIGH", True, "seller_historical_delay_rate")
    assert seller == ACTION_BY_DRIVER["seller_historical_delay_rate"]
    assert suggested_action("CRITICAL", True, "seller_customer_distance_km").startswith("URGENT: ")
    assert suggested_action("HIGH", True, "some_unmapped_feature") == DEFAULT_ACTION


def test_every_mapped_driver_is_a_real_model_feature():
    from scoring import ACTION_BY_DRIVER
    assert set(ACTION_BY_DRIVER) <= set(FEATURE_COLUMNS)


def test_scored_rows_always_carry_an_action_and_rejected_rows_never_do(bundle, heldout):
    from scoring import score_orders
    model, meta, stats, lookup = bundle
    df = _inputs(heldout.head(300))
    df.loc[0, "total_price"] = -5.0  # force one rejection
    out = score_orders(df, model, meta["threshold"], stats, lookup)
    scored, rejected = out[out.status == "SCORED"], out[out.status == "REJECTED"]
    assert len(rejected) >= 1 and (rejected.suggested_action == "").all()
    assert (scored.suggested_action.str.len() > 0).all()
    # flagged orders get a driver-specific or default action, never the "no action" text
    assert not scored[scored.flagged.astype(bool)].suggested_action.str.contains("No action").any()
    crit = scored[scored.risk_level == "CRITICAL"]
    assert crit.suggested_action.str.startswith("URGENT").all()
