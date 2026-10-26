"""The ensemble: blend maths, weights fitted on out-of-fold predictions, safe save/load, explanations."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ensemble import (BlendModel, EnsembleError, build_hgb, build_lr, build_xgb, evaluate_scores, fit_weights,  # noqa: E402
                      load_ensemble, oof_predictions, save_ensemble, stacked_cv_score)
from features import FEATURE_COLUMNS                                                                     # noqa: E402
from modeling import CATEGORICAL, NUMERIC                                                                # noqa: E402
from prediction import explain_orders                                                                    # noqa: E402


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(0)
    n = 1400
    X = pd.DataFrame({c: rng.uniform(0, 10, n) for c in NUMERIC})
    cats = {"product_category": ["a", "b", "c", "d"], "customer_state": ["SP", "RJ", "BA"], "seller_state": ["SP", "MG"],
            "payment_type": ["credit_card", "boleto"]}
    for c in CATEGORICAL:
        X[c] = rng.choice(cats[c], n)
    z = 0.6 * X["seller_customer_distance_km"] - 0.4 * X["promised_delivery_days"] + (X["customer_state"] == "BA") * 1.5 - 1.5
    y = pd.Series((rng.rand(n) < 1 / (1 + np.exp(-z))).astype(int))
    return X[FEATURE_COLUMNS], y


def _pipes():
    return {"xgb": build_xgb({"n_estimators": 40, "max_depth": 3}), "hgb": build_hgb({"max_iter": 40}), "lr": build_lr()}


def test_blend_fit_predict_and_weights(data):
    X, y = data
    blend = BlendModel(_pipes(), weights={"xgb": 2, "hgb": 1, "lr": 1}).fit(X, y)
    assert abs(sum(blend.weights.values()) - 1) < 1e-9 and blend.weights["xgb"] == 0.5
    p = blend.predict_proba(X)
    assert p.shape == (len(X), 2) and np.allclose(p.sum(axis=1), 1) and p.min() >= 0 and p.max() <= 1
    # the blend is exactly the weighted mean of its members
    mp = blend.member_probabilities(X)
    assert np.allclose(p[:, 1], 0.5 * mp["xgb"] + 0.25 * mp["hgb"] + 0.25 * mp["lr"])


def test_bad_weights_are_refused(data):
    with pytest.raises(ValueError):
        BlendModel(_pipes(), weights={"xgb": 0, "hgb": 0, "lr": 0})
    with pytest.raises(ValueError):
        BlendModel(_pipes(), weights={"xgb": -1, "hgb": 1, "lr": 1})
    with pytest.raises(ValueError):
        BlendModel({})


def test_fitted_weights_are_a_simplex_and_not_worse_than_best_member(data):
    X, y = data
    oof = oof_predictions(_pipes(), X, y, cv=3, log=lambda m: None)
    w, best = fit_weights(oof, y, step=0.1)
    assert abs(sum(w.values()) - 1) < 1e-9 and all(v >= 0 for v in w.values())
    from sklearn.metrics import roc_auc_score
    assert best >= max(roc_auc_score(y, oof[c]) for c in oof) - 1e-12   # single members are on the grid
    auc, ap = stacked_cv_score(oof, y)
    assert 0.5 < auc <= 1 and 0 < ap <= 1


def test_oof_has_no_row_predicted_by_a_model_that_saw_it(data):
    X, y = data
    oof = oof_predictions({"lr": build_lr()}, X, y, cv=4, log=lambda m: None)
    assert oof["lr"].notna().all() and (oof["lr"] > 0).all() and (oof["lr"] < 1).all()


def test_save_load_roundtrip_and_tamper_checks(data, tmp_path):
    X, y = data
    blend = BlendModel(_pipes(), weights={"xgb": 1, "hgb": 1, "lr": 0}).fit(X, y)
    d = str(tmp_path / "ens")
    meta = save_ensemble(blend, d, FEATURE_COLUMNS, 0.55, {"roc_auc": 0.7}, extra={"serve_by_default": False})
    loaded, meta2 = load_ensemble(d, FEATURE_COLUMNS)
    assert np.allclose(loaded.predict_proba(X)[:, 1], blend.predict_proba(X)[:, 1])
    assert meta2["threshold"] == 0.55 and meta["weights"]["lr"] == 0
    # a changed feature list, a missing file and an edited file are all refused
    with pytest.raises(EnsembleError):
        load_ensemble(d, FEATURE_COLUMNS[:-1])
    with open(os.path.join(d, "hgb_pipeline.joblib"), "ab") as f:
        f.write(b"x")
    with pytest.raises(EnsembleError, match="hash"):
        load_ensemble(d, FEATURE_COLUMNS)
    d2 = str(tmp_path / "ens2")
    save_ensemble(blend, d2, FEATURE_COLUMNS, 0.55, {})
    os.remove(os.path.join(d2, "lr_pipeline.joblib"))
    with pytest.raises(EnsembleError, match="missing"):
        load_ensemble(d2, FEATURE_COLUMNS)
    with pytest.raises(EnsembleError, match="No ensemble"):
        load_ensemble(str(tmp_path / "nowhere"), FEATURE_COLUMNS)


def test_explanations_come_from_the_xgboost_member(data):
    X, y = data
    blend = BlendModel(_pipes()).fit(X, y)
    X2 = X.head(5).copy()
    why = explain_orders(blend, X2, top_n=3)
    direct = explain_orders(blend.members["xgb"], X2, top_n=3)
    assert why == direct and len(why) == 5


def test_evaluate_scores_fields():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0, .5, 90), np.linspace(.4, 1, 10)])
    r = evaluate_scores(y, p, 0.6)
    assert 0.5 < r["roc_auc"] <= 1 and r["recall_at_top10pct"] >= 0.5 and 0 <= r["flagged_share"] <= 1


def test_calibration_is_monotone_bounded_and_fixes_inflated_scores():
    from ensemble import apply_calibration, fit_calibration, reliability_table
    rng = np.random.RandomState(1)
    true_p = rng.beta(1, 11, 20000)                       # real late chance, about 8%
    y = (rng.rand(20000) < true_p).astype(int)
    score = 1 / (1 + np.exp(-(np.log(true_p / (1 - true_p)) + 2.5)))   # class-balanced style: inflated, same ranking
    cal = fit_calibration(score, y)
    c = apply_calibration(cal, np.sort(score))
    assert np.all(np.diff(c) >= -1e-12) and c.min() >= 0 and c.max() <= 1
    _, ece_raw = reliability_table(y, score)
    _, ece_cal = reliability_table(y, apply_calibration(cal, score))
    assert ece_raw > 0.2 and ece_cal < 0.02
    assert apply_calibration(cal, [-5, 5]).shape == (2,)              # out-of-range scores are clipped, not errors
