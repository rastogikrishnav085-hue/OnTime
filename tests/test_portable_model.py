"""The portable model format and the load fallbacks.

Background: the original joblib pickle (which embeds XGBoost's binary
booster) loaded on Linux but failed on a Windows laptop with
'XGBoostError: input stream corrupted'. The model is now stored as a
scikit-learn preprocessor + XGBoost's native JSON, and load_model()
falls back to retraining locally if even that cannot be read."""
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
READY = os.path.join(ROOT, "data", "processed", "ontime_model_ready.csv")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(MODELS, "ontime_xgb_booster.json")),
                                reason="portable model not exported")


@pytest.fixture(scope="module")
def sample():
    df = pd.read_csv(READY, nrows=3000, parse_dates=["order_purchase_timestamp"])
    return df[FEATURE_COLUMNS]


def _copy_models(tmp_path, names=("model_metadata.json", "ontime_preprocessor.joblib", "ontime_xgb_booster.json")):
    for n in names:
        shutil.copy(os.path.join(MODELS, n), tmp_path / n)
    return str(tmp_path)


def test_portable_model_loads_without_the_pickle(tmp_path, sample):
    from artifacts import load_model
    d = _copy_models(tmp_path)  # deliberately no ontime_final_model.joblib
    model, meta = load_model(d, FEATURE_COLUMNS, allow_rebuild=False)
    p = model.predict_proba(sample)[:, 1]
    assert np.all((p >= 0) & (p <= 1)) and "portable" in meta


def test_portable_matches_the_original_pickle(sample):
    from artifacts import load_model, load_model_checked
    portable, _ = load_model(MODELS, FEATURE_COLUMNS, allow_rebuild=False)
    try:
        original, _ = load_model_checked(os.path.join(MODELS, "ontime_final_model.joblib"),
                                         os.path.join(MODELS, "model_metadata.json"), FEATURE_COLUMNS)
    except Exception:
        pytest.skip("this machine cannot read the original pickle (the reason the portable format exists)")
    assert np.allclose(portable.predict_proba(sample)[:, 1], original.predict_proba(sample)[:, 1], atol=1e-6)


def test_tampered_booster_is_refused_not_silently_rebuilt(tmp_path):
    from artifacts import load_model, ModelIntegrityError
    d = _copy_models(tmp_path)
    with open(os.path.join(d, "ontime_xgb_booster.json"), "a") as f:
        f.write(" ")
    with pytest.raises(ModelIntegrityError, match="hash"):
        load_model(d, FEATURE_COLUMNS, data_path=READY)


def test_unreadable_model_is_rebuilt_from_data_and_reused(tmp_path, sample):
    from artifacts import load_model
    d = _copy_models(tmp_path)
    meta_path = os.path.join(d, "model_metadata.json")
    meta = json.load(open(meta_path))
    meta["portable"]["booster_file"] = "not_there.json"   # simulate a model this machine can't read
    json.dump(meta, open(meta_path, "w"))
    # A small real-data slice is enough to test the recovery path; training the full
    # 96k-row dataset here turns a unit test into a multi-minute benchmark.
    small_data = tmp_path / "small_ready.csv"
    pd.read_csv(READY, nrows=12000).to_csv(small_data, index=False)
    model, m = load_model(d, FEATURE_COLUMNS, data_path=str(small_data), log=lambda *_: None)
    assert 0.5 < m["rebuilt_test_roc_auc"] <= 1.0
    assert os.path.exists(os.path.join(d, "local_rebuild", "ontime_xgb_booster.json"))
    again, m2 = load_model(d, FEATURE_COLUMNS, allow_rebuild=False)   # next start: no retraining
    assert np.allclose(model.predict_proba(sample)[:, 1], again.predict_proba(sample)[:, 1])
