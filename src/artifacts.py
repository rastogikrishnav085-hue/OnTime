"""
artifacts.py
-------------
Everything needed to *use* the trained model on new orders, saved
alongside the model so the three travel together:

  - rate_lookup.json    : current seller / category / state delay rates,
                          so a new order only needs a seller_id, category
                          and state -- not three hand-computed rates.
  - model_metadata.json : feature list, threshold, library versions and a
                          SHA-256 of the model file.
  - load_model_checked(): refuses to load a model whose file was changed,
                          or whose expected feature list no longer matches
                          the code, instead of failing silently later.

Why the metadata matters: a joblib pickle is only guaranteed to load
correctly under the same scikit-learn / xgboost versions it was saved
with, and it has no memory of which columns it expects. Without these
checks a mismatch shows up as a wrong prediction, not as an error.
"""

import hashlib
import json
import os
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd


def build_rate_lookup(joined_df, min_orders=1):
    """
    Delay rate per seller / product category / customer state over ALL
    historical orders, plus the global rate used for anything unseen.
    This mirrors how the training features were built (raw historical
    mean, global rate when there is no history) so a scored order sees
    the same kind of number the model was trained on.
    """
    def table(key):
        g = joined_df.groupby(key)["is_delayed"].agg(["mean", "count"])
        g = g[g["count"] >= min_orders]
        return {str(k): {"rate": float(r["mean"]), "n_orders": int(r["count"])} for k, r in g.iterrows()}

    return {
        "global_rate": float(joined_df["is_delayed"].mean()),
        "seller": table("primary_seller_id"),
        "category": table("product_category"),
        "customer_state": table("customer_state"),
        "built_from_orders": int(len(joined_df)),
    }


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lib_versions():
    import sklearn, xgboost
    return {
        "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
    }


def save_model_metadata(model_path, feature_columns, numeric, categorical, threshold, metrics, path):
    meta = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_file": os.path.basename(model_path),
        "model_sha256": _sha256(model_path),
        "feature_columns": list(feature_columns),
        "numeric": list(numeric),
        "categorical": list(categorical),
        "threshold": float(threshold),
        "metrics_on_test": metrics,
        "library_versions": _lib_versions(),
    }
    save_json(meta, path)
    return meta


class ModelIntegrityError(RuntimeError):
    pass


def load_model_checked(model_path, metadata_path, expected_feature_columns):
    """
    Loads the model only if (1) the file matches the hash recorded when
    it was saved and (2) the feature list it was trained on equals the
    one the current code would feed it. Library-version drift is a
    warning, not an error, because it usually still works -- but it is
    the first thing to check if predictions ever look wrong.
    """
    meta = load_json(metadata_path)
    actual_hash = _sha256(model_path)
    if actual_hash != meta["model_sha256"]:
        raise ModelIntegrityError(
            "Model file does not match the hash recorded in model_metadata.json -- "
            "it was modified or replaced after saving. Refusing to load."
        )
    if list(expected_feature_columns) != meta["feature_columns"]:
        missing = sorted(set(meta["feature_columns"]) - set(expected_feature_columns))
        extra = sorted(set(expected_feature_columns) - set(meta["feature_columns"]))
        raise ModelIntegrityError(
            "Feature list in the current code differs from the one this model was trained on "
            f"(missing from code: {missing}, new in code: {extra}). Retrain the model."
        )
    now, then = _lib_versions(), meta["library_versions"]
    for lib in ("scikit-learn", "xgboost"):
        if now[lib].split(".")[:2] != then[lib].split(".")[:2]:
            warnings.warn(
                f"{lib} is {now[lib]} now but the model was saved with {then[lib]}; "
                "if predictions look wrong, retrain under the current version."
            )
    return joblib.load(model_path), meta


# ---------------------------------------------------------------------------
# Portable model format (works across Windows / macOS / Linux)
# ---------------------------------------------------------------------------
# A joblib pickle of an XGBoost pipeline embeds XGBoost's binary booster
# buffer. That loaded fine on Linux but failed on a Windows laptop with
# "XGBoostError: input stream corrupted", even though the file's SHA-256
# matched. So the model is also stored in two portable pieces:
#   ontime_preprocessor.joblib  - the scikit-learn ColumnTransformer
#   ontime_xgb_booster.json     - the XGBoost model in its native JSON
#                                 format, which XGBoost guarantees to load
#                                 on any platform
# load_model() uses these first. If they still cannot be loaded on some
# machine, it retrains the identical model locally from
# data/processed/ontime_model_ready.csv (about a minute) instead of failing.

PORTABLE_PRE = "ontime_preprocessor.joblib"
PORTABLE_BOOSTER = "ontime_xgb_booster.json"
LOCAL_REBUILD_DIR = "local_rebuild"


def _json_safe_params(params):
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, float) and np.isnan(v):
            continue  # 'missing=nan' is XGBoost's default anyway
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
    return out


def export_portable(pipe, model_dir, metadata_path=None):
    """Write the preprocessor + JSON booster next to the model and record
    their hashes (and the XGBoost hyperparameters) in the metadata."""
    os.makedirs(model_dir, exist_ok=True)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    pre_path = os.path.join(model_dir, PORTABLE_PRE)
    booster_path = os.path.join(model_dir, PORTABLE_BOOSTER)
    joblib.dump(pre, pre_path)
    clf.save_model(booster_path)
    metadata_path = metadata_path or os.path.join(model_dir, "model_metadata.json")
    meta = load_json(metadata_path) if os.path.exists(metadata_path) else {}
    meta.update({
        "portable": {
            "preprocessor_file": PORTABLE_PRE, "preprocessor_sha256": _sha256(pre_path),
            "booster_file": PORTABLE_BOOSTER, "booster_sha256": _sha256(booster_path),
        },
        "xgb_params": _json_safe_params(clf.get_params()),
    })
    save_json(meta, metadata_path)
    return meta


def _load_portable(model_dir, meta):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    info = meta["portable"]
    pre_path = os.path.join(model_dir, info["preprocessor_file"])
    booster_path = os.path.join(model_dir, info["booster_file"])
    for path, key in ((pre_path, "preprocessor_sha256"), (booster_path, "booster_sha256")):
        if _sha256(path) != info[key]:
            raise ModelIntegrityError(
                f"{os.path.basename(path)} does not match the hash in model_metadata.json -- "
                "it was modified or replaced after saving. Refusing to load.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre = joblib.load(pre_path)
    clf = XGBClassifier()
    clf.load_model(booster_path)
    return Pipeline([("pre", pre), ("clf", clf)])


def _check_features(meta, expected_feature_columns):
    if list(expected_feature_columns) != meta["feature_columns"]:
        missing = sorted(set(meta["feature_columns"]) - set(expected_feature_columns))
        extra = sorted(set(expected_feature_columns) - set(meta["feature_columns"]))
        raise ModelIntegrityError(
            "Feature list in the current code differs from the one this model was trained on "
            f"(missing from code: {missing}, new in code: {extra}). Retrain the model.")


def rebuild_model(model_dir, meta, data_path, log=print):
    """Retrain the final model from the processed data with the saved
    hyperparameters, on the same stratified training split, and save it
    in the portable format under models/local_rebuild/."""
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score
    from features import TARGET_COLUMN
    from modeling import build_preprocessor, stratified_split
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot rebuild the model: {data_path} is missing.")
    log("Rebuilding the model locally from the processed data (one-off, about a minute)...")
    df = pd.read_csv(data_path, parse_dates=["order_purchase_timestamp"])
    train, test = stratified_split(df)
    cols = meta["feature_columns"]
    pipe = Pipeline([("pre", build_preprocessor()), ("clf", XGBClassifier(**meta["xgb_params"]))])
    pipe.fit(train[cols], train[TARGET_COLUMN])
    auc = roc_auc_score(test[TARGET_COLUMN], pipe.predict_proba(test[cols])[:, 1])
    out_dir = os.path.join(model_dir, LOCAL_REBUILD_DIR)
    os.makedirs(out_dir, exist_ok=True)
    local_meta = {k: v for k, v in meta.items() if k not in ("portable",)}
    local_meta["rebuilt_locally_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    local_meta["rebuilt_test_roc_auc"] = round(float(auc), 4)
    local_meta["library_versions"] = _lib_versions()
    save_json(local_meta, os.path.join(out_dir, "model_metadata.json"))
    export_portable(pipe, out_dir)
    log(f"Model rebuilt (test ROC-AUC {auc:.3f}) and saved to {out_dir}.")
    return pipe, load_json(os.path.join(out_dir, "model_metadata.json"))


def load_model(model_dir, expected_feature_columns, data_path=None, allow_rebuild=True, log=print):
    """
    The one function everything should use to get the model.

    Order of attempts:
      1. portable files (preprocessor joblib + XGBoost JSON)
      2. a previous local rebuild (models/local_rebuild/)
      3. the original joblib pickle
      4. retrain locally from the processed data
    A hash mismatch or feature-list mismatch always stops immediately:
    those mean the files were tampered with or the code changed, and a
    silent fallback would hide that.
    """
    meta = load_json(os.path.join(model_dir, "model_metadata.json"))
    _check_features(meta, expected_feature_columns)
    problems = []

    if meta.get("portable"):
        try:
            return _load_portable(model_dir, meta), meta
        except ModelIntegrityError:
            raise
        except Exception as e:  # noqa: BLE001 - any load failure moves to the next option
            problems.append(f"portable model: {e.__class__.__name__}: {e}")

    local_dir = os.path.join(model_dir, LOCAL_REBUILD_DIR)
    local_meta_path = os.path.join(local_dir, "model_metadata.json")
    if os.path.exists(local_meta_path):
        try:
            local_meta = load_json(local_meta_path)
            _check_features(local_meta, expected_feature_columns)
            return _load_portable(local_dir, local_meta), local_meta
        except Exception as e:  # noqa: BLE001
            problems.append(f"local rebuild: {e.__class__.__name__}: {e}")

    legacy = os.path.join(model_dir, meta.get("model_file", "ontime_final_model.joblib"))
    if os.path.exists(legacy):
        try:
            return load_model_checked(legacy, os.path.join(model_dir, "model_metadata.json"), expected_feature_columns)
        except ModelIntegrityError:
            raise
        except Exception as e:  # noqa: BLE001
            problems.append(f"joblib model: {e.__class__.__name__}: {e}")

    if allow_rebuild and data_path and meta.get("xgb_params"):
        for p in problems:
            log(f"Could not load {p}")
        return rebuild_model(model_dir, meta, data_path, log=log)

    raise RuntimeError("The model could not be loaded on this machine:\n  " + "\n  ".join(problems or ["no model files found"]))
