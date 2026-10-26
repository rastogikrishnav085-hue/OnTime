"""
export_portable_model.py
-------------------------
Writes the portable copy of the final model (scikit-learn preprocessor +
XGBoost JSON) from models/ontime_final_model.joblib, and records their
hashes and the XGBoost settings in models/model_metadata.json.

Run it after retraining in notebooks 06-08 (rebuild_app_data.bat does this
for you). The app loads the portable copy first because XGBoost's JSON
format reads the same on Windows, macOS and Linux, while the joblib pickle
failed on Windows with "input stream corrupted".

    python export_portable_model.py
"""
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from features import FEATURE_COLUMNS                             # noqa: E402
from artifacts import load_model_checked, export_portable        # noqa: E402

M = os.path.join(HERE, "models")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe, _ = load_model_checked(os.path.join(M, "ontime_final_model.joblib"),
                                     os.path.join(M, "model_metadata.json"), FEATURE_COLUMNS)
    meta = export_portable(pipe, M)
    print("Portable model written:", meta["portable"]["preprocessor_file"], "+", meta["portable"]["booster_file"])
