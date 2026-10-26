# Windows startup fix: optional LightGBM/Dask import failure

Date: 2026-10-24

## Root cause

The affected environment crashes while importing `lightgbm`; LightGBM opportunistically imports Dask, and that installed Dask/pandas combination raises a `TypeError` during module initialization. The failure happens before FastAPI can start, even though the shipped serving path does not require LightGBM training.

## Changes applied

- `src/enriched_model.py` treats LightGBM as optional for serving and catches import-time exceptions, including `TypeError`. Training helpers provide a clear error if LightGBM is actually requested but unavailable.
- `requirements-app.txt` no longer installs LightGBM in the minimal app environment. `requirements.txt` retains LightGBM for the full training/notebook environment.
- `backend/app/engine.py` emits a non-fatal diagnostic if LightGBM is unavailable and continues loading the shipped serving artifacts.
- `tests/test_optional_lightgbm_import.py` injects the same class of import-time `TypeError` and verifies that the application engine still loads the order queue and synthetic operational model.
- `README.md` documents the fix and recommends launching through `run_all.bat` or `start_ontime.bat`, which use the project `.venv`.

## Verification

- Full automated test suite: 211 passed, 2 warnings.
- Python compilation: passed.
- Frontend JavaScript syntax check: passed.
- Regression test: passed with a simulated LightGBM import `TypeError`.
- Local engine startup under the simulated import failure: passed; historical queue and live model registry loaded.

The two remaining warnings are existing legacy XGBoost/pickle compatibility warnings; they are separate from this startup crash.

## Windows instructions

1. Extract this updated bundle into a new folder rather than overlaying it on the old copy in OneDrive.
2. Double-click `run_all.bat` (recommended) or `start_ontime.bat`. The launcher uses the project's isolated `.venv`.
3. If the local environment is stale, close OnTime, delete only the project's `.venv` folder, and run `setup.bat` once. Do not delete `models`, `data`, or `backend/app_data`.
4. Open `http://localhost:8000`. If startup still fails, send the first new traceback after applying this package.
