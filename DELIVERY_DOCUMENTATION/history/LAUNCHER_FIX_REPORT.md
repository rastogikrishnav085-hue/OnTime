# OnTime v5 - launcher fix (build 3)

## What was wrong
1. `run_all.bat` (the file you double-click) was saved with Unix (LF-only) line endings. Windows `cmd.exe` can misparse such files. Every other `.bat` already used Windows (CRLF) endings.
2. `run_all.bat` did not have the OneDrive database protection, the "port already in use" check or the error handling that `start_ontime.bat` has, and it still said "OnTime v2".
3. If `.venv` existed but a package was missing or broken, `start_ontime.bat` did not repair it.

## What changed
- `run_all.bat` is now a small CRLF wrapper that calls `start_ontime.bat`, so there is one launcher to maintain.
- `start_ontime.bat` now checks that fastapi, uvicorn, pandas, scikit-learn, xgboost, joblib and numpy import cleanly, and re-runs `setup.bat` if not. It also creates `.env` if missing and returns the real error code when the server stops.
- New test `tests/test_launcher_line_endings.py` fails if any `.bat` has bare-LF endings.

## Verified here (Linux, Python 3.12)
- 214 tests passed (211 before + 3 new).
- HTTP smoke test 11/11 against a running server.
- All 24 GET endpoints that take no parameters answered without errors; no tracebacks in the server log.
- Scoring with all four models (XGBoost, ensemble, enriched, dispatch) and edge-case inputs (empty, negative, text, unknown category) returned no server errors.
- Your exact LightGBM/Dask `TypeError` was simulated: the app still started and all four models scored.

## NOT verified here
- I cannot run Windows `.bat` files in this environment. The batch changes are small and follow the patterns already used in `setup.bat`, but only launching it on your laptop confirms them.
- Live weather needs internet from your network.
- Real-world prediction accuracy needs real delivery outcomes (see REAL_WORLD_VALIDATION_GUIDE.md).
