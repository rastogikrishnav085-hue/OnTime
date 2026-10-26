# OnTime v5 — Complete All-in-One Bundle (Startup Fix Included)

This archive contains the full OnTime v5 project, data, model artifacts, frontend/backend, tests, validation tooling, and reports, including the fix for the LightGBM/Dask import crash.

## Start here
1. Extract the `OnTime_v5/` folder into a new local folder (avoid overlaying an old OneDrive-synced copy).
2. Double-click `OnTime_v5/run_all.bat` or `OnTime_v5/start_ontime.bat`; these launchers use the project-local virtual environment.
3. Read `OnTime_v5/README.md` for setup and launch instructions.
4. Read `OnTime_v5/DELIVERY_DOCUMENTATION/LIGHTGBM_STARTUP_FIX_REPORT.md` for details of the startup fix.
5. Read `OnTime_v5/DELIVERY_DOCUMENTATION/REAL_WORLD_VALIDATION_GUIDE.md` for validating predictions against genuine delivery outcomes.

## Verification and limitations
- Automated suite: 211 passed, 2 legacy XGBoost/pickle compatibility warnings.
- A regression test simulates the reported LightGBM/Dask import-time `TypeError` and confirms the application engine can still start.
- Automated tests do not establish real-world predictive accuracy. That requires genuine, previously unseen delivery outcomes and testing in the target deployment environment.
- External-service connectivity must be verified in the target environment.
