# OnTime v5 cleanup report

Date: 2026-10-24

## Removed after inspection

| Path | Size before removal | Reason |
|---|---:|---|
| `backend/app_data/tracking` | 8,951,019 bytes | Duplicate/unused app-side copy of simulated tracking context; current serving code reads data/synthetic/; 6 files |
| `models/synthetic_operational/synthetic_sample_1000.csv` | 176,335 bytes | Unreferenced 1,000-row sample; canonical full synthetic CSV is data/raw/synthetic_operational_orders.csv |
| `experiments` | 8,111 bytes | Standalone exploratory scripts are not imported or invoked by production, setup, or tests; 2 files |
| `CHANGES_AND_HANDOFF.md` | 21,826 bytes | Superseded historical handoff with stale paths/status; current README and model docs cover supported workflow |
| `FINAL_TEST_RESULTS.txt` | 1,344 bytes | Superseded by refreshed FINAL_PACKAGE_VERIFICATION.txt after cleanup tests |

**Total source bytes removed:** 9,158,635 bytes (8.73 MiB). This is uncompressed source size; compressed ZIP reduction differs.

## Preserved

- All nine original Olist CSVs under `data/raw/`, SHA-256 verified against `data/processed/raw_manifest.json`.
- Separate `data/raw/synthetic_operational_orders.csv` (80,000 rows, 38 columns), its training script, saved model, metrics, feature importance and data dictionary.
- Production checkout and dispatch scoring code, models, APIs, frontend, configuration templates, tests, notebooks, reports, dependency manifests and reproducible build/training scripts.
- `.env.example` retained; no actual `.env`, local database or SQLite file was found in the inspected package.
- `data/interim/` and `data/processed/` tables were preserved because build, training, API, dashboard, or test code references them.
- `data/synthetic/` retained as the runtime source for simulated tracking. The duplicate copies under `backend/app_data/tracking/` were removed because current serving code reads `data/synthetic/`.

## Verification

- Full test suite: **204 passed, 2 warnings**. Warnings are the existing legacy XGBoost 3.4.1 pickle compatibility warning under XGBoost 3.1.3.
- Synthetic benchmark saved-model reproduction test passed: chronological held-out test ROC-AUC **0.9432527722**, PR-AUC **0.8927009432**.
- Chronological split: 56,000 train / 12,000 validation / 12,000 untouched test rows. The score applies only to the simulated dispatch benchmark, not observed real-world Olist performance.
- Final archive is checked separately for ZIP integrity, cache/environment exclusion and raw dataset hashes.
