# OnTime v6 integrated final validation record

Date: 26 October 2026

## Implemented in this revision

- Changed fresh-install single-order and batch scoring default from the synthetic dispatch benchmark to `enriched`, the real-Olist checkout-time model. The synthetic dispatch model remains selectable for demonstrations and is clearly described as synthetic/opt-in.
- Migrated legacy saved `auto` scoring defaults to `enriched`; explicit saved model choices are preserved.
- Updated assistant reliability responses to state the real chronological result first and distinguish simulated/generated scores.
- Updated README, model policy, static benchmark page, model labels, report and presentation with consistent evidence provenance.
- Moved submission artifacts into `report/` and `ppt/`.
- Corrected the report cover and poster to identify Dr. Bela Joglekar as the internal mentor; certificate retains the internal mentor and two external mentors with signature lines.
- Fixed poster conclusion overflow by shortening the text and setting explicit font sizes.
- Added a reproducible, read-only audit of all nine original Olist CSVs (`scripts/audit_olist_data.py`) and saved its outputs.
- Re-ran the feature leakage audit at four temporal cut dates.
- Optimized the portable-model fallback unit test to use a 12,000-row real-data slice; the test checks recovery behavior, not full-data benchmark performance.

## Data integrity and audit

- All nine original Olist CSVs match the nine user-uploaded files byte-for-byte by SHA-256.
- Source row counts: customers 99,441; geolocation 1,000,163; order items 112,650; payments 103,886; reviews 99,224; orders 99,441; products 32,951; sellers 3,095; category translations 71.
- `python audit_leakage.py`: PASS. No checkout-time features changed when future delivery/carrier outcomes were scrambled at cut dates 2017-06-01, 2017-11-01, 2018-03-01 and 2018-06-15.
- Full raw-file hashes, column names, null counts, duplicate-key checks and timestamp ranges: `data/processed/olist_raw_audit.json` and `DATA_AUDIT_REPORT.md`.

## Model result policy

- Checkout-time model: ROC-AUC 0.71679, PR-AUC 0.10244, on the newest 19,294 real Olist orders in the chronological holdout.
- Handoff model: ROC-AUC 0.85243 on a random split; mean 0.77102 across 17 monthly tests.
- In-transit model without simulated tracking: ROC-AUC 0.84000 on held-out snapshots; mean 0.76688 across seven monthly tests.
- In-transit model with simulated tracking/weather/strike signals: ROC-AUC 0.91460 on held-out simulated snapshots; mean 0.86866 across seven monthly tests.
- Generated dispatch benchmark: ROC-AUC 0.94325 (rounded 0.9433), PR-AUC 0.89270 on 12,000 generated test rows with generated labels. This is not real-world performance.
- No saved model weights were retrained in this revision; the improvement is safer defaults, clearer evidence, better reproducibility and validation.

## Verification

- Automated test suite: 233 tests reported PASSED, 2 existing XGBoost compatibility warnings. The sandbox process did not exit cleanly after printing the final pytest summary, so the full-run process exit status could not be independently confirmed. Focused API tests: 15 passed; model-evaluation tests: 6 passed; portable fallback test: 1 passed.
- Live HTTP smoke test: 11/11 checks passed against the running local app (frontend assets, health/readiness, model metadata/listing, summary, queue, single scoring, batch template and batch scoring).
- LibreOffice headless render: report 23 pages, presentation 17 slides/pages, poster 1 page. Visual review found and fixed poster conclusion overflow. Windows-native launch and native Microsoft Office rendering remain untested.

## Still required before submission or real-world use

- Replace the report's plagiarism/AI-detection placeholder with the institution's genuine report.
- Obtain mentor approval and signatures; confirm the submission date is correct.
- Run a clean Windows launch on the target machine.
- For real-world performance claims, record timestamped predictions before outcomes and validate them against genuine observed operational outcomes. Olist does not include the actual dispatch/tracking telemetry needed to certify a prospective dispatch-stage AUC above 0.85.
