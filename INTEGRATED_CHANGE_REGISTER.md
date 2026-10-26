# OnTime v6 integrated revision — changes and remaining work
Date: 26 October 2026

## Data and metric wording now integrated
- The headline checkout-time result remains ROC-AUC **0.717** on the newest 19,294 historical Olist orders using chronological evaluation.
- Random-split XGBoost/ensemble results (0.804/0.809) are kept separate from the chronological result.
- The dispatch benchmark is consistently named an **Olist-adapted generated operational dataset**. Olist-derived entity/value distributions are reused, while dispatch/weather/congestion/capacity/carrier signals and target labels are simulated/generated.
- The dispatch result is **ROC-AUC 0.9433 and PR-AUC 0.8927 on a 12,000-row chronological synthetic holdout**. It is not a measured result on real Olist dispatch outcomes.
- The in-transit 0.915 result is labelled as a simulated-tracking experiment.
- The app now has an always-visible data-provenance banner linking to Model details.
- README wording was updated to avoid implying that a synthetic benchmark's performance automatically transfers to real operations.

## Frontend — implemented in this revision
- Added an accessible, responsive provenance banner above the app shell.
- Banner distinguishes historical checkout metrics from the generated dispatch benchmark and links to the Model page.
- Kept existing navigation, themes, workflows, and styling intact; no unrelated provider or feature changes were introduced.
- Added compact mobile styling for the banner.

## Backend and model — current state and required verification
This revision does not retrain or alter saved model weights. Existing backend/model architecture is retained. The project contains stage-specific artifacts and synthetic benchmark metadata. Before production claims:
1. Verify every inference endpoint records the model version, prediction stage, data provenance, timestamp, and score meaning.
2. Verify stage-specific feature allowlists and reject features unavailable at prediction time.
3. Audit target-generation code and all features for any deterministic shortcut to the generated label.
4. Confirm synthetic benchmark training/validation/test rows are disjoint and test data was not used for tuning or threshold selection.
5. Re-run the training script and compare the resulting metrics against `models/synthetic_operational/metrics.json`.
6. Run the full automated suite and clean Windows launcher test. This archive update alone does not certify either.
7. Test malformed uploads, missing fields, unseen categories, empty files, oversized batches, database failures, and optional-provider failures.
8. Confirm action-board activity log persists across application restarts.

## Round 6 (interface)
See ROUND6_CHANGES.md.

## Frontend — recommended follow-up work
- Model page: show data origin, stage, holdout size/date range, ROC-AUC, PR-AUC, positive rate, and clear comparability warnings for every metric.
- Risk queue: distinguish risk ranking from calibrated probability; retain clear empty/error/loading states.
- Risk explanations: explain that feature drivers are model associations, not confirmed causes.
- Action board: validate transitions/notes and make owner and activity history obvious.
- Briefing: show source, period, sample size, and provenance next to each insight.
- Alert planner: show the number of orders investigated alongside estimated late orders caught and false-alert burden.
- Batch scoring/adaptation: report row-level validation errors and output provenance.
- Accessibility: verify keyboard operation, focus visibility, contrast, responsive layouts, and reduced-motion mode.

## Backend — recommended follow-up work
- Validate request schemas, field ranges, upload size, row counts, and CSV schemas.
- Enforce explicit stage-specific feature allowlists.
- Save reproducible evaluation metadata, feature list, split dates, thresholds, seeds, data hashes, and model version.
- Maintain leakage tests that cut off data at prediction timestamps.
- Separate ranking scores from calibrated probabilities and disclose calibration quality.
- Ensure ensemble explanations are accurately described if they use one member model's SHAP values.
- Keep optional LightGBM import failures from blocking normal inference; show actionable messages only when retraining that model is requested.
- Add end-to-end API/UI smoke tests, database persistence tests, and clean-install Windows verification.
- Add drift monitoring, retraining criteria, rollback, and operational outcome logging before pilot.

## Recommended viva/presentation wording
“The checkout-time model achieved ROC-AUC 0.717 on a chronological holdout of 19,294 historical Olist orders. The 0.804 and 0.809 figures are from a different random-split experiment and are not directly comparable. Because Olist does not provide the operational feeds needed for dispatch-stage evaluation, we generated a separate Olist-adapted dataset with simulated operational signals and generated labels. The dispatch model achieved ROC-AUC 0.9433 on a held-out synthetic test set. This is a benchmark result, not evidence of 0.9433 performance on real delivery operations. Validation on real dispatch and tracking feeds is the next step.”

## Submission checklist — still needs the team's action
- (Done in v6 update 2) Submission date set to 27/10/2026.
- Attach the genuine institute plagiarism/AI-detection report; do not fabricate it.
- Obtain formal internal mentor approval.
- Verify the front page lists only the internal mentor and the certificate lists internal and external mentors with signature space.
- Open the report and deck in Word/PowerPoint and inspect pagination, charts, fonts, and layout.
- Run tests on the target Windows machine and retain actual test output.


## v6 audit note — what the 0.9433 benchmark does and does not show
- `train_synthetic_operational.py` draws `synthetic_late_delivery` from a logistic formula over the same 16 dispatch signals the model is trained on. The model is therefore learning the generator's own rule; the ceiling is set by the simulator's noise, not by real delivery behaviour. There is no timestamp/status leakage, but the result is circular by design.
- The script also stops with an error unless test ROC-AUC exceeds 0.91, i.e. the benchmark was built to clear a target. State this openly in the report and viva.
- The only real-data number for the project is **0.717** (chronological, checkout stage). 0.9433 may be quoted only as "pipeline works on a generated benchmark".

## v6 code changes (verified by the automated suite)
- Every scoring response (`/api/score`, `/api/score/batch`, `/api/score/handoff`, `/api/score/tracking`) now includes a `provenance` block: model, stage, data origin, score meaning, synthetic flag.
- Batch CSV downloads include model/stage/data_origin/synthetic_benchmark columns.
- Upload caps: batch CSV 10 MB (`ONTIME_MAX_BATCH_BYTES`), adapt upload/score 40 MB; oversized files return HTTP 413 without being fully read.
- `/api/score` rejects nested/oversized `fields`; handoff/tracking request fields have length limits.
- Model page: explicit generated-data warning on the dispatch benchmark; simulator/batch show a provenance note and a "model associations, not causes" note.
- New tests: `tests/test_v6_provenance_limits.py`.


## v6 update 2 (27 Oct 2026) — implemented in this build
- **Submission documents** (`report/OnTime_Report_ZeepleWork.docx`, `ppt/OnTime_Presentation.pptx`, `ppt/OnTime_Poster.pptx`): front page and poster identify Dr. Bela Joglekar as internal mentor; the certificate includes Dr. Bela Joglekar and external mentors Mr. Sanchit Mehrotra and Mr. Chetan Bais with signature lines. Model evidence is labelled by real, simulated and generated provenance. The report contains a plagiarism/AI-detection placeholder that must be completed before submission.
- **New `backend/app/monitoring.py` + endpoints** `GET /api/monitoring/summary`, `POST /api/monitoring/drift`, `POST /api/monitoring/outcomes`: bootstrap 95% intervals (ROC-AUC, PR-AUC, precision/recall at the alert budget), reliability bins/ECE/Brier, monthly stability, PSI drift (first vs second half of the test period, and any uploaded batch), outcome tracking against real labels, retraining criteria and rollback note.
- **Model page** shows these checks; **batch scoring** lists rejected rows with reasons (`row_errors`, `error_reasons`).
- Measured on the held-out real queue: ROC-AUC 0.717 (95% 0.704–0.730); precision 10.0% (8.9–10.7%); recall 45.3% (42.5–48.3%); calibrated-ensemble ECE 0.037 but Brier 0.0534 vs 0.0501 base-rate baseline (not better than guessing the base rate); fixed-model monthly ROC-AUC 0.76 (Jun), 0.67 (Jul), 0.56 (Aug 2018).
- Tests: `tests/test_v6_monitoring.py`.

## Still genuinely open
- The plagiarism/AI-detection report placeholder in the report (page after References) must be replaced with the institute's real report; mentor approval and signatures; confirm page numbers in Word (the contents page was set from a LibreOffice render).
- Correction: a rolling *retraining* backtest already exists (expanding window, retrained every month, 17 test months Apr 2017 to Aug 2018; mean checkout ROC-AUC 0.673 for the original XGBoost, 0.68 for the blend; Jun to Aug 2018 enriched model 0.77 / 0.72 / 0.78). Scripts: `rolling_backtest_core.py`, `rolling_backtest_enriched.py`; results: `data/processed/rolling_backtest_*.csv`; shown on the Model page. The separate *fixed-model* month view added in the monitoring work (June 0.76, July 0.67, August 0.56) is a drift check and is not a retraining backtest. It could not be re-run in the last revision because xgboost was not installed in the build sandbox.
- Real operational validation and a clean Windows launch test remain to be done.


## Final integrated update (26 October 2026)
- The default live scorer is now the **real-Olist enriched checkout-time model**; the synthetic dispatch model is opt-in for demonstrations. This avoids presenting a generated-label model as the default operational predictor.
- Fresh installs and legacy saved `auto` settings use `enriched`; explicit saved model choices are preserved.
- All nine user-uploaded raw Olist CSVs were hash-compared with `data/raw/` and match byte-for-byte. `scripts/audit_olist_data.py` creates a reproducible audit report.
- The leakage audit passed at four temporal cut dates.
- Stage comparisons now explicitly include the 17-month handoff mean (0.771) and seven-month in-transit means (0.767 without simulated tracking; 0.869 with simulated tracking).
- The report, presentation and poster are stored in `report/` and `ppt/`; the report and poster mentor name is Dr. Bela Joglekar. The poster conclusion text was shortened and visually rendered to confirm it no longer overflows.
- See `DELIVERY_DOCUMENTATION/INTEGRATED_FINAL_VALIDATION.md` for test counts, data checks, smoke tests and remaining limitations.

## Final round (2026-10-26)
- Assistant engine now defaults to **Auto** (cloud AI if a key is set, else Ollama if running, else built-in; falls back by itself). Existing saved settings are respected.
- Settings: changing the scoring model or the assistant engine now saves immediately and refreshes the app's model label (before, it only applied after pressing Save lower on the page). Choices: Automatic, Tuned XGBoost, Weighted ensemble, Enriched chronological model (default), Dispatch operational model (benchmark, opt-in).
- Top banner shortened: the generated-benchmark figure is no longer in the page header; it stays on the Model page, labelled as generated. Talk through it in the viva.
- Documents live in `submission_documents/` (names and PRNs of all three members, mentor Pradnya Patil, date 27/10/2026).
- The dispatch benchmark figure stays **0.9433** (0.94325 in `models/synthetic_operational/metrics.json`). It was not changed to 0.9344, because that value does not match any measured result; changing a reported metric by hand would misstate it. If the model is retrained and measures 0.9344, update it from `metrics.json`.
- Not run in this environment (no network, FastAPI/XGBoost not installed): the test suite and the Windows launcher. Run `python -m pytest -q` before pushing.

## Document check (2026-10-26)
- Poster: the group-members box held the three names in an invalid position inside the paragraph (after the end-of-paragraph tag), so PowerPoint could drop them, and the line was too wide for the box. Rebuilt the paragraph; names and PRNs now show at 32 pt inside the box.
- Report fact-check against the project files (`dashboard/*.json`, `models/*_meta.json`, `backend/app_data/queue.csv`): headline results, alert-size table, risk bands, monthly figures, random-split tables, ensemble weights, backtest ranges, feature counts (33 + 37 = 70) and delay patterns all match. Corrected: calibration method is isotonic (not sigmoid), served-model calibration is ECE 0.050 and Brier 0.056 against 0.050 for the base rate; test count 220 to 233; feature-count wording; assistant Auto engine in the pages table.
- Deck: same two calibration corrections (slides 10 and 14).
