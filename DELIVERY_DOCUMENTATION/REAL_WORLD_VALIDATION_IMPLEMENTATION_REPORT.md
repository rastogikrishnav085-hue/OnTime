# OnTime v5 — Operational validation and end-to-end test update

Date: 2026-10-24

## Implemented

- Added `GET /api/health/diagnostics` to distinguish local application readiness from untested weather connectivity and the absence of real-world labelled outcomes.
- Added UTC `scored_at` timestamps and `order_id` to single-order scoring responses; batch scoring returns a shared UTC timestamp both at response level and per output row. This provides a prediction-time audit field for later outcome matching.
- Added `scripts/http_smoke_test.py`, which tests the running service over HTTP: frontend, JavaScript, health, diagnostics, model metadata, model listing, summary, queue, single scoring, batch template and batch scoring.
- Added `scripts/validate_operational_outcomes.py` to join timestamped predictions with a separate observed-outcomes file and report coverage, invalid scores, ROC-AUC, average precision, Brier score, precision, recall, F1, alert counts and a chronological last-20% holdout.
- The outcome validator rejects duplicate order IDs and malformed labels/timestamps, never creates labels, and does not automatically approve a model for production. Reports are marked `ELIGIBLE_FOR_HUMAN_REVIEW` only when the row minimum, both classes and chronological holdout requirements are met.
- Added a guide with run commands, expected schemas, deployment checks and limitations.
- Added automated tests and wired the deployment HTTP smoke test into Docker CI.

## Verification results

- Full automated test suite: **210 passed, 2 warnings**.
- Python compilation for backend, scripts and tests: passed.
- Live local HTTP smoke test: **11 checks passed** against a running OnTime server.
- The HTTP smoke test exercised the frontend, model metadata, queue, single scoring, batch template and batch scoring.
- Weather provider check: attempted from this build environment and failed with `URLError`; the endpoint correctly does not substitute synthetic weather. Re-run the weather check from the actual deployment network.
- Docker is not installed in this build environment, so the container image itself could not be built here. CI is configured to run the HTTP smoke test inside the built container.

## What is and is not fixed

The software now has a repeatable way to run end-to-end HTTP checks and a defensible workflow to evaluate timestamped predictions against later observed outcomes. The synthetic dispatch model remains a simulated benchmark, not a measured real-world model. No real operational outcomes file was supplied/configured, so no real-world accuracy claim can yet be made. That requires genuine observed delivery outcomes and a prospective/chronological evaluation in the operating environment.

## Remaining compatibility note

Two existing warnings remain when tests compare a legacy XGBoost pickle created under a different XGBoost version. The portable JSON inference artifact is retained and is the intended inference path.
