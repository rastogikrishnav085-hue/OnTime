# OnTime v5 — Frontend/backend synchronization report

Date: 2026-10-24

## Changes
- The Model page now displays the model configured for live single-order and batch scoring, using the same live model key/label returned by backend metadata.
- The public `/api/model` payload is focused on live scoring configuration and the selected model's matching benchmark metadata. It no longer mixes historical checkout-model evaluation tables into the active Model-page payload.
- `/api/health`, `/api/meta`, `/api/models`, and `/api/model` are regression-tested to return consistent live-scoring model identity and availability.
- The Batch page identifies the model used for the upload and labels synthetic benchmark scores appropriately.
- The About page describes prediction stages and simulated inputs without presenting outdated metric values as the active live-model result.
- The assistant's model-quality answer follows the live Settings selection and explains the synthetic benchmark scope.
- Frontend JavaScript is syntax-checked, and its script URL includes a cache-busting version.

## Verification
- Python compile check: passed.
- Frontend JavaScript syntax check (`node --check frontend/assets/app.js`): passed.
- Full test suite: 205 passed, 2 compatibility warnings.
- The two warnings concern loading an older XGBoost pickle in a different XGBoost version. The portable JSON model remains the inference artifact.
- Synthetic model reload/test reproducibility is covered by the existing production integration test: chronological synthetic ROC-AUC above 0.91.
- Original Olist raw CSV hashes remain unchanged.
- Historical queue and dispatch-time live scoring remain separate; missing dispatch telemetry is not fabricated into historical records.

## Scope
The synthetic dispatch benchmark is simulated and its metric is not a claim of measured real-world performance. Live weather is fetched through the weather endpoint when requested; it is not silently substituted with synthetic data if the provider is unavailable.
