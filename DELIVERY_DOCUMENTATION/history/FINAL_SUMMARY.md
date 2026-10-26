> Historical note: this summary describes an earlier build. The current default is the enriched real-Olist checkout-time model; the synthetic dispatch model is opt-in.

# OnTime v5 — Current Production Integration Summary

**Verification date:** 24 October 2026

## Current live-scoring workflow

- Fresh installations default single-order and batch scoring to `synthetic_operational`, a portable XGBoost dispatch-time model.
- The selected live model is reported by `/api/meta` and `/api/model`; Settings, the Model page, single scoring and batch scoring use the same model key and label.
- Model responses include the actual model identifier, threshold, risk ranking, explanations, warnings and a list of operational inputs that were estimated.
- The Model page presents the configured live model and its matching benchmark metadata. Historical evaluation metrics are not included in the active Model-page payload.
- The historical risk queue remains a separate view based on stored historical features. Dispatch-time telemetry is not fabricated to rescore records that do not contain it.
- `Adapt to your data` remains on checkout-time feature schemas and rejects the dispatch-only model for incompatible training tasks.

## Synthetic operational dataset and model

- Separate raw CSV: `data/raw/synthetic_operational_orders.csv`.
- 80,000 rows and 38 columns; the nine original Olist CSVs remain unchanged.
- 56,000 training rows, 12,000 validation rows and 12,000 untouched chronological synthetic test rows.
- Held-out synthetic ROC-AUC: **0.9433**; PR-AUC: **0.8927**.
- Feature importance includes XGBoost gain and held-out permutation importance.
- Weather severity, congestion, carrier reliability, backlog, route disruptions and target labels are simulated. These metrics describe the synthetic task and do not establish real-world delivery performance.

## Data integrity, API and UI checks

- The model loads from portable XGBoost JSON.
- `/api/score` and `/api/score/batch` use the same configured model unless an explicit request override is supplied.
- `/api/model` now returns the live scoring configuration and matching benchmark information for the active Model page, not the legacy model-comparison payload.
- `/api/meta` exposes the effective live-scoring model key, label and availability.
- The Model page and batch results identify the model used. The assistant's model-quality response follows the current live scoring setting.
- Live weather remains a separate Open-Meteo lookup. It provides operational context but is not inserted into historical Olist rows or silently replaced with simulated weather when unavailable.
- Original Olist CSV hashes match the packaged raw-data manifest.

## Tests

- Full suite: **210 passed, 2 warnings**.
- A live local HTTP smoke test passed 11 checks across frontend, health/readiness, metadata, queue, single scoring, and batch scoring.
- `/api/health/diagnostics` now reports external weather connectivity as not checked and real-world outcome validation as not configured until observed labels are supplied.
- Single and batch scoring responses carry UTC `scored_at` timestamps to support prediction-time auditing.
- `scripts/validate_operational_outcomes.py` evaluates timestamped predictions against separate observed delivery outcomes using a chronological holdout. No real operational outcomes were provided, so real-world accuracy remains unvalidated.
- Weather provider connectivity could not be verified from this build environment; Docker is unavailable here, though CI now runs the HTTP smoke test inside the built container.
- The two warnings concern a legacy XGBoost pickle compatibility test. The dispatch model is loaded from portable JSON.

## Reproduction

From the project root:

```powershell
python train_synthetic_operational.py
python -m pytest tests -q
```

For dataset provenance and model scope, see `SYNTHETIC_BENCHMARK_README.md` and `data/raw/SYNTHETIC_OPERATIONAL_DATA_DICTIONARY.md`.
