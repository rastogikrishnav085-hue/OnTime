# OnTime v5 — Production Scoring Architecture

## Active live scoring

The current live single-order and batch workflow uses the model selected in Settings. Fresh installations default to the `enriched` real-Olist checkout-time model. The `synthetic_operational` dispatch-time XGBoost model is opt-in for demonstrations only. This is deliberate: the operational benchmark is trained on generated labels and cannot substantiate real-world dispatch performance. Its portable model artifact is stored at `models/synthetic_operational/xgboost_synthetic_operational.json`.

The frontend reads the effective model key, label and availability from `/api/meta`, and the Model page reads benchmark information from `/api/model`. The backend includes the model key and label in scoring results. This keeps the Settings page, Model page, single-order scoring, batch scoring and assistant aligned.

## Synthetic dispatch benchmark

The separate raw CSV is `data/raw/synthetic_operational_orders.csv` (80,000 rows, 38 columns). It is adapted to the existing Olist order, item, customer, seller and product tables. The original nine Olist source CSVs remain unchanged.

| Partition | Rows |
|---|---:|
| Training | 56,000 |
| Validation | 12,000 |
| Untouched chronological test | 12,000 |

Held-out synthetic benchmark ROC-AUC is **0.9433** and PR-AUC is **0.8927**. The saved JSON model is reloaded and independently evaluated by `tests/test_synthetic_operational_production.py`. Feature importance is reported using both XGBoost gain and held-out permutation importance.

Operational signals and labels are simulated. The benchmark reflects the synthetic task only and must not be described as measured performance on real delivery traffic. Synthetic scores are marked `SYNTHETIC_BENCHMARK`, are not calibrated probabilities, and may include estimated/defaulted dispatch inputs if actual values are not supplied.

## Separate historical queue

The historical risk queue is a distinct product view built from stored order features. It is not silently re-scored using invented dispatch telemetry. The Adapt-to-data workflow continues to use checkout-time model schemas; dispatch-only models are rejected for incompatible adaptation tasks.

## API contract

- `GET /api/meta`: effective live-scoring model key, label, availability and feature metadata.
- `GET /api/model`: current live model configuration and its matching benchmark fields for the Model page.
- `POST /api/score`: single-order scoring; an explicit model override is optional.
- `POST /api/score/batch`: batch scoring; uses the same configured model unless overridden.
- `GET /api/weather/forecast`: live Open-Meteo operational context; not historical training data.

## Verification

The latest full project suite passed **204 tests**, with two existing warnings from the legacy XGBoost pickle compatibility test. The production dispatch model uses portable XGBoost JSON.
