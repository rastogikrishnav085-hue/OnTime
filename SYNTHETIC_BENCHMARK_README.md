# Synthetic operational model — Olist-adapted and production-wired for dispatch scoring

The separate dataset is stored at `data/raw/synthetic_operational_orders.csv`. It contains 80,000 rows and preserves Olist-style order/item/customer/seller/product columns while adding dispatch-time operational risk signals. Entity/value distributions are sampled from the project's Olist CSVs; weather, congestion, capacity, carrier reliability, disruption signals and target labels are simulated. See `data/raw/SYNTHETIC_OPERATIONAL_DATA_DICTIONARY.md`.

## Training and held-out result
Run `python train_synthetic_operational.py` from project root to regenerate the CSV and train XGBoost. The script writes the portable model, metrics and feature importance, and fails if the chronological test ROC-AUC does not exceed 0.91.

- Training: 56,000 rows
- Validation: 12,000 rows
- Untouched chronological test: 12,000 rows
- Held-out synthetic ROC-AUC: **0.9432527722**
- Held-out synthetic PR-AUC: **0.8927009432**
- Feature importance: `models/synthetic_operational/feature_importance.csv` (model gain and held-out permutation importance)

## Production wiring
`backend/app/engine.py` loads this model and exposes it as `synthetic_operational`. The dispatch-checkpoint model is an opt-in demonstration. Fresh installations default `Score an order` and `Batch scoring` to the enriched real-Olist checkout-time model. The simulator includes operational inputs, validates numeric ranges, and reports estimated/defaulted inputs. Batch scoring accepts either the checkout CSV template or a CSV containing the complete 16-feature dispatch schema. The model is loaded from portable XGBoost JSON.

The historical risk queue remains a separate view based on recorded order features because those historical rows do not include actual dispatch-time telemetry. `Adapt to your data` continues to use checkout-time models.

## Scope warning
The 0.9433 ROC-AUC applies only to this synthetic dispatch-time benchmark. The target is generated from simulated operational signals, so this does not establish real-world performance. Scores are marked `SYNTHETIC_BENCHMARK` and are not calibrated probabilities. Supply actual dispatch telemetry and evaluate on later real labelled outcomes before relying on alerts.
