# Production live-scoring model switch

## What changed
- Live single-order scoring and batch scoring now support `synthetic_operational`, an XGBoost dispatch-checkpoint model trained from `data/raw/synthetic_operational_orders.csv`. It is an opt-in demonstration setting; fresh installations now default to the enriched real-Olist checkout-time model because the dispatch benchmark uses generated signals and labels.
- The simulator exposes dispatch-time features. The scorer uses supplied values when present and otherwise derives conservative defaults from checkout fields and lists every default/estimate in the response.
- The historical risk queue remains scored by the existing real-Olist chronological model because its records are checkout-time/historical and do not contain the dispatch-time feature schema. This avoids falsely treating simulated dispatch features as observed historical facts.

## Important limitation
The synthetic operational model achieved ROC-AUC above 0.91 on its untouched synthetic chronological test set. The weather/congestion/capacity signals and target labels are simulated. This metric is not real-world validation and its output is deliberately labelled `SYNTHETIC_BENCHMARK`; it must not be presented as a calibrated real-world probability. Use observed dispatch telemetry and validate on labelled production outcomes before relying on it for operational decisions.

## Verification
Run `python -m pytest -q`. The model artifact is loaded at engine startup, checked by a prediction smoke test, and surfaced as an explicit model choice in Settings and Score an order.
