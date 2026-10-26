# OnTime current model policy (2026-10-26)

## Default behavior

Fresh installs use `enriched`, the real-Olist checkout-time model, for single-order and batch scoring. The synthetic dispatch model is available only when deliberately selected for demonstration. The historical risk queue remains a distinct checkout-time view.

## Results and evidence boundaries

| Experiment | Result | Evidence type | How to describe it |
|---|---:|---|---|
| Checkout-time risk queue | ROC-AUC 0.71679; PR-AUC 0.10244 | Real Olist, newest 19,294 orders in chronological holdout | Primary headline result |
| Checkout random split | ROC-AUC 0.804–0.809 | Real Olist, random split | Secondary diagnostic, not comparable to chronological result |
| After carrier handoff | ROC-AUC 0.85243 random split; 0.77102 mean over 17 monthly tests | Real Olist-derived handoff stage | Use monthly mean as the more realistic temporal summary |
| In transit, without simulated tracking | ROC-AUC 0.84000 held-out snapshot test; 0.76688 mean over 7 monthly tests | Olist-derived records and time features | Historical snapshot experiment; not a live prospective trial |
| In transit, with simulated tracking/weather/strike signals | ROC-AUC 0.91460 held-out snapshot test; 0.86866 mean over 7 monthly tests | Simulated signals | Simulation only; do not call real-world accuracy |
| Dispatch benchmark | ROC-AUC 0.94325 (rounded 0.9433); PR-AUC 0.89270 | Generated rows and generated labels | Synthetic benchmark only; labels are generated from simulated inputs |

ROC-AUC measures ranking discrimination, not percentage accuracy. Scores from different prediction stages, splits, and data provenance must not be compared as if they were one common test.

## What is and is not validated

- The nine original Olist CSVs are preserved unchanged in `data/raw/` and match the nine uploaded source CSVs byte-for-byte.
- The adversarial feature leakage audit passed at four cut dates for the checkout-time features.
- The historical Olist evaluation is retrospective. It is not a prospective operational pilot.
- The project has no verified carrier GPS scans, hub telemetry, dispatch events or observed live weather feeds for these historical orders.
- No claim is made that ROC-AUC above 0.85 has been achieved on real future dispatch orders. To validate that, capture predictions before outcomes, join to genuine observed outcomes, and run `scripts/validate_operational_outcomes.py`.
