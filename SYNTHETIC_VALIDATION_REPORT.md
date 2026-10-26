# Synthetic Dataset and Model Verification

## Dataset placement and provenance
- Separate CSV: `data/raw/synthetic_operational_orders.csv`
- Rows / columns: 80,000 / 38
- Generated using the existing Olist orders, items, customers, sellers, and products CSVs as sources for entity IDs and observed distributions of price, freight, product category, product weight, city/state and related fields.
- The nine original Olist CSVs remain separate and their manifest hashes are unchanged. Synthetic order IDs use the `SYNTH_` prefix and do not collide with observed order IDs.
- Weather, congestion, route disruption, carrier reliability, facility backlog, dispatch lag, and the target are simulated. This is not observed ground truth.

## Model training and held-out evaluation
| Measure | Result |
|---|---:|
| Training rows | 56,000 |
| Validation rows | 12,000 |
| Untouched chronological test rows | 12,000 |
| Validation ROC-AUC | 0.9362 |
| **Held-out chronological synthetic ROC-AUC** | **0.9433** |
| Held-out chronological synthetic PR-AUC | 0.8927 |
| Test precision at 0.5 | 0.8248 |
| Test recall at 0.5 | 0.7789 |
| Test F1 at 0.5 | 0.8012 |

## Data and leakage checks
- 80,000 rows; 38 columns.
- Zero duplicate synthetic order IDs; all IDs have the `SYNTH_` prefix.
- Target is binary (`synthetic_late_delivery` in {0,1}).
- The 16 input features are listed in `models/synthetic_operational/metrics.json`; target, order/customer/seller/product IDs, raw statuses, raw timestamps and customer delivery timestamp are excluded from model features.
- The `order_delivered_customer_date` field is blank for all synthetic rows.
- Saved JSON model was reloaded and independently scored; the ROC-AUC exactly reproduced the report.
- `feature_importance.csv` includes XGBoost gain and held-out test permutation importance for all 16 features.
- Generator rerun completed successfully and reproduces the dataset checksum.
- Full project tests: **205 passed, 2 compatibility warnings** (legacy pickle/XGBoost version warnings in an existing compatibility test).

## Interpretation limit
The >0.91 threshold is achieved for the synthetic dispatch-time benchmark only. The model learns relationships intentionally embedded in simulated pre-dispatch signals and simulated labels. It is wired into live single-order and batch dispatch scoring, clearly labelled `SYNTHETIC_BENCHMARK`; that integration does **not** establish real-world performance. The historical queue remains a separate view based on its stored historical features.
