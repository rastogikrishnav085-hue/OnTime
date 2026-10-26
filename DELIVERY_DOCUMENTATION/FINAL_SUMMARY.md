# OnTime v5 — Final Verification Summary

**Verification date:** 24 October 2026  
**Data source:** The nine original Brazilian Olist CSV files in `data/raw/`; synthetic tracking/weather files are excluded from checkout-model training.

## Training/evaluation result

- Prediction cutoff: order purchase time.
- Chronological split: oldest 65% training (62,705 orders), next 15% validation (14,471), newest 20% untouched test (19,294).
- Test period: 2018-05-26 18:18:03 through 2018-08-29 15:00:37; test late rate 5.2918%.
- Selected model: enriched XGBoost, 70 features (33 core + 37 enriched), trained on real Olist only.
- Untouched chronological test: ROC-AUC 0.71679; PR-AUC 0.10244; precision 0.09974; recall 0.45348; F1 0.16352 at the validation-selected threshold; recall among top 10% highest-risk orders 0.18707.
- Comparison: core ensemble ROC-AUC 0.59071 on the same test rows. Paired difference +0.12569, bootstrap 95% CI [+0.11077, +0.14175].
- Rolling monthly test blocks: enriched model mean ROC-AUC 0.75458 across three evaluated months; core XGBoost 0.70695; enriched model won 2/3 months.
- Target ROC-AUC > 0.90: **not achieved**. Do not state otherwise. Temporal distribution shift and limited operational signals are major constraints.
- Calibration: newest-period drift is detected; calibrated chance display is withheld. Treat scores as rankings, not literal probabilities.

## Data integrity and leakage

- Rebuilt model-ready data from the original CSVs: 96,470 rows, 73 columns.
- Raw CSV hashes were unchanged by the build.
- Data validator: ACCEPT, 23 checks, 0 failures, 0 warnings.
- Leakage audit: 70 features checked at four cut dates; no outcome-dependent feature movement detected.
- Live weather: frontend and FastAPI endpoint use Open-Meteo for current/forecast context. Live weather is not training data and is not used to retrospectively score Olist orders. The provider request was unavailable from the build environment (network URLError); route success/error behavior is covered by mocked tests.
- Synthetic tracking/weather/congestion/strike files remain clearly separated from real Olist model features.

## Backend and tests

- Enriched model is registered in the model engine and selectable for single and batch scoring.
- Held-out queue remains the newest chronological test block; core XGBoost and ensemble comparison scores are included without replacing the enriched serving scores.
- Disagreement filter compares the served model to its actual comparison model.
- API now converts non-finite numeric values to JSON null instead of emitting invalid NaN tokens.
- Test suite: **200 passed** on the latest completed run (`python -m pytest -q --tb=short`). Two compatibility warnings came from a legacy XGBoost pickle test; the portable JSON booster is retained for cross-platform loading.

## Reproduction commands (from project root)

```powershell
.venv\Scripts\activate
python build_dataset.py
python audit_leakage.py
python backend/add_core_second_opinions.py
python -m pytest tests -q
```

For a new model training run, use the documented `train_enriched.py` and `evaluate_chronological.py` protocols; do not tune on the newest test block. After retraining, regenerate model metadata, chronological reports, and queue scores before packaging.
