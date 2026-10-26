# OnTime: deployment smoke tests and real-world outcome validation

This guide separates three kinds of evidence:

1. **Automated/unit tests** check code behavior using fixtures and mocked services.
2. **Deployment smoke tests** check that a running app serves the frontend and that its real HTTP API paths work.
3. **Real-world outcome validation** checks predictions against actual delivery outcomes collected after those predictions were made.

Passing one level does not imply the others have passed.

## 1. Local test and deployment smoke test

From the project root, install `requirements-app.txt`, then run:

```bash
python -m pytest tests -q
python run.py
```

In another terminal, run the HTTP smoke test against the running server:

```bash
python scripts/http_smoke_test.py http://127.0.0.1:8000
```

The smoke test covers the frontend, health/readiness endpoints, model metadata, summary, risk queue, single-order inference, batch template and batch inference. It is not a load test or security assessment.

`GET /api/health/diagnostics` separates local readiness from unverified dependencies. It intentionally does not call external services on every health check. To test the live weather integration explicitly, request (with internet access):

```text
GET /api/weather/forecast?latitude=-23.55&longitude=-46.63
```

A `503` response means the provider could not be reached from that deployment at that time; OnTime must not silently replace live weather with synthetic values.

## 2. Capture prediction-time outputs

For a batch, use `GET /api/template.csv` to see the expected input layout, then call `POST /api/score/batch` with your operational input CSV. Each output row now contains `order_id`, `probability`, `model`, `status`, and `scored_at`. Keep the original request file and output unchanged so you can audit which features were available at prediction time.

Only use a scoring model with the feature schema it expects. The synthetic dispatch model defaults some operational inputs if they are absent; its output explicitly warns about estimates and is not evidence of real-world accuracy. For a real trial, supply trustworthy values observed at dispatch time and record which fields were estimated.

## 3. Validate against observed outcomes

After the delivery outcomes are known, create a separate outcomes CSV keyed by `order_id`. It must contain either:

- `is_delayed` with values `0/1`, `true/false`, `yes/no`, `late/on_time`; or
- both `delivered_at` and `promised_by` timestamps, from which delay is defined as `delivered_at > promised_by`.

The predictions CSV must contain `order_id`, `probability` and `scored_at`. Join only by stable order ID; do not add outcomes to the model inputs. Predictions must have been recorded before the outcomes were known. Do not use the synthetic training/benchmark CSV as real-world evidence.

Run:

```bash
python scripts/validate_operational_outcomes.py \
  --predictions predictions.csv \
  --outcomes observed_outcomes.csv \
  --threshold 0.5 \
  --min-rows 200 \
  --out operational_validation_report.json
```

The script reports join coverage, dropped invalid scores, class balance, ROC-AUC, average precision/PR-AUC, Brier score, precision, recall, F1, alert volume, and a chronological last-20% holdout where timestamps permit. Duplicate IDs and malformed labels/timestamps are rejected instead of silently reconciled. The default minimum is 200 matched rows and the report must contain both classes and a valid chronological holdout to be marked `ELIGIBLE_FOR_HUMAN_REVIEW`.

**`ELIGIBLE_FOR_HUMAN_REVIEW` is not production approval.** A responsible reviewer must also inspect calibration, performance by route/carrier/region/time, missingness and drift, costs of false positives/false negatives, and whether interventions actually improve outcomes. Demonstrating intervention benefit requires a controlled or otherwise defensible comparison of actions versus no action; prediction metrics alone cannot establish that.

## 4. Deployment validation checklist

- [ ] Run the full test suite in the deployment's Python/OS environment.
- [ ] Run `scripts/http_smoke_test.py` against the started service.
- [ ] Confirm the active scoring model and schema in `/api/meta`, `/api/models`, and `/api/model`.
- [ ] Test the weather endpoint from the deployment network if weather is part of the workflow.
- [ ] Capture timestamped predictions for real orders before outcomes are known.
- [ ] Collect observed delivery/promise timestamps from the operational source of truth.
- [ ] Run chronological outcome validation and review the JSON report.
- [ ] Evaluate by relevant operational segments and monitor drift before expanding rollout.
- [ ] Start with a monitored pilot; do not make autonomous customer or carrier decisions solely from a model score.

## Current evidence boundary

The packaged project contains historical Olist data and a synthetic dispatch benchmark. The former supports historical checkout-time analysis; the latter is a simulated dispatch task. Neither substitutes for a prospective, labelled operational pilot using real dispatch telemetry and observed outcomes. External weather connectivity depends on the deployment network and must be checked there.
