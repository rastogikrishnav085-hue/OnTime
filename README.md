# OnTime v6 - Delivery Delay Early Warning for Logistics Teams

## Data note (read this before a demo or viva)

- **Real:** the nine Olist order CSVs and everything built on them (risk queue, delay patterns, the 0.717 chronological ROC-AUC).
- **Generated/simulated operational benchmark:** Olist does not provide the required carrier scans, observed weather/congestion, capacity/backlog, or complete dispatch telemetry. To prototype those later-stage workflows, we generated an Olist-adapted operational dataset. Its dispatch ROC-AUC is **0.9433 on a held-out synthetic test set**. This is a result on generated data, not observed real Olist outcomes.
- Suggested wording: *"The public Olist data did not contain the operational feeds needed to evaluate dispatch-stage prediction. We therefore generated a separate, reproducible Olist-adapted benchmark with simulated operational signals and labels. The dispatch experiment achieved ROC-AUC 0.9433 on its held-out synthetic test set. This demonstrates the prototype's performance on that benchmark; real-world performance remains unverified until evaluated on real dispatch and tracking feeds."*
- Moved: older fix and verification reports are in `DELIVERY_DOCUMENTATION/history/`.

> **Quick Start:** Double-click `run_all.bat` in this folder. On the first run it creates a private `.venv`, installs the packages, and then opens the app at `http://localhost:8000`. On later runs it checks the packages, repairs them if needed, and starts the app. To run the tests, double-click `run_tests.bat`. Gemini and Ollama are optional; add a key or start Ollama, then pick the engine in Settings.

---

# OnTime — Delivery Delay Early Warning for Logistics Teams

Mini project for Business Analytics (CA-3). Uses the Olist Brazilian
e-commerce dataset to flag orders at risk of missing their promised
delivery date, before the delivery actually happens.

## Which ROC-AUC is which (read this before quoting a number)

| Score | ROC-AUC | Data | How it was tested |
|---|---|---|---|
| **Risk queue (real orders, the one to quote)** | **0.717** | Real Olist | Trained on the oldest 65% of orders, tested on the newest 20% it never saw. Alerts: precision about 10%, recall about 45%, with 5.3% of orders late. |
| Original XGBoost / ensemble | 0.804 / 0.809 | Real Olist | Random 80/20 split. Optimistic, because past and future orders are mixed. Not comparable with 0.717. |
| After carrier handoff | 0.852 random split; 0.771 monthly mean | Real Olist | Random split is secondary; temporal backtest is more realistic. |
| In transit, with tracking | 0.915 held-out simulated-signal test; 0.869 monthly mean | Real order backbone + **simulated** tracking/weather/strike signals | Both are simulation results, not verified operational performance. |
| Dispatch operational model | **0.9433** | Generated Olist-adapted dataset; simulated dispatch signals and generated labels | Held-out synthetic benchmark only; not measured real-world accuracy. |

The benchmark goal of ROC-AUC above 0.90 is **met on the generated dispatch benchmark only**. On real Olist checkout-time data it was not reached; the chronological headline figure is 0.717. The Model page in the app shows the model results and labels the synthetic results separately from real-data evaluation.

## Data sources and citation

- **Real data:** Olist & Sionek (2018). *Brazilian E-Commerce Public Dataset by Olist* [Data set]. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce (about 100,000 orders, 2016 to 2018; licence CC BY-NC-SA 4.0, so non-commercial use with attribution). The nine CSVs are in `data/raw/` unchanged.
- **Generated, not real:** `data/raw/synthetic_operational_orders.csv` (80,000 rows, dispatch benchmark) and `data/synthetic/tracking_events_simulated.csv` (simulated carrier scans). Why they exist: Olist has no carrier scans, observed weather or congestion, or dispatch telemetry, so these stages were prototyped on generated data. Results on them show the pipeline works; they are not measured delivery accuracy. Definitions: `data/raw/SYNTHETIC_OPERATIONAL_DATA_DICTIONARY.md`.
- **Real-data result to quote:** 0.717 ROC-AUC (chronological test). Rolling monthly retraining backtest: mean about 0.68 over 17 months.

## Current live-scoring setup

The live single-order and batch scoring endpoints use the model selected in Settings. Fresh installations default to the **enriched real-Olist checkout-time model**. The separate dispatch operational XGBoost benchmark is opt-in for demonstrations and uses inputs such as route distance, simulated weather/congestion, carrier reliability, facility backlog, remaining delivery slack and tracking-scan age.

The synthetic benchmark is reproducible: 56,000 training rows, 12,000 validation rows and 12,000 untouched chronological test rows. Its held-out synthetic ROC-AUC is **0.9433** and PR-AUC is **0.8927**. These figures describe a simulated dispatch task, not measured real-world delivery performance. Synthetic operational signals and target labels are explicitly identified; the score is a prioritisation signal, not a calibrated probability.

The historical order queue remains a separate historical-data view because its rows do not contain actual dispatch-time telemetry. The application does not invent observed weather, carrier scans, road disruptions or hub congestion to backfill that history. Use the live scorer or batch API when dispatch-time operational values are available.

The Model page, Settings and backend metadata now share the configured live model key and label. `/api/score` and `/api/score/batch` use the same setting unless a request explicitly overrides it. The API returns model identity, thresholds, explanations and any estimated inputs.

The live weather lookup is connected to the frontend and FastAPI at `/api/weather/forecast` using Open-Meteo. It provides current conditions and a three-day forecast for supplied coordinates. Weather is operational context and is not inserted into historical Olist training rows; if the provider/network is unavailable, the endpoint returns an error rather than silently substituting synthetic weather.

The nine original Olist CSVs are included unchanged under `data/raw/`. Rebuild the reproducible feature table with `python build_dataset.py`; run the leakage audit with `python audit_leakage.py`; run tests with `python -m pytest tests -q`. For synthetic model training, run `python train_synthetic_operational.py`. See `SYNTHETIC_BENCHMARK_README.md`, `data/raw/SYNTHETIC_OPERATIONAL_DATA_DICTIONARY.md`, and `models/synthetic_operational/metrics.json` for provenance, features, test metrics and limitations.

## New in version 3

| What | Where |
|---|---|
| **Action board** with a stage guide, one clear button per card ("Start investigating", "Tell the customer", "Mark resolved"), owner filter, risk sorting | Action board |
| **Second score after the carrier collects the parcel** (ROC-AUC 0.85 on a random split, 0.77 month by month, against 0.81 and 0.68 at checkout). A separate score, never a replacement. | Order drawer, Model page; `train_handoff_model.py`, `backend/app/handoff.py` |
| **Third score for orders in transit, on SIMULATED tracking data** (carrier scans, failed delivery attempts, weather, strikes; invented for the same Olist orders). ROC-AUC 0.91 with it against 0.84 without, on the newest 20% of orders (carrier, hub congestion, road disruptions, weather, strikes and real public holidays added). These figures describe the simulation, not real Olist. | Order drawer, Model page; `make_tracking_dataset.py`, `train_tracking_model.py`, `src/tracking_features.py`, `backend/app/tracking.py`; data in `data/synthetic/` (clearly labelled simulated demo data) |
| **Alert-size table** (precision, recall, accuracy for the riskiest 2% to 30% of orders), next to the Alert planner | Model page |
| **Auto assistant engine**: cloud AI, else Ollama, else built-in, with automatic fallback | Settings, Ask OnTime |
| Clean white and black default themes; Insights and Alert planner pages kept | whole app |

Current live-scoring scope and limitations: `PRODUCTION_MODEL_SWITCH.md` and `SYNTHETIC_BENCHMARK_README.md`.

## New in version 2 (tuning, ensemble, literature, other data sets, smarter assistant)

| What | Where |
|---|---|
| **Weighted ensemble** of XGBoost, histogram gradient boosting and logistic regression (a random forest was tried and got zero weight). Served for the queue because it beat the single XGBoost on the held-out test orders (ROC-AUC 0.809 vs 0.804) **and** on a month-by-month retraining backtest (0.6799 vs 0.6615, ahead in 16 of 17 months). | Model page; `src/ensemble.py`, `train_ensemble.py`, notebook 09 |
| **Wider hyperparameter tuning**: 24 XGBoost and 10 histogram-boosting settings, 3-fold CV on training rows only, the old settings as trial 0. | `src/tuning.py` |
| **Calibrated "real chance of being late"** shown next to the score. The score ranks orders (class-balanced), so 0.7 is not 70%. Calibration error on unseen orders: 0.5 points vs 25 for the raw score. | order panel, simulator, Model page; `calibrate_ensemble.py` |
| **Second opinion** from the other model on every order, a "Models disagree" queue filter, and a model choice in Settings. | queue, order panel, Settings |
| **Adapt to your data**: upload a CSV with other column names, units and date formats; see what the model really received; check the transfer against real outcomes; optionally learn a model from your own history. | Adapt to your data page; `src/adapter.py` |
| **Literature page**: 25 sources in 9 themes, what each found and how OnTime uses it, with links and a filter. | Literature page; `backend/app_data/literature.json` |
| **Assistant**: remembers what the last answer was about ("What should we do about it?"), and answers questions on the ensemble, tuning, papers and other data sets. | Ask OnTime |
| **Accessibility and themes**: all eight themes meet WCAG AA text contrast (a guard test enforces it); theme changes fade smoothly. | `tests/test_theme_contrast.py` |

**To reproduce the ensemble** (about 20 minutes on one core): double-click `train_ensemble.bat`, or run
`.venv\Scripts\activate`, then `python train_ensemble.py`, `python backend/add_second_opinion.py`, `python calibrate_ensemble.py`.
No new packages are needed. If the run is interrupted, run it again: finished tuning trials are kept.

Full details, numbers and caveats: `MODEL_IMPROVEMENTS.md`.

## Problem

Logistics teams usually find out an order is late once it's already late.
OnTime scores every order at the point of purchase/approval and gives it
a delay-risk probability, a risk band (LOW / MEDIUM / HIGH / CRITICAL),
and a short list of the factors driving that score, so risky shipments
can be prioritised for intervention instead of discovered after the fact.

## Data

`olist_orders_dataset.csv` plus five supporting tables (items, customers,
sellers, products + category translation, payments, geolocation). See
`notebooks/01_data_understanding.ipynb` for the raw schema and
`notebooks/02_data_cleaning.ipynb` for how they're joined into one
order-level table.

## Quick start (web app)

**Windows: double-click `start_ontime.bat`.** The first time, it finds
Python, creates a private environment in `.venv`, installs the packages
(a few minutes), then starts the server and opens the app in your browser.
After that it starts in seconds. Keep its window open while you use the
app; close it to stop.

| Script (double-click) | What it does |
|---|---|
| `start_ontime.bat` | Sets up on first run, starts the app, opens the browser. `start_ontime.bat 8010` uses another port. |
| `setup.bat` | Just the setup (Python check, `.venv`, packages, `.env`). `setup.bat full` also installs Jupyter for the notebooks. |
| `run_tests.bat` | Runs the full test suite. |
| `score_orders.bat` | Drag a CSV of orders onto it; writes `<name>_scored.csv` next to it. Run it alone to get a template. |
| `rebuild_app_data.bat` | Re-scores the queue and rebuilds insights and the dashboard after re-running the notebooks. |
| `docker_start.bat` / `docker_stop.bat` | Run or stop the app in Docker Desktop instead. |
| `reset_board.bat` | Empties the action board (asks you to type YES first). |

You need Python 3.11 or newer from python.org (tick "Add python.exe to
PATH" while installing). On macOS or Linux, run `./start_ontime.sh`.

Manually, on any system:

```
pip install -r requirements.txt
python run.py
```

Open **http://localhost:8000**. Interactive API docs are at
http://localhost:8000/docs. With Docker instead:

```
docker compose up --build
```

The zip ships with the trained model and the prepared app data, so the app
runs straight away. The raw Olist CSVs are only needed to re-run the
notebooks (put them in `data/raw/`).

## Troubleshooting

| What you see | What it means / what to do |
|---|---|
| Console says *"Rebuilding the model locally…"* on first start | This computer couldn't read the saved model file, so OnTime retrains the identical model from the bundled data (about a minute, once). It is saved in `models/local_rebuild/` and reused after that. |
| `XGBoostError: input stream corrupted` (older versions) | The original joblib pickle doesn't load on some Windows builds of XGBoost. Fixed: the app now loads a portable copy (`models/ontime_xgb_booster.json` + `ontime_preprocessor.joblib`) and falls back to the local rebuild above. |
| *"Port 8000 is already in use"* | OnTime is probably already running; the browser opens it. Or run `start_ontime.bat 8010`. |
| *"Python 3.11 or newer was not found"* | Install Python from python.org with **Add python.exe to PATH** ticked, then run `start_ontime.bat` again. |
| Page says *"OnTime couldn't start"* | The message on the page and in the console says why. Usually re-extracting the zip into a fresh folder fixes missing files. |
| Project folder is inside **OneDrive** | Fine. The action board's database is then kept in `%LOCALAPPDATA%\OnTime\` so OneDrive syncing can't lock or corrupt it. |
| A CSV from Excel won't score | Byte-order marks, Windows (ANSI) encoding and semicolon separators are all handled; a rejected row says exactly which field is wrong. |

## What the app does

**Monitor**
| Page | What a coordinator does there |
|---|---|
| **Overview** | Today's risk at a glance, how reliable each risk band has been, the biggest delay patterns, the latest briefing, and the theme circles. |
| **Risk queue** | Every incoming order scored at checkout. Filter by band, state, category, delay pattern or focus categories; sort by risk or by *priority* (risk × category weight). |
| **Order detail** | Risk gauge, top reasons (TreeSHAP), suggested action, a written **story** of the order, the delay pattern it resembles, a simulated logistics view, a draft **customer message**, and what actually happened. |
| **Action board** | Shared Kanban (To review → Investigating → Customer informed → Resolved) with owners, notes and an activity log. |
| **Briefing** | The periodic check-in the mentor preferred over real-time alerts: a written monthly briefing (patterns, focus categories, critical orders, outcome check, next actions). Copy, download as Markdown, print, or rewrite for leadership. |

**Understand**
| Page | |
|---|---|
| **Delay patterns** | K-means groups the 7,826 late orders into 6 patterns with a shared cause (e.g. *Local last-mile, tight-promise orders to the South-East*). Each has a story, what sets it apart, lift-based states/products, and specific recommendations such as "set the minimum promise to 25 days on these routes". |
| **Focus categories** | Six product types where a late delivery matters most (health & beauty, baby, watches & gifts, computers, furniture, bed & bath), each with a criticality tier, impact, trend, worst destinations, pattern mix and a playbook checklist. |
| **Network (simulated)** | Simulated delivery model (direct / regional hub / hub-and-spoke), hub transfers, shipping mode, carriers and hubs, built from real fields by documented rules, plus a sensitivity test showing they add no model lift. |
| **Insights** | Late rate by state, category, distance, promise window, weekday; seller watchlist. |

**Act**
| Page | |
|---|---|
| **Score an order** | What-if simulator: change the route, promise, month, seller or basket and see risk, reasons, action, matching pattern and simulated logistics live. |
| **Batch scoring** | Upload a CSV (Excel-friendly), every row scored or rejected with a reason, download the results. |
| **Alert planner** | Enter the cost of a check and of a late delivery; see alerts, coverage and net saving at every threshold, and the cost-optimal threshold. |

**Help**: **Ask OnTime** (assistant), **Model**, **How it works** (incl. a table mapping each review suggestion to where it is answered, and the roadmap), **Settings**.

Eight colour themes (amber, emerald, ocean, paper white, each with a dark partner), works on a phone, and runs fully offline.

## Ask OnTime: three engines

| Engine | What it is | Needs |
|---|---|---|
| **Built-in** | Recognises the question (an order ID, a pattern, a state, a category, the model…) and answers from the app's own numbers. | Nothing. Offline, instant. |
| **Laptop model** | A local AI model through [Ollama](https://ollama.com). Nothing leaves your computer. | Install Ollama, then `ollama pull llama3.2` |
| **Cloud AI** | A hosted AI model through your own API key. | A free key from aistudio.google.com, pasted in Settings or set as `GEMINI_API_KEY` in `.env` |

Whatever the engine, the built-in engine first gathers the relevant **facts**; the AI engines only see those facts and the question and are told not to add anything, so they improve the wording without inventing numbers. Every AI call has a hard **time limit (5 s to 2 minutes, set in Settings)**; if the engine is missing, errors, or runs out of time, the built-in answer is shown with the reason. The same engines power the "Rewrite" buttons on stories, briefings and customer messages.

## What the review asked for

| Mentor's suggestion (review, 2 Oct 2026) | Where it is answered |
|---|---|
| Use e-commerce data but focus on 5-6 product types; a late healthcare product is not like a late cushion | Focus categories (tiers, impact, playbooks, priority sort) |
| Form clusters of delayed orders sharing a pattern (vendor, area, product) and understand why | Delay patterns |
| Don't just surface problems; suggest how to solve them | Pattern recommendations, per-order suggested actions, playbooks |
| Narrative reporting for a cluster or a single record | Stories for orders, patterns, categories, monthly briefings |
| Consider delivery models (hub-and-spoke…); simulate data you don't have | Network (simulated) + sensitivity test |
| A periodic dashboard check is more practical than real-time alerting | Briefing |

## GitHub

Double-click `github_push.bat` (or run the usual `git init / add / commit / push`). The repo includes:
- `.github/workflows/ci.yml`: runs the test suite on **Windows and Linux** for every push, then builds the Docker image and checks the app answers inside it.
- `.gitattributes` (keeps `.bat` files in Windows line endings), `.gitignore` (raw CSVs, local database, settings and API key stay out), `LICENSE` (MIT for code; Olist data is CC BY-NC-SA 4.0), `CONTRIBUTING.md`.

## Architecture

```
frontend/  (HTML + CSS + plain JS, no build step)
   │  fetch /api/...
   ▼
backend/app/main.py   FastAPI routes ─┬─ engine.py  loads model, guardrails, queue; scores orders
                                      └─ store.py   action board + activity log (SQLite)
   │
   ▼
src/  the project's own modules: features, scoring, prediction (TreeSHAP),
      artifacts (hash-checked model loading), validation guardrails
models/  trained XGBoost pipeline + metadata + rate lookup + reference stats
```

API routes: `GET /api/summary`, `GET /api/orders`, `GET /api/orders/{id}`,
`GET /api/sellers/{id}`, `POST /api/score`, `POST /api/score/batch`,
`GET /api/template.csv`, `GET|POST /api/board`, `PATCH|DELETE /api/board/{id}`,
`POST /api/board/autofill`, `GET /api/board/activity`, `GET /api/insights`, `GET /api/tracking/model`, `GET /api/tracking/{order_id}`, `POST /api/score/tracking`,
`GET /api/model`, `GET /api/meta`, `GET /api/health`.

## Project layout

```
OnTime/
├── start_ontime.bat      Windows one-click start (plus setup.bat, run_tests.bat, score_orders.bat,
│                         rebuild_app_data.bat, docker_start/stop.bat, reset_board.bat)
├── start_ontime.sh       the same for macOS / Linux
├── github_push.bat       push the project to a new GitHub repo; .github/workflows/ci.yml runs tests on Windows + Linux
├── run.py                start the web app
├── backend/
│   ├── app/              main.py (FastAPI), engine.py (model, queue, briefing, planner), store.py (board), assistant.py (Ask OnTime)
│   ├── app_data/         queue.csv, patterns.json, categories.json, network.json, insights.json, meta.json
│   └── prepare_app_data.py
├── frontend/
│   ├── index.html
│   └── assets/           app.css, app.js, bundled fonts and Chart.js
├── src/                  reusable analysis + scoring code
│   ├── clustering.py, simulation.py, narrative.py, categories.py  ← patterns, simulated logistics, stories, focus categories
│   ├── preprocessing.py, features.py, modeling.py
│   ├── prediction.py           ← per-order TreeSHAP explanations
│   ├── artifacts.py            ← rate lookup, integrity-checked model loading
│   ├── scoring.py              ← batch scoring, guardrails, suggested actions
│   ├── data_validation.py      ← ACCEPT / WARN / REJECT gate for a raw dataset
│   └── inference_validation.py ← ACCEPT / REJECT / LOW_CONFIDENCE gate for one order
├── models/               final model: portable copy (XGBoost JSON + preprocessor, loaded first),
│                         original joblib pickle, metadata, reference stats, rate lookup
├── notebooks/            01–08: the analysis, in order, with outputs
├── data/
│   ├── raw/              the 9 Olist CSVs (add them to re-run the notebooks)
│   └── processed/        model-ready table, scored test set, backtest results
├── train_ensemble.py / calibrate_ensemble.py / train_ensemble.bat   tuning, ensemble, calibration
├── tests/                pytest suite (239 tests) — `pytest tests/`
├── dashboard/            static HTML dashboard, chart PNGs, JSON exports
├── submission_documents/ OnTime_Report_ZeepleWork.docx, OnTime_Presentation.pptx, OnTime_Poster.pptx
├── score.py              command-line scorer for a CSV of new orders
├── export_portable_model.py, export_risk_summary.py, build_dashboard.py, make_report_charts.py
├── Dockerfile, docker-compose.yml, .dockerignore
├── requirements.txt      full environment (pinned to the model's versions)
├── requirements-app.txt  minimal runtime for the app / Docker
└── .env.example          settings (model folder, board database, host/port, batch limit)
```

## Method, in one paragraph

Orders with `order_status == delivered` and valid dates get a label:
`is_delayed = 1` if delivered after the estimated delivery date. Every
feature used to predict that label is restricted to information that
would genuinely be known at order time (order value, weight, category,
promised delivery window, seller/customer location and distance, payment
type, product photo count/description length) plus three
**time-ordered historical rates** (seller, product category, customer
state) that only look at orders that happened strictly earlier — a
naive `groupby().mean()` on the whole table would leak the outcome of
an order into its own risk score, which is avoided here deliberately.
Logistic Regression, a Decision Tree, a Random Forest and XGBoost are
trained and compared on precision/recall/F1/ROC-AUC (not accuracy — the
~8% base delay rate makes accuracy meaningless). XGBoost wins, is
hyperparameter-tuned, and has its decision threshold optimised for F1
rather than left at an arbitrary 0.5. The final tuned model is saved to
`models/ontime_final_model.joblib`.

## The seasonality problem — found, then actually fixed

Notebook 05 found that a strict time-based train/test split collapses
performance (ROC-AUC ~0.48, worse than a coin flip) because the delay
rate itself is highly seasonal (see the monthly trend chart in notebook
03). Notebook 06 doesn't stop at reporting that: it runs a
**rolling-origin backtest** — retrain monthly, predict only the next
month, slide forward — across 17 months, recovering a mean ROC-AUC of
~0.68. The conclusion isn't "the model can't handle seasonality", it's
"the model needs scheduled retraining", and that's now demonstrated
rather than just asserted.

## A hardware note worth knowing before you re-run this
This project was developed in a **single-core** sandbox. `n_jobs=-1`
anywhere in scikit-learn/XGBoost on a single core is pure multiprocessing
overhead with no real parallelism benind it, and a standard k-fold
`RandomizedSearchCV` for Random Forest was impractically slow as a
result (a single 300-tree fit took ~100 seconds). `src/modeling.py`
works around this with `tune_model_holdout()`, a single train/validation
split search instead of k-fold, and a deliberately smaller search grid
for Random Forest than for XGBoost (XGBoost's histogram-based training
is ~15-30x faster per fit here). On a normal multi-core machine, feel
free to swap in `sklearn.model_selection.RandomizedSearchCV` with
`n_jobs=-1` and a larger grid for both models — the code is written so
that's a drop-in change, not a rewrite.

## Data quality guardrails — is a given dataset even good enough?

Two separate gates were added after an adversarial audit of the whole
pipeline (notebook 07), because nothing originally stopped bad input
from silently producing a plausible-looking but meaningless result:

- **`src/data_validation.py`** checks a whole raw dataset *before*
  cleaning starts: required files present, required columns present,
  minimum row counts, duplicate primary keys, referential integrity
  between tables (do `order_items` rows actually point at real orders
  and products?), date-sequence sanity, and whether there's even enough
  delivered-order data to build a label from. Returns a verdict —
  `ACCEPT`, `ACCEPT_WITH_WARNINGS`, or `REJECT` — with specific,
  itemised reasons. Notebook 07 proves this actually rejects bad data,
  not just approves good data, by running it against a deliberately
  corrupted copy (missing column, duplicated IDs, orphaned foreign
  keys, negative prices) and confirming it comes back `REJECT`.

- **`src/inference_validation.py`** checks a single *new* order at
  prediction time: are all required fields present, are they the right
  type, are any structurally impossible (negative price, a rate above
  1.0)? Those get **rejected outright** rather than silently scored.
  Orders that are well-formed but unusual (e.g. a shipping distance far
  beyond anything in training) are still scored, but flagged
  `LOW_CONFIDENCE` with the specific reason — refusing to score
  anything slightly unusual would make the system useless, but scoring
  it with false confidence would be worse than not scoring it.

  One subtlety worth knowing about: the three historical delay-rate
  features are deliberately given a *structural* [0,1] check instead of
  a percentile-based "this is unusual" check. A first version flagged
  genuinely high-risk orders (unusually high seller delay rate) as
  low-confidence purely for being unusual — exactly backwards, since
  that's the signal the whole system exists to catch. This was caught
  by testing the guardrail against a real order before trusting it, not
  assumed to be correct.

## Loopholes found and fixed (notebook 07)

An adversarial pass over the whole pipeline, done specifically to find
mistakes rather than confirm the pipeline was fine:

1. **No gate on the raw input data** → fixed with `data_validation.py`.
2. **A real threshold-leakage bug**: the decision threshold was
   originally selected using the test set's own labels
   (`find_best_threshold(y_test, proba_test)`), then F1 was reported on
   that same test set — tuning a knob on the exam and grading yourself
   with it. Fixed by selecting the threshold on a validation slice
   carved out of training data only (`select_operating_threshold` in
   `src/modeling.py`). The bias turned out to be small here (F1 0.383
   leaky vs. 0.380 honest) — reassuring, but not something that could
   have been known in advance, which is why it's fixed rather than left
   as "probably fine."
3. **No guardrails at prediction time** → fixed with
   `inference_validation.py`.
4. **The guardrail's own out-of-distribution check was backwards for
   rate features** → fixed as described above.
5. **The raw-data validator itself crashed on a corrupted numeric
   column** (text sneaked into a `price` field raised an unhandled
   pandas `TypeError` from a raw `<` comparison, instead of the
   validator reporting a clean FAIL). Fixed by coercing declared-numeric
   columns with `pd.to_numeric(errors="coerce")` first and reporting
   the coercion failure rate as its own check. Verified against an
   empty directory, files with the wrong schema entirely, a zero-byte
   file, and a price column with 4.4% text values sneaked in — all four
   now return a clean verdict instead of a stack trace.
6. **A temporal leakage bug in the historical-rate features**: "history"
   was defined as same-group orders purchased earlier, not delivered
   earlier. A typical order takes ~10 days to arrive, so on average
   ~8% (up to double that in some states) of what counted as
   "history" had not actually been delivered yet — its outcome did not
   exist at prediction time. The fallback compounded this further,
   defaulting to the platform-wide rate computed over the ENTIRE
   dataset, future orders included. Fixed in `add_historical_rates()`:
   an order's history is now same-group orders delivered strictly
   before its own purchase time, and the fallback is the platform-wide
   rate similarly restricted to what was knowable at that moment.
   Verified two ways: a from-scratch, independent recomputation against
   the raw dates that matches the feature on every sampled order
   (notebook 04), and a battery of unit tests that check an order's own
   outcome never affects its own feature, a later order never affects
   an earlier one's feature, and the fallback never uses the future
   (`tests/test_features_leakage.py`). On this dataset the aggregate
   performance impact of the bug was small (ROC-AUC 0.801 → 0.805) —
   reported honestly rather than oversold, since the point is that the
   feature is now correct by construction, not that it moved a metric.
7. **No practical way to score a genuinely new order.** Guardrails
   existed but required hand-computing three historical-rate features
   the caller would never have. Fixed with `src/artifacts.py` (a
   seller/category/state rate lookup, and a model loader that refuses
   to load a tampered file or a model trained on a different feature
   list than the current code expects), `src/scoring.py` (batch scoring
   with the rates filled in automatically), and `score.py` (a command-
   line wrapper needing no code at all).
8. **No regression protection.** All of the above were caught by
   one-off scratch scripts that would not stop the bug from coming
   back. Turned into a permanent suite (`tests/`, now 169 tests) and checked
   that each test actually fails when its corresponding fix is
   reverted, not just that it passes now.

## Dashboard

`dashboard/OnTime_Dashboard.html` is a static snapshot of `dashboard/*.json`
at the time it was built — it does **not** auto-update when the model or
data changes. After re-running notebooks 06/07/08, regenerate it with:

```
python export_risk_summary.py   # re-scores the test set -> dashboard/risk_summary.json, top_risk_orders.json
python build_dashboard.py       # bakes all dashboard/*.json into the HTML
python make_report_charts.py    # charts 15-17 used in the report and deck
```

The dashboard has four pages: Overview, Action queue (how often each risk band
was really late, the highest-risk orders with driver, suggested action and real
outcome), Root causes, and Model.

An earlier version of this project deleted the build script after first
use, which is exactly what let the dashboard silently go stale relative
to the model (baked-in numbers from a superseded Random Forest run) —
kept here deliberately so that doesn't happen again.

## Running it

```
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_data_understanding.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_model_improvements.ipynb
jupyter nbconvert --to notebook --execute --inplace 07_data_quality_and_guardrails.ipynb
jupyter nbconvert --to notebook --execute --inplace 08_production_scoring.ipynb
```

## Scoring new orders

Score
a CSV of new orders without writing any code:

```
python score.py --template              # writes template_orders.csv to look at
python score.py --input your_orders.csv --output scored.csv
```

You only need to supply `seller_id`, `product_category` and
`customer_state` — the three historical delay-rate features are filled
in automatically from `models/rate_lookup.json`. Every input row comes
back as exactly one output row: `SCORED` (with a probability, a
LOW/MEDIUM/HIGH/CRITICAL band, a confidence flag, and — for anything
over the alert threshold — the specific reasons) or `REJECTED` (with why).

Every scored row also gets a **`suggested_action`** — the thing a
coordinator should actually do, not just a number:

| Situation | Action |
|---|---|
| LOW, not flagged | No action needed. |
| MEDIUM, not flagged | Monitor: review if no carrier scan within 3 days. |
| HIGH but just under the alert line | Close to the alert line: check for a carrier scan within 2 days. |
| Flagged, main driver = seller's track record | Contact the seller to confirm dispatch; escalate after 48h. |
| Flagged, main driver = long distance | Book an express / long-haul carrier option. |
| Flagged, main driver = tight promise window | Send the customer a realistic ETA now. |
| Flagged, main driver = peak season | Prioritise in dispatch queue, pad the ETA. |
| …and so on for region, category, approval delay, weight, freight | |
| Any CRITICAL order | The same action, prefixed `URGENT:` so it sorts to the top. |

The action is chosen from the order's own top TreeSHAP driver, so two
orders with the same probability can get different instructions.
Rules live in `ACTION_BY_DRIVER` in `src/scoring.py`.

## Environment and Docker

`requirements.txt` pins scikit-learn and XGBoost to the exact versions in
`models/model_metadata.json`. That matters: unpickling the pipeline under a
different scikit-learn version raises `InconsistentVersionWarning` and is
not guaranteed to give the same predictions.

Settings (copy `.env.example` to `.env`; `run.py` and docker compose read it):

- `ONTIME_MODEL_DIR` — where the model artifacts live (default `models/`)
- `ONTIME_DB` — the action board's SQLite file (default `backend/ontime.db`)
- `ONTIME_HOST`, `ONTIME_PORT` — web server address (default `127.0.0.1:8000`)
- `ONTIME_MAX_BATCH_ROWS` — largest CSV batch scoring accepts (default 5,000)
- `ONTIME_TOP_REASONS` — risk drivers `score.py` lists per flagged order (default 3)

`docker compose up --build` builds one image (code, model, app data; no raw
data or notebooks) and keeps the board and assistant settings in a named volume.
`docker compose --profile llm up --build` also starts an Ollama container for the
laptop engine (then run `docker compose exec ollama ollama pull llama3.2` once). The Dockerfile
was written in a sandbox that couldn't reach Docker Hub, so run the build
once on your own machine before relying on it.

## Running the test suite

```
pytest tests/ -v
```

169 tests across thirteen files: the assistant (with fake Ollama/Gemini servers, time limits and fallbacks), clustering, simulation and narratives, the portable model and its load fallbacks, the web API (every route, the board lifecycle,
single and batch scoring, filters and paging), the raw-data validator (does it accept good
data and reject every kind of broken data without crashing), the
historical-rate feature (does an order's own outcome ever leak into its
own feature, does the future ever leak into a fallback), the per-order
guardrail (are blanks/negatives/impossible values rejected, are unusual-
but-valid values flagged without being rejected, are high-risk orders
never penalised just for being high-risk), and the saved model +
scoring pipeline end to end (tampered model file refused, mismatched
feature list refused, every input row produces exactly one output row,
every scored row carries a suggested action and CRITICAL ones are marked URGENT).
Each fixed bug has a test that was confirmed to fail when that bug was
temporarily reintroduced, not just a test that happens to pass now.


## Synthetic dispatch model (opt-in demonstration)

Fresh installs default **Score an order** and **Batch scoring** to `enriched`, the real-Olist checkout-time model. Older saved `auto` defaults migrate to `enriched`; explicit user-selected models are preserved. The separate `synthetic_operational` model remains available as an opt-in demonstration against `data/raw/synthetic_operational_orders.csv`. The untouched chronological synthetic test ROC-AUC is 0.9433 (PR-AUC 0.8927; 56,000 train / 12,000 validation / 12,000 test rows). The simulator exposes dispatch-time operational inputs and identifies estimates/defaults when live telemetry is not supplied. Batch scoring accepts either the checkout template or a dataset with the complete 16-feature dispatch schema.

**Critical scope distinction:** weather/congestion/capacity signals and labels in this benchmark are simulated. The 0.9433 ROC-AUC is not real-world validation and scores are not calibrated probabilities. Use actual operational telemetry and validate on later real labelled outcomes before operational reliance. The default live scorer and the precomputed historical risk queue both use a real-Olist checkout-time model because those records do not contain dispatch-time telemetry. Adapt-to-data remains on checkout-time models by default.


## Deployment smoke testing and real-world validation

See [`REAL_WORLD_VALIDATION_GUIDE.md`](REAL_WORLD_VALIDATION_GUIDE.md) for the distinction between automated tests, deployment-level HTTP smoke tests, and evaluation against observed delivery outcomes. Run `python scripts/http_smoke_test.py http://127.0.0.1:8000` against a running app. Use `scripts/validate_operational_outcomes.py` with timestamped predictions and a separate, genuinely observed outcomes CSV before making real-world performance claims.

### Windows startup error involving LightGBM / Dask / pandas

The app does not require LightGBM to serve the bundled models. `setup.bat` installs the minimal runtime dependencies from `requirements-app.txt`; LightGBM remains in `requirements.txt` for users who need the full training/notebook stack. If an existing Python environment has an incompatible optional LightGBM/Dask combination, OnTime catches that import failure and continues with the shipped compatible model artifacts rather than crashing during startup. For the cleanest setup, launch with `run_all.bat` or `start_ontime.bat` so the project-local `.venv` is used instead of unrelated packages installed in global Python.

If the issue persists, close the server window, delete only the project-local `.venv` folder, and run `setup.bat` again. Do not delete `models`, `data`, or `backend/app_data`.


## Data provenance (read before quoting any number)
- **0.717 ROC-AUC** — checkout stage, newest 19,294 real Olist orders, chronological test. The headline result.
- 0.804 / 0.809 — random-split experiments; not comparable with 0.717.
- 0.915 — held-out in-transit experiment with simulated tracking/weather/strike signals; the rolling monthly mean is 0.869.
- **0.9433** — dispatch stage on a *generated* dataset whose labels come from a formula over the model's own inputs; it demonstrates the pipeline only and is not real-world performance.
