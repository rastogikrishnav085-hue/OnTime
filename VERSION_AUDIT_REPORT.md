# OnTime v5 - audit, fixes and honest status (build 4)

## 1. What was fixed in this build
| Problem found | Fix |
|---|---|
| `run_all.bat` had Unix line endings and skipped OneDrive/port/repair checks | Now a CRLF wrapper around `start_ontime.bat`, which repairs a broken `.venv` (previous build) |
| **Model page showed only the simulated 0.943** and had lost the real-data results | Model page now shows: real-order ROC-AUC/PR-AUC/precision/recall, the alert-size table, and one table of every score with its data and test method (`/api/model/evaluation`) |
| **Alert-size table did not match the Overview** (it used an old random-split test file, 8.1% late rate) | `/api/model/thresholds` now uses the same 19,294-order queue as the Overview (5.3% late) |
| Assistant "How reliable is the model?" quoted only the simulated 0.943 | Now gives the real-order figures first, then the simulated benchmark, clearly labelled |
| Assistant said "Served by default: the ensemble" (false; the queue serves the enriched model) | Corrected, with a note that random-split numbers are optimistic |
| "Second opinion" model is the older random-split model, which saw ~80% of the queue orders in training | Caveat added in the order drawer; its scores are not used for any quoted AUC |
| README quick start described steps `run_all.bat` no longer does; no guide to which AUC is which | Quick start corrected; "Which ROC-AUC is which" table added |

## 2. ROC-AUC: what is true (independently recomputed, not copied)
| Score | ROC-AUC | Real or simulated | Test |
|---|---|---|---|
| Risk queue (served) | **0.717** | Real | Newest 20% of orders, never seen in training |
| Original XGBoost / ensemble | 0.804 / 0.809 | Real | Random split (optimistic) |
| After carrier handoff | 0.852 | Real | Random split (0.77 month by month) |
| Tracking | 0.915 | Real orders + simulated tracking | Simulation only |
| Dispatch operational (synthetic benchmark; opt-in) | 0.9433 | Simulated | Newest 12,000 simulated orders |

Recomputed here: the shipped dispatch model gives exactly 0.9433 (PR-AUC 0.8927, precision 0.825, recall 0.779) on its held-out slice.
The simulated labels were generated from a formula of the same simulated inputs, so 0.9433 shows the pipeline works; it does not show real-world accuracy.
The queue alerts: 24.1% of orders flagged, precision 10.0%, recall 45.3%, against a 5.3% late rate. On the newest 20% of that holdout the AUC falls to about 0.59, so the model is weaker on the very latest orders.
**The goal of ROC-AUC above 0.90/0.93 is met only on simulated data. It was not reached on real data, and no number in the app claims otherwise.**

## 3. Feature checklist (verified by loading every page in a real browser and clicking through)
Overview, Risk queue (+ order drawer with handoff and tracking scores), Action board (4 stages, one button per card, all three buttons exercised), Briefing, Insights, Delay patterns (6 clusters), Focus categories (6 categories), Network (simulated), Literature (25 sources, 9 themes), Alert planner, Score an order, Batch scoring, Adapt to your data, Ask OnTime (Auto / Built-in / Ollama / Gemini), Model, How it works, Settings (8 themes). No JavaScript, network or server errors on any page.

## 4. Known gaps (not fixed here)
- the submission documents were written for an earlier version and have since been updated. They previously led with the random-split XGBoost/ensemble figures (0.804/0.809) and the slide says "Check 1 in 8 orders, catch half the delays", while the app now says 1 in 4 and 45%. They need regenerating from the numbers in section 2.
- Earlier v3/v4 ZIPs were not available in this session, so there was no file-by-file diff against them; the check was against your requirements list instead.
- `.bat` launchers could not be executed on Windows here.
- Batch scoring needs the full template columns.
- Live weather needs internet.
- Real-world accuracy needs real operational outcomes (REAL_WORLD_VALIDATION_GUIDE.md).

## 5. Verification run in this environment
220 automated tests passed (211 originally, +3 launcher, +6 evaluation consistency); HTTP smoke test 11/11; all 17 pages loaded with 0 errors; board, drawer, simulator, assistant, planner and settings flows clicked through; the exact LightGBM/Dask crash simulated with all four models still scoring.
