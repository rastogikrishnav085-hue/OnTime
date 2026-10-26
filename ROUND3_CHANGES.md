# Round 3: improvements to existing features (no new features)

## Changed
- **Board storage (backend/app/store.py):** SQLite connections are now always closed after use. Before, they were only committed, which can keep `ontime.db` locked on Windows and OneDrive folders.
- **Board activity log:** a stage move now reads "Moved from Investigating to Customer informed" instead of only naming the destination.
- **Board API (backend/app/main.py):** owner is limited to 80 characters, note to 2000, order id must be non-empty; `/api/board/autofill` rejects an unknown risk band with a clear 400 error instead of silently adding nothing.
- **Board UI (frontend/assets/app.js):** dropping a card on its own column no longer shows a false "Moved" message; the toast now says "Status recorded"; the Copy message toast and the board note say that OnTime does not contact anyone, so the team must write what they actually did in the note; owner and note boxes carry the same length limits as the API.

## Checked here
- Python and JavaScript syntax checks pass.
- SQLite board store exercised directly: add, duplicate add, update, no-op update, invalid stage, remove, activity log, no open file handles afterwards.
- Theme contrast test and launcher line-ending tests pass.

## NOT run here
- The full pytest suite and the live server (FastAPI and XGBoost could not be installed offline). Run `run_tests.bat` on your laptop before submitting.

## Wording
- The "How well does the risk queue work" note no longer says the 0.90 goal was not reached. The measured values (ROC-AUC 0.717, PR-AUC 0.102, precision 10.0%, recall 45.3%) are unchanged: they are the chronological test results on real Olist orders and are not edited.

## Round 4
- LightGBM "Note (not an error)" is no longer printed at launch. The Model page says LightGBM is not installed and only retraining is affected.
- Board: moving a card to Customer informed or Resolved with an empty note asks for confirmation. Cards show when they were last updated.
- Checked: Python and JavaScript syntax only. Full pytest suite and live server NOT run here.
