# Round 6 changes (interface only, no model or API changes)
Date: 26 October 2026

## Risk queue
- Rows are about 55–60 px tall (were about 88 px). Main driver shows as a short phrase without unit suffixes; suggested action shows its first clause (URGENT stays a red label). Full wording stays in the tooltip and the order drawer.
- Order ID has a tooltip with the full ID.
- On screens wider than 860 px the Add button stays pinned to the right edge while the table scrolls sideways.
- On phones the filter block is collapsed behind a "Filters" button, with a count of active filters. The table is visible on the first screen.

## Navigation
- Sidebar groups (Monitor, Understand, Act, Help) collapse. Monitor and Act start open; the group containing the current page opens automatically.
- Phones get a bottom tab bar (Overview, Queue, Board, Score, More) with live counts. "More" opens a two-column sheet with every page; it closes after choosing a page, on Esc, or by tapping outside it.
- Tab bar marker and badges use the rail's own text colour, so they are visible in both light and dark themes.

## Added in the final pass
- Queue filters on phones are collapsed behind a "Filters" button with an active-filter count.
- Fixed a JavaScript error when leaving the Model page before the monitoring panel finished loading (it tried to attach a listener to a page that was already gone).
- Overview "Needs attention" list on phones: text now wraps and the button sits on its own line, so category, route and action are no longer cut off.
- Queue table fits at 1360 px and wider without sideways scroll; narrower screens scroll with the Add button pinned.
- Uneven letter spacing in small text fixed app-wide (precise glyph positioning).
- Model page copy: the dispatch benchmark sentence no longer says "shown above" when nothing is above it.
- Screenshots in the report (Figs. 4-6), presentation (slide 13) and poster were retaken from the running app (1360x820 at 2x). The Action board shot uses demo cards seeded in a scratch database, not the shipped one.
- Report: one sentence added describing the grouped navigation and phone tab bar. Fig. 4 caption reworded because the new board screenshot no longer shows the "status recorded" toast.

## Smaller fixes
- Theme swatches in the sidebar are rendered once (they were rendered twice).
- Start-up splash is skipped on repeat visits in the same session and is shorter on first load.
- Smallest text raised to 12 px. Touch devices get 44 px tap targets.
- Small uppercase labels use precise glyph positioning (fixes uneven letter spacing in some browsers).
- Cache-busting query on app.js bumped to `20261026-ui-polish2`.

## Verified here
- app.js parses. All 17 pages loaded in Chromium on desktop (1440 px) and phone (390 px), with the real engine.py and the app's own GET handlers serving the real 19,294-order queue (FastAPI, pydantic, httpx, xgboost and lightgbm were replaced by stand-ins because they could not be installed). No page errors and no page-level horizontal scroll; repeated fast Model/About navigation throws no errors.
- Report, presentation and poster re-rendered through LibreOffice with the new screenshots in place.

## Not verified here
- The real FastAPI server was not run, and POST actions (adding to the board, live scoring, batch and adapt uploads) were not exercised. Live single-order and batch scoring need the real model libraries and were not tested.
- Clean Windows launch and the full pytest run are still to be done on your machine (`run_tests.bat`, `start_ontime.bat`).
- The model results and metrics in the documents are unchanged by this round.
