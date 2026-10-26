@echo off
setlocal EnableExtensions
title OnTime - rebuild app data
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\.ontime_ready" (
  call "%~dp0setup.bat" app nopause
  if errorlevel 1 exit /b 1
)

rem Rebuilds the scored queue, insights and dashboard from data\processed.
rem Needed only after re-running the notebooks or retraining the model.

if not exist "data\processed\ontime_joined_raw.csv" (
  echo.
  echo  [X] data\processed\ontime_joined_raw.csv is missing.
  echo      Put the 9 Olist CSVs in data\raw, run setup.bat full, then run
  echo      notebooks 01 to 08 in order. That recreates the processed files.
  echo.
  pause
  exit /b 1
)
echo  [0/3] Writing the portable model copy ...
".venv\Scripts\python.exe" export_portable_model.py || goto :fail
if exist "models\local_rebuild" rmdir /s /q "models\local_rebuild"
echo  [1/3] Scoring the held-out queue ...
".venv\Scripts\python.exe" backend\prepare_app_data.py || goto :fail
if exist "models\ensemble\ensemble_meta.json" (
  echo  [1b/3] Adding the ensemble's second opinion to the queue ...
  ".venv\Scripts\python.exe" backend\add_second_opinion.py || goto :fail
)
echo  [2/3] Refreshing the risk summary ...
".venv\Scripts\python.exe" export_risk_summary.py || goto :fail
echo  [3/3] Rebuilding the static dashboard ...
".venv\Scripts\python.exe" build_dashboard.py || goto :fail
echo.
echo  Done. Restart start_ontime.bat to see the new data.
pause
exit /b 0

:fail
echo.
echo  [X] A step failed - see the messages above.
pause
exit /b 1
