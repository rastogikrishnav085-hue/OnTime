@echo off
setlocal EnableExtensions
title OnTime - tune models and train the ensemble
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\.ontime_ready" (
  call "%~dp0setup.bat" app nopause
  if errorlevel 1 exit /b 1
)

rem Tunes XGBoost and histogram boosting, builds the weighted ensemble, compares everything
rem on the same test orders, and scores the app's queue with both models.
rem Takes about 20-30 minutes on one CPU core. If it is interrupted, run it again: finished
rem tuning trials are kept and it carries on. Add "quick" for a 5-minute test run.

set "ARGS="
if /i "%~1"=="quick" set "ARGS=--quick --no-rf"

echo  [1/2] Tuning, training and comparing (this is the long step) ...
".venv\Scripts\python.exe" train_ensemble.py %ARGS% || goto :fail
echo  [2/3] Scoring the queue with both models ...
".venv\Scripts\python.exe" backend\add_second_opinion.py || goto :fail
echo  [3/3] Calibrating the ensemble score into a real chance ...
".venv\Scripts\python.exe" calibrate_ensemble.py || goto :fail
echo.
echo  Done. Restart start_ontime.bat, then open the Model page.
pause
exit /b 0

:fail
echo.
echo  [X] A step failed - see the messages above.
pause
exit /b 1
