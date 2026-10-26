@echo off
setlocal EnableExtensions
title OnTime - score a CSV of orders
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\.ontime_ready" (
  call "%~dp0setup.bat" app nopause
  if errorlevel 1 exit /b 1
)

rem Drag a CSV file onto this .bat file to score it.
rem Run it without a file to create an example template first.

if "%~1"=="" (
  echo.
  echo  No file given - writing an example input file: template_orders.csv
  ".venv\Scripts\python.exe" score.py --template template_orders.csv
  echo.
  echo  Fill in your own orders in that layout, then drag the CSV onto score_orders.bat.
  echo.
  pause
  exit /b 0
)

set "IN=%~f1"
set "OUT=%~dpn1_scored.csv"
echo.
echo  Scoring "%IN%"
echo.
".venv\Scripts\python.exe" score.py --input "%IN%" --output "%OUT%"
if errorlevel 1 (
  echo.
  echo  [X] The file could not be scored - see the message above.
) else (
  echo.
  echo  Results saved next to your file:
  echo  "%OUT%"
)
echo.
pause
endlocal
