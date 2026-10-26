@echo off
setlocal EnableExtensions
title OnTime - web app
cd /d "%~dp0"
set "PYTHONUTF8=1"

rem ---- first run: set everything up automatically ----
if not exist ".venv\.ontime_ready" (
  echo  First run detected - running setup...
  call "%~dp0setup.bat" app nopause
  if errorlevel 1 exit /b 1
)

rem ---- self-check: repair the environment if a package is missing or broken ----
".venv\Scripts\python.exe" -c "import fastapi, uvicorn, pandas, sklearn, xgboost, joblib, numpy" >nul 2>&1
if errorlevel 1 (
  echo  Some packages are missing or broken - repairing the environment...
  call "%~dp0setup.bat" app nopause
  if errorlevel 1 exit /b 1
)

rem ---- make sure a settings file exists ----
if not exist ".env" (
  if exist ".env.example" copy /y ".env.example" ".env" >nul
)

rem ---- read the port from .env, default 8000 ----
set "PORT=8000"
if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if /i "%%A"=="ONTIME_PORT" set "PORT=%%B"
  )
)
if not "%~1"=="" set "PORT=%~1"
set "ONTIME_PORT=%PORT%"

rem ---- OneDrive folders sync files while they are open, which can corrupt
rem ---- or lock the board database, so keep it in local AppData there ----
echo "%~dp0" | findstr /i "OneDrive" >nul
if not errorlevel 1 (
  if not defined ONTIME_DB set "ONTIME_DB=%LOCALAPPDATA%\OnTime\ontime.db"
)

rem ---- is something already using the port? ----
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
  echo  Port %PORT% is already in use - OnTime may already be running.
  echo  Opening http://localhost:%PORT% ...
  start "" "http://localhost:%PORT%"
  echo  To use another port, run:  start_ontime.bat 8010
  pause
  exit /b 0
)

echo.
echo  ============================================
echo   OnTime is starting on http://localhost:%PORT%
echo   Keep this window open while you use the app.
echo   Close it, or press Ctrl+C, to stop the server.
echo  ============================================
echo.

rem ---- open the browser once the server answers ----
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='http://localhost:%PORT%/'; for($i=0;$i -lt 240;$i++){ try{ Invoke-WebRequest -UseBasicParsing $u -TimeoutSec 2 | Out-Null; Start-Process 'http://localhost:%PORT%'; break } catch { Start-Sleep -Milliseconds 700 } }"

".venv\Scripts\python.exe" run.py
set "CODE=%errorlevel%"
if not "%CODE%"=="0" (
  echo.
  echo  [X] The server stopped with an error ^(code %CODE%^). Read the messages above.
  echo      If packages are missing, run setup.bat again.
  exit /b %CODE%
)
endlocal
