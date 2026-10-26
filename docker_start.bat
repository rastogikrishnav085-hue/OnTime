@echo off
setlocal EnableExtensions
title OnTime - Docker
cd /d "%~dp0"
set "PYTHONUTF8=1"
where docker >nul 2>&1
if errorlevel 1 (
  echo  [X] Docker was not found. Install Docker Desktop, or use start_ontime.bat instead.
  pause
  exit /b 1
)
docker info >nul 2>&1
if errorlevel 1 (
  echo  [X] Docker Desktop is not running. Start it, wait until it says "running", then try again.
  pause
  exit /b 1
)
echo  Building and starting OnTime in Docker - the first build takes a few minutes ...
docker compose up --build -d
if errorlevel 1 (
  echo  [X] docker compose failed - see above.
  pause
  exit /b 1
)
echo  Waiting for the app to answer ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "for($i=0;$i -lt 90;$i++){ try{ Invoke-WebRequest -UseBasicParsing 'http://localhost:8000/api/health' -TimeoutSec 2 | Out-Null; exit 0 } catch { Start-Sleep 1 } }; exit 1"
if errorlevel 1 (
  echo  [X] The container started but the app did not answer. Logs: docker compose logs
  pause
  exit /b 1
)
start "" "http://localhost:8000"
echo.
echo  OnTime is running in Docker at http://localhost:8000
echo  Stop it with docker_stop.bat
pause
