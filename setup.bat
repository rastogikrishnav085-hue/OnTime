@echo off
setlocal EnableExtensions
title OnTime - setup
cd /d "%~dp0"
set "PYTHONUTF8=1"

echo.
echo  ============================================
echo   OnTime - one-time setup
echo  ============================================
echo.

rem ---- choose what to install: "setup.bat full" adds notebooks/Jupyter ----
set "REQ=requirements-app.txt"
if /i "%~1"=="full" set "REQ=requirements.txt"

rem ---- find Python 3.11 or newer ----
set "PY="
for %%V in (3.12 3.13 3.11 3.14) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo  [X] Python 3.11 or newer was not found.
  echo      Install it from https://www.python.org/downloads/
  echo      and tick "Add python.exe to PATH" during installation.
  echo      Then run setup.bat again.
  echo.
  pause
  exit /b 1
)
for /f "delims=" %%I in ('%PY% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%I"
echo  [1/4] Using Python %PYVER%  ^(%PY%^)

rem ---- virtual environment ----
if not exist ".venv\Scripts\python.exe" (
  echo  [2/4] Creating a virtual environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo  [X] Could not create the virtual environment.
    pause
    exit /b 1
  )
) else (
  echo  [2/4] Virtual environment already exists
)

rem ---- packages ----
echo  [3/4] Installing packages from %REQ% - the first time takes a few minutes ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r "%REQ%"
if errorlevel 1 (
  echo.
  echo  [X] Package installation failed. Check your internet connection and try again.
  pause
  exit /b 1
)
echo %REQ%> ".venv\.ontime_ready"

rem ---- settings file ----
if not exist ".env" (
  copy /y ".env.example" ".env" >nul
  echo  [4/4] Created .env from .env.example
) else (
  echo  [4/4] .env already exists, left unchanged
)

echo.
echo  Setup complete. Double-click start_ontime.bat to open the app.
echo.
if /i not "%~2"=="nopause" pause
endlocal
exit /b 0
