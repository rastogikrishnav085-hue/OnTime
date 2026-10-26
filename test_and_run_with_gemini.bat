@echo off
setlocal EnableExtensions
title OnTime v2 - Runner & Gemini Diagnostic
cd /d "%~dp0"
set "PYTHONUTF8=1"

echo ============================================================
echo   OnTime v2: Delivery Delay Early Warning + Gemini Assistant
echo ============================================================
echo.

rem ---- Step 1: Check virtual environment ----
if not exist ".venv\.ontime_ready" (
  echo [1/4] First run detected: Setting up virtual environment...
  call "%~dp0setup.bat" app nopause
  if errorlevel 1 (
    echo [ERROR] Setup failed. Please check error messages above.
    pause
    exit /b 1
  )
) else (
  echo [1/4] Virtual environment is ready.
)

rem ---- Step 2: Configure Gemini API Key ----
if not exist ".env" (
  if exist ".env.example" (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example
  )
)

set "KEY_FOUND="
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
  if /i "%%A"=="GEMINI_API_KEY" (
    if not "%%B"=="" (
      if not "%%B"=="YOUR_GEMINI_API_KEY_HERE" (
        set "KEY_FOUND=%%B"
      )
    )
  )
)

if defined KEY_FOUND (
  echo [2/4] Found GEMINI_API_KEY in .env.
) else (
  echo [2/4] GEMINI_API_KEY is not configured in .env.
  echo.
  echo If you want to enable the AI assistant with Gemini, enter your key now.
  echo (Free API key can be obtained at: https://aistudio.google.com/apikey)
  echo Press ENTER without typing anything to continue with built-in rule assistant.
  set /p USER_KEY="Enter Gemini API Key (optional): "
  if defined USER_KEY (
    powershell -NoProfile -Command "(Get-Content .env) -replace '^GEMINI_API_KEY=.*', ('GEMINI_API_KEY=' + $env:USER_KEY) | Set-Content .env"
    set "KEY_FOUND=%USER_KEY%"
    echo Saved GEMINI_API_KEY to .env!
  ) else (
    echo Running with default built-in assistant engine.
  )
)

echo.
echo [3/4] Running test verification suite...
call "%~dp0run_tests.bat"
if errorlevel 1 (
  echo [WARNING] Some tests reported issues, but continuing app launch.
) else (
  echo [3/4] Tests passed successfully!
)

echo.
echo [4/4] Starting OnTime v2 Web App...
echo ============================================================
echo   Access the app at: http://localhost:8000
echo   - Interactive Queue & Second Opinion
echo   - Adapt to Your Data (Upload CSVs)
echo   - Literature & Model Diagnostics
echo   - Ask OnTime (Gemini Chat Assistant)
echo ============================================================
call "%~dp0start_ontime.bat"
