@echo off
setlocal EnableExtensions
title OnTime - tests
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\.ontime_ready" (
  call "%~dp0setup.bat" app nopause
  if errorlevel 1 exit /b 1
)
echo.
echo  Running the OnTime test suite...
echo.
".venv\Scripts\python.exe" -m pytest tests -q -p no:warnings
if errorlevel 1 (
  echo.
  echo  [X] Some tests failed - see above.
) else (
  echo.
  echo  All tests passed.
)
echo.
pause
endlocal
