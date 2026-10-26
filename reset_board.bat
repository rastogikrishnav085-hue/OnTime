@echo off
setlocal EnableExtensions
title OnTime - reset action board
cd /d "%~dp0"
set "PYTHONUTF8=1"
echo.
echo  This deletes every card and the activity log on the action board.
echo  Scores, the model and the data are not affected.
echo  Stop the app first ^(close the start_ontime window^).
echo.
set /p "ANS=Type YES to reset the board: "
if /i not "%ANS%"=="YES" (
  echo  Cancelled - nothing was changed.
  pause
  exit /b 0
)
set "FOUND="
for %%F in ("backend\ontime.db" "%LOCALAPPDATA%\OnTime\ontime.db") do (
  if exist "%%~F" (
    set "FOUND=1"
    del /f /q "%%~F"
    if exist "%%~F" echo  [X] Could not delete %%~F - is the app still running?
  )
)
if defined FOUND (
  echo  Board reset. It starts empty next time you open the app.
) else (
  echo  The board is already empty.
)
pause
endlocal
