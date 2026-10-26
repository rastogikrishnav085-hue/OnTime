@echo off
rem OnTime - one-click launcher. Does setup if needed, then starts the app.
rem All logic lives in start_ontime.bat so there is only one launcher to maintain.
cd /d "%~dp0"
call "%~dp0start_ontime.bat" %*
if errorlevel 1 pause
