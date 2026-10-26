@echo off
setlocal EnableExtensions
title OnTime - push to GitHub
cd /d "%~dp0"
set "PYTHONUTF8=1"
where git >nul 2>&1
if errorlevel 1 (
  echo  [X] Git is not installed. Get it from https://git-scm.com/download/win and run this again.
  pause
  exit /b 1
)
echo.
echo  Create an EMPTY repository on github.com first ^(no README, no licence^),
echo  then paste its address here, for example https://github.com/yourname/ontime.git
echo.
set /p "REPO=Repository URL: "
if "%REPO%"=="" (
  echo  No address given - nothing was changed.
  pause
  exit /b 1
)
if not exist ".git" (
  git init -b main
)
git add -A
git commit -m "OnTime: delivery delay early warning app" >nul 2>&1
git remote remove origin >nul 2>&1
git remote add origin "%REPO%"
git push -u origin main
if errorlevel 1 (
  echo.
  echo  [X] Push failed. If GitHub asked you to sign in, finish that and run this again.
  pause
  exit /b 1
)
echo.
echo  Done. The GitHub Actions tab will show the tests running on Windows and Linux.
pause
endlocal
