@echo off
cd /d "%~dp0"
set "PYTHONUTF8=1"
docker compose down
echo  OnTime container stopped. The action board is kept in the Docker volume.
pause
