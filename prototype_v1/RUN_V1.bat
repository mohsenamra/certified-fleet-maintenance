@echo off
cd /d "%~dp0"
if not exist results mkdir results
set LOG=results\run_status_v1.txt
echo [%date% %time%] start >> %LOG%
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY goto nopy
if not exist ..\data\CMAPSS\train_FD004.txt goto nodata
%PY% -c "import numpy,scipy,sklearn,highspy; print('PACKAGES_OK')" >> %LOG% 2>&1
if errorlevel 1 goto fail
echo Paper 4 prototype v1: all C-MAPSS subsets, distribution shift, baselines, online adaptive margin.
echo About 2.5-3.5 hours (best run overnight). Keep the computer awake and plugged in.
echo You can close this window and re-run: finished parts are kept.
echo [%date% %time%] offline >> %LOG%
%PY% p4v1_offline.py 2>> %LOG%
if errorlevel 1 goto fail
echo [%date% %time%] online >> %LOG%
if not exist results\v1_online.npz %PY% p4v1_online.py 2>> %LOG%
if errorlevel 1 goto fail
echo [%date% %time%] analysis >> %LOG%
%PY% p4v1_analyze.py 2>> %LOG%
if errorlevel 1 goto fail
echo [%date% %time%] ALL DONE >> %LOG%
echo.
echo ALL DONE.
pause
exit /b
:fail
echo [%date% %time%] FAILED exit code %errorlevel% >> %LOG%
echo An error occurred: the details are in prototype_v1\results\run_status_v1.txt
pause
exit /b
:nodata
echo C-MAPSS data not found. Run ..\data\RUN_GET_DATA.bat first.
pause
exit /b
:nopy
echo Python 3 was not found.
pause
