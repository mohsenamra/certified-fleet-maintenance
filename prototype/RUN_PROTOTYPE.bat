@echo off
cd /d "%~dp0"
if not exist results mkdir results
set LOG=results\run_status.txt
echo [%date% %time%] start > %LOG%
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY where python >nul 2>nul && set PY=python
if not defined PY goto nopy
if not exist ..\data\CMAPSS\train_FD001.txt goto nodata
%PY% -c "import numpy,scipy,sklearn,highspy; print('PACKAGES_OK')" >> %LOG% 2>&1
if errorlevel 1 goto fail
echo Paper 4 prototype: RUL predictor + fleet maintenance MILP on C-MAPSS FD001+FD003.
echo About 30-60 minutes. Keep the computer awake.
echo [%date% %time%] run >> %LOG%
%PY% p4_run.py 2>> %LOG%
if errorlevel 1 goto fail
echo [%date% %time%] analysis >> %LOG%
%PY% p4_analyze.py 2>> %LOG%
if errorlevel 1 goto fail
echo [%date% %time%] ALL DONE >> %LOG%
echo.
echo ALL DONE.
pause
exit /b
:fail
echo [%date% %time%] FAILED exit code %errorlevel% >> %LOG%
echo An error occurred: the details are in prototype\results\run_status.txt
pause
exit /b
:nodata
echo C-MAPSS data not found. Run ..\data\RUN_GET_DATA.bat first.
pause
exit /b
:nopy
echo [%date% %time%] PYTHON_NOT_FOUND >> %LOG%
echo Python 3 was not found.
pause
