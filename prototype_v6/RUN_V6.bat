@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
set OMP_NUM_THREADS=1
set P4_WORKERS=3
echo Paper 4 prototype v6 (N-CMAPSS, ten sub-data sets).
echo Step 1: flight-level aggregation of every sub-data set (extracts one file at a time, about 5-15 min each).
echo Step 2: predictors + exact lambda-path records (about 1-3 hours). Step 3: analysis.
echo Resumable: re-run if it stops. Writes only into data\ncmapss\cycles_*.npz and prototype_v6\results.
%PY% -c "import h5py" 2>nul || %PY% -m pip install h5py || goto :fail
if not exist results mkdir results
echo started %date% %time% > results\run_status_v6.txt
%PY% ..\data\prep_ncmapss.py || goto :fail
%PY% p4v6_offline.py || goto :fail
%PY% p4v6_analyze.py || goto :fail
echo finished %date% %time% >> results\run_status_v6.txt
echo.
echo DONE.
pause
goto :eof
:fail
echo FAILED at %date% %time% >> results\run_status_v6.txt
echo Something failed - see the messages above.
pause
