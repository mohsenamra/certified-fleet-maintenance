@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
echo Paper 4 - Backblaze case study v2: predictor (step 1, about 10-20 min) and planning experiments (step 2, about 30-60 min).
echo Needs data\backblaze\panel_*.pkl.gz (already downloaded). Resumable: step 1 skips finished drive models.
if not exist results mkdir results
echo started %date% %time% > results\run_status_bb.txt
%PY% p4bb_data.py || goto :fail
%PY% p4bb_eval.py || goto :fail
echo finished %date% %time% >> results\run_status_bb.txt
echo.
echo DONE.
pause
goto :eof
:fail
echo FAILED at %date% %time% >> results\run_status_bb.txt
echo Something failed - see the messages above.
pause
