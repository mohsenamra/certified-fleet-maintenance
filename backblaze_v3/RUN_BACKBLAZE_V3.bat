@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
echo Paper 4 - Backblaze case study v3: one consistent risk model (step 1, about 20-40 min) and planning experiments with abstention, frontier and cost-aware online rule (step 2, about 3-4 hours).
echo Results go to backblaze_v3\results (v2 results are NOT touched). Resumable: step 1 skips finished drive models.
if not exist results mkdir results
echo started %date% %time% > results\run_status_bb3.txt
%PY% p4bb3_data.py || goto :fail
%PY% p4bb3_eval.py || goto :fail
echo finished %date% %time% >> results\run_status_bb3.txt
echo.
echo DONE.
pause
goto :eof
:fail
echo FAILED at %date% %time% >> results\run_status_bb3.txt
echo Something failed - see the messages above.
pause
