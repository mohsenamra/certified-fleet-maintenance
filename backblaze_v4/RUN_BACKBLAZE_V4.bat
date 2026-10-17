@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
echo Paper 4 - Backblaze v4: stronger predictor (step 1, about 1-1.5 hours) and the full evaluation (step 2, about 5-7 hours).
echo Protocol: PROTOCOLS.md, P1. Results: backblaze_v4\results\summary_bb4.txt. v1-v3 files are NOT touched.
echo Resumable: finished drive models are skipped in both steps. Close Chrome and other heavy programs while it runs.
if not exist results mkdir results
echo started %date% %time% >> results\run_status_bb4.txt
%PY% p4bb4_data.py || goto :fail
echo step 1 finished %date% %time% >> results\run_status_bb4.txt
%PY% p4bb4_eval.py || goto :fail
echo finished %date% %time% >> results\run_status_bb4.txt
echo.
echo DONE.
pause
goto :eof
:fail
echo FAILED at %date% %time% >> results\run_status_bb4.txt
echo Something failed - see the messages above. Running this file again resumes where it stopped.
pause
