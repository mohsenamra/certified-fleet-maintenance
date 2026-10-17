@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
echo Paper 4 prototype v2 (C-MAPSS): exact lambda-path records, online runs, analysis.
echo Step 1 about 45-60 min, step 2 about 15-30 min, step 3 a few minutes. Resumable: re-run if it stops.
if not exist results mkdir results
echo started %date% %time% > results\run_status_v2.txt
%PY% p4v2_offline.py || goto :fail
%PY% p4v2_online.py || goto :fail
%PY% p4v2_analyze.py || goto :fail
echo finished %date% %time% >> results\run_status_v2.txt
echo.
echo DONE.
pause
goto :eof
:fail
echo FAILED at %date% %time% >> results\run_status_v2.txt
echo Something failed - see the messages above.
pause
