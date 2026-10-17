@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
echo Paper 4 prototype v5 (C-MAPSS, richer planner): exact lambda-path records, exactness checks, analysis.
echo Step 1 about 1-2 hours (4 parallel workers), step 2 about 15-30 min. Resumable: re-run if it stops.
echo Only writes into prototype_v5\results. Earlier results are not touched.
if not exist results mkdir results
echo started %date% %time% > results\run_status_v5.txt
%PY% p4v5_offline.py || goto :fail
%PY% p4v5_analyze.py || goto :fail
echo finished %date% %time% >> results\run_status_v5.txt
echo.
echo DONE.
pause
goto :eof
:fail
echo FAILED at %date% %time% >> results\run_status_v5.txt
echo Something failed - see the messages above.
pause
