@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
echo Paper 4 - Backblaze v3c: CRC vs high-probability (LTT) certification. Uses the existing p4bb3b data files; about 30-60 min.
echo Results go to backblaze_v3\results\summary_bb3c.txt (v3b results are NOT touched).
if not exist results mkdir results
echo started %date% %time% > results\run_status_bb3c.txt
%PY% p4bb3c_ltt.py || goto :fail
echo finished %date% %time% >> results\run_status_bb3c.txt
echo.
echo DONE.
pause
goto :eof
:fail
echo FAILED at %date% %time% >> results\run_status_bb3c.txt
echo Something failed - see the messages above.
pause
