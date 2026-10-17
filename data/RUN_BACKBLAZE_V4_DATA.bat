@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
echo Paper 4 - Backblaze v4 data: downloads 6 older quarters (Q1_2024 .. Q2_2025, about 7-9 GB) and builds richer
echo per-model panels for all 10 quarters (panel4_*). Old files are NOT changed. Several hours; re-run to resume.
echo started %date% %time% > backblaze\run_status_v4data.txt
%PY% get_backblaze_v4.py || goto :fail
echo finished %date% %time% >> backblaze\run_status_v4data.txt
echo.
echo DONE.
pause
goto :eof
:fail
echo FAILED at %date% %time% >> backblaze\run_status_v4data.txt
echo Not finished - see the messages above. Running this file again resumes where it stopped.
pause
