@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
echo Paper 4: download N-CMAPSS (NASA) and keep only sub-data set DS02 in data\ncmapss\.
echo Tries to extract only DS02 from the remote archive; otherwise downloads the archive (several GB, resumable).
echo Safe to re-run if it stops.
%PY% get_ncmapss.py || goto :fail
echo.
echo DONE.
pause
goto :eof
:fail
echo FAILED - see the messages above.
pause
