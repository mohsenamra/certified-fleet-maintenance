@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Downloading NASA C-MAPSS (about 12 MB)...
where py >nul 2>nul && (py -3 get_data.py %* & goto :eof)
where python >nul 2>nul && (python get_data.py %* & goto :eof)
echo Python is not installed.
pause
