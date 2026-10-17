@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Checking whether the Backblaze server is reachable (no download)...
where py >nul 2>nul && (py -3 get_backblaze.py --probe & goto :eof)
where python >nul 2>nul && (python get_backblaze.py --probe & goto :eof)
echo Python is not installed.
pause
