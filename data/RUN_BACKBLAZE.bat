@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Backblaze real hard-drive fleet data: 4 quarters, about 5.7 GB download + processing (1-3 hours). Re-run to resume.
where py >nul 2>nul && (py -3 get_backblaze.py  & goto :eof)
where python >nul 2>nul && (python get_backblaze.py  & goto :eof)
echo Python is not installed.
pause
