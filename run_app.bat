@echo off
title Few-Shot Text Classification Project
cd /d "%~dp0"
echo Installing required packages...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Check Python and internet connection.
  pause
  exit /b 1
)
echo Starting application...
python -m streamlit run app.py
pause
