@echo off
cd /d "%~dp0"
title Demand Forecast and Stock Planner
echo.
echo  Starting the Demand Forecast and Stock Planner. Your browser opens in a few seconds.
echo  Keep this window open while you use the app. Close it to stop.
echo.
".venv\Scripts\python.exe" -m streamlit run app\app.py --server.headless false
pause