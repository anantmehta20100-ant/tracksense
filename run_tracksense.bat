@echo off
REM ============================================================
REM  TrackSense - local launcher (Windows)
REM  Double-click this file to start the app. It binds to your
REM  whole network so your PHONE (on the same Wi-Fi) can open it
REM  too. Press Ctrl+C here to stop it.
REM ============================================================
cd /d "%~dp0"

REM Bind to all network interfaces so other devices (your phone) can connect.
REM Remove this line to restrict access to this PC only (localhost).
set TRACKSENSE_HOST=0.0.0.0

echo(
echo   Starting TrackSense...
echo(
echo   On this PC:    http://localhost:5000/
echo   On your phone: shown below as "http://192.168.x.x:5000/"
echo                  (phone must be on the SAME Wi-Fi)
echo(
echo   (Keep this window open. Press Ctrl+C to stop.)
echo(

".venv\Scripts\python.exe" backend\app.py

echo(
echo   TrackSense stopped.
pause
