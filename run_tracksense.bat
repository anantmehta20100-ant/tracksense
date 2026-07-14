@echo off
REM ============================================================
REM  TrackSense - local launcher (Windows)
REM  Binds to your whole network (so your PHONE on the same Wi-Fi
REM  can connect) and serves over HTTPS (required for the phone
REM  camera page). Press Ctrl+C to stop.
REM ============================================================
cd /d "%~dp0"

REM Reachable by other devices (your phone), not just this PC.
set TRACKSENSE_HOST=0.0.0.0
REM HTTPS so the phone browser will allow camera access on /phone.
set TRACKSENSE_HTTPS=1

echo(
echo   Starting TrackSense (HTTPS)...
echo(
echo   The URLs are printed below. On your phone, open the
echo   "Phone camera input" link and, when the browser warns the
echo   site is "not secure", tap Advanced -^> Proceed. That warning
echo   is expected for a local self-signed certificate.
echo(
echo   (Keep this window open. Press Ctrl+C to stop.)
echo(

".venv\Scripts\python.exe" backend\app.py

echo(
echo   TrackSense stopped.
pause
