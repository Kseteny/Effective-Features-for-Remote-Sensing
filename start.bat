@echo off
title Feature Selection Tool
cd /d "%~dp0"

echo.
echo   Feature Selection Tool
echo   ---------------------
echo.

if not exist "venv\Scripts\activate.bat" (
    echo   Python environment not found.
    echo   Run install.bat first.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist "web\.output\public\index.html" (
    echo   Web interface not built yet.
    echo   The server will show build instructions in the browser.
    echo.
)

echo   Opening browser...
start "" "http://localhost:8000"

echo   Server is running. Close this window or press Ctrl+C to stop.
echo.

cd src
python -m uvicorn effective_features.api:app --host 127.0.0.1 --port 8000

echo.
echo   Server stopped.
pause
