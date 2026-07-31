@echo off
title Install Feature Selection Tool
cd /d "%~dp0"

echo.
echo   Installing Feature Selection Tool
echo   ---------------------------------
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo   Python not found.
    echo   Get it from python.org and tick "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

if not exist "venv" (
    echo   [1/3] Creating Python environment...
    python -m venv venv
    if errorlevel 1 goto fail
) else (
    echo   [1/3] Environment already exists, skipping.
)

call venv\Scripts\activate.bat

echo   [2/3] Installing Python packages...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 goto fail

where npm >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [3/3] Node.js not found - cannot build the web interface.
    echo   Calculations will still work from the command line.
    echo   Get Node.js LTS from nodejs.org, then run build_web.bat
    echo.
    goto done
)

echo   [3/3] Building web interface...
cd web
call npm install
if errorlevel 1 goto fail
call npm run generate
if errorlevel 1 goto fail
cd ..

:done
echo.
echo   Done. Use start.bat to run the tool.
echo.
pause
exit /b 0

:fail
echo.
echo   Something went wrong - see the error above.
echo.
pause
exit /b 1
