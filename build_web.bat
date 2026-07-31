@echo off
title Build Web Interface
cd /d "%~dp0"

echo.
echo   Rebuilding web interface
echo   ------------------------
echo.
echo   Needed after changes in the web folder.
echo.

cd web
call npm run generate
if errorlevel 1 (
    echo.
    echo   Build failed - see the error above.
    pause
    exit /b 1
)

echo.
echo   Done. Use start.bat to run the tool.
echo.
pause
