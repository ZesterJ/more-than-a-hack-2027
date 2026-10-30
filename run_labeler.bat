@echo off
setlocal
chcp 65001 >nul
title Barca Vision Labeler - Tactical Annotation Studio

echo ========================================================
echo   Launching Barca Vision Labeler Studio...
echo ========================================================

rem Resolve repository root directory
set "APP_DIR=%~dp0"
if exist "%APP_DIR%labeler\run_labeler.py" (
    cd /d "%APP_DIR%"
) else if exist "%APP_DIR%repo\labeler\run_labeler.py" (
    cd /d "%APP_DIR%repo"
) else (
    echo [ERROR] Could not locate labeler\run_labeler.py!
    echo Looked in: "%APP_DIR%" and "%APP_DIR%repo"
    pause
    exit /b 1
)

rem Locate Python 3
set "PY="
where python >nul 2>&1 && set "PY=python"
if "%PY%"=="" where py >nul 2>&1 && set "PY=py -3"
if "%PY%"=="" where python3 >nul 2>&1 && set "PY=python3"
if "%PY%"=="" (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    if exist "%ProgramFiles%\Python311\python.exe" set "PY=%ProgramFiles%\Python311\python.exe"
    if exist "%ProgramFiles%\Python312\python.exe" set "PY=%ProgramFiles%\Python312\python.exe"
)

if "%PY%"=="" (
    echo [ERROR] Python 3 was not detected on your system.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

rem Check required packages without problematic syntax
"%PY%" -c "import flask, numpy, pandas" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing missing requirements: flask, numpy, pandas...
    "%PY%" -m pip install -r labeler\requirements.txt
)

rem Launch the studio
"%PY%" labeler\run_labeler.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [NOTICE] Studio stopped with exit code %ERRORLEVEL%.
    pause
)
