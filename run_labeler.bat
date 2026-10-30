@echo off
title Barca Vision Labeler - Tactical Annotation Studio
echo ========================================================
echo  Launching Barca Vision Labeler Studio...
echo ========================================================
cd /d "%~dp0"

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set PYTHON_EXE="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    ) else (
        echo Error: Python was not found. Please install Python 3.10+ from python.org.
        pause
        exit /b 1
    )
) else (
    set PYTHON_EXE=python
)

%PYTHON_EXE% -c "import flask, numpy, pandas" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies (flask, numpy, pandas)...
    %PYTHON_EXE% -m pip install -r labeler\requirements.txt
)

%PYTHON_EXE% labeler\run_labeler.py
pause
