@echo off
title SeedVC Voice-Conversion Worker (:8022)
rem ---------------------------------------------------------------------------
rem SeedVC pins torch 2.4, which cannot coexist with the studio env -- so it runs
rem from its own virtualenv, as its own process, and the studio talks to it over
rem HTTP. Point these two at your checkout and venv once; see voxcpm+vc/README.md
rem for how to create them.
rem
rem The model weights are large (~2 GB). If you already have them under another
rem checkout, set HF_HOME to that cache instead of downloading a second copy.
rem ---------------------------------------------------------------------------
if "%SEEDVC_REPO%"=="" set SEEDVC_REPO=C:\Users\%USERNAME%\Desktop\seed-vc

if "%SEEDVC_PYTHON%"=="" (
    if exist "%SEEDVC_REPO%\seedvc-venv\Scripts\python.exe" (
        set "SEEDVC_PYTHON=%SEEDVC_REPO%\seedvc-venv\Scripts\python.exe"
    ) else if exist "%SEEDVC_REPO%\..\seedvc-venv\Scripts\python.exe" (
        set "SEEDVC_PYTHON=%SEEDVC_REPO%\..\seedvc-venv\Scripts\python.exe"
    ) else (
        set "SEEDVC_PYTHON=%SEEDVC_REPO%\seedvc-venv\Scripts\python.exe"
    )
)

if not exist "%SEEDVC_PYTHON%" (
    echo ERROR: SeedVC python not found at %SEEDVC_PYTHON%
    echo Set SEEDVC_PYTHON ^(and SEEDVC_REPO^) or see voxcpm+vc\README.md
    pause
    exit /b 1
)

cd /d "%~dp0voxcpm+vc"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ================================================================
echo  Starting SeedVC worker on http://127.0.0.1:8022
echo  repo:   %SEEDVC_REPO%
echo  python: %SEEDVC_PYTHON%
echo ================================================================
"%SEEDVC_PYTHON%" tools\seedvc_server.py --seedvc-repo "%SEEDVC_REPO%" --port 8022
pause
