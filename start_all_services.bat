@echo off
title SiangTTS Master Service Launcher
echo ================================================================
echo           🚀 Launching SiangTTS Microservices in Order
echo ================================================================
echo.

echo [1/4] Starting Python GPU Worker on port 8021...
start "[1/4] GPU Service (:8021)" cmd /k "cd /d ""%~dp0voice-cloning"" && uv run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8021"

echo Waiting 5 seconds for GPU Service initialization...
timeout /t 5 /nobreak >nul

echo.
echo [2/4] Starting Go Fiber Queue Gateway on port 8020...
start "[2/4] Go Queue Gateway (:8020)" cmd /k ""%~dp0start_queue_service.bat""

echo Waiting 2 seconds for Queue Gateway...
timeout /t 2 /nobreak >nul

echo.
echo [3/4] Starting Production Webhook on port 8010...
start "[3/4] Webhook Service (:8010)" cmd /k "cd /d ""%~dp0voice-cloning"" && uv run uvicorn src.webhook:app --host 0.0.0.0 --port 8010"

echo.
echo [4/4] Starting Tone Studio Web App on port 8011...
start "[4/4] Tone Studio (:8011)" cmd /k "cd /d ""%~dp0voice-cloning-with-tones"" && py -m uvicorn app.main:app --reload --port 8011"

echo.
echo ================================================================
echo  ✅ All services launched successfully in separate windows!
echo.
echo  👉 Webhook API (n8n):    http://localhost:8010/webhook
echo  👉 Tone Studio UI:       http://localhost:8011
echo  👉 Benchmark Suite:      http://localhost:8011/test
echo  👉 Go Queue Gateway:     http://localhost:8020
echo  👉 Python GPU Backend:   http://localhost:8021
echo ================================================================
echo.
pause
