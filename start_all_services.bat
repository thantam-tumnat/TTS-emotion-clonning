@echo off
title SiangTTS Master Service Launcher
echo ================================================================
echo           🚀 Launching SiangTTS Microservices in Order
echo ================================================================
echo.

echo [1/6] Starting Python GPU Worker on port 8021...
start "[1/6] GPU Service (:8021)" cmd /k "cd /d ""%~dp0voice-cloning"" && uv run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8021"

echo Waiting 5 seconds for GPU Service initialization...
timeout /t 5 /nobreak >nul

echo.
echo [2/6] Starting Go Fiber Queue Gateway on port 8020...
start "[2/6] Go Queue Gateway (:8020)" cmd /k ""%~dp0start_queue_service.bat""

echo Waiting 2 seconds for Queue Gateway...
timeout /t 2 /nobreak >nul

echo.
echo [3/6] Starting Production Webhook on port 8010...
start "[3/6] Webhook Service (:8010)" cmd /k "cd /d ""%~dp0voice-cloning"" && uv run uvicorn src.webhook:app --host 0.0.0.0 --port 8010"

echo.
echo [4/6] Starting Tone Studio Web App on port 8011...
start "[4/6] Tone Studio (:8011)" cmd /k "cd /d ""%~dp0voice-cloning-with-tones"" && py -m uvicorn app.main:app --reload --port 8011"

echo.
echo [5/6] Starting SeedVC voice-conversion worker on port 8022...
start "[5/6] SeedVC Worker (:8022)" cmd /k ""%~dp0start_seedvc_worker.bat""

echo Waiting 3 seconds for the SeedVC worker to claim the GPU...
timeout /t 3 /nobreak >nul

echo.
echo [6/6] Starting VoxCPM2+SeedVC Studio on port 8013...
start "[6/6] VoxCPM2+SeedVC Studio (:8013)" cmd /k "cd /d ""%~dp0voxcpm+vc"" && py -m uvicorn app.main:app --reload --port 8013"

echo.
echo ================================================================
echo  ✅ All services launched successfully in separate windows!
echo.
echo  👉 Webhook API (n8n):    http://localhost:8010/webhook
echo  👉 Tone Studio UI:       http://localhost:8011
echo  👉 Benchmark Suite:      http://localhost:8011/test
echo  👉 Go Queue Gateway:     http://localhost:8020
echo  👉 Python GPU Backend:   http://localhost:8021
echo  👉 VoxCPM2+SeedVC Studio: http://localhost:8013
echo  👉 SeedVC Worker:        http://localhost:8022/health
echo ================================================================
echo.
pause
