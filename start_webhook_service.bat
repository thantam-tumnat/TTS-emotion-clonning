@echo off
title [Webhook] SiangTTS Production Webhook (:8010)
cd /d "%~dp0voice-cloning"
echo ================================================================
echo  🚀 Starting SiangTTS Production Webhook on port 8010...
echo  👉 Target Queue Gateway: http://127.0.0.1:8020
echo ================================================================
uv run uvicorn src.webhook:app --host 0.0.0.0 --port 8010
pause
