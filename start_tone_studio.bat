@echo off
title [3/3] SiangTTS Tone Studio Web App (:8011)
cd /d "%~dp0voice-cloning-with-tones"
echo ================================================================
echo  🚀 Starting SiangTTS Tone Studio on http://localhost:8011...
echo  👉 Target Queue Gateway: http://127.0.0.1:8020
echo ================================================================
py -m uvicorn app.main:app --reload --port 8011
pause
