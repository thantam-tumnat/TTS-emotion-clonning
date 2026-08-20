@echo off
title Thai TTS Tone Studio (Port 8011)
echo Starting Thai TTS Tone Studio on port 8011...
uvicorn app.main:app --host 0.0.0.0 --port 8011 --reload
pause
