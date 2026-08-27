@echo off
title [1/3] SiangTTS Python GPU Worker (:8021)
cd /d "%~dp0voice-cloning"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ================================================================
echo  🚀 Starting SiangTTS Python GPU Worker on port 8021...
echo ================================================================
uv run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8021
pause
