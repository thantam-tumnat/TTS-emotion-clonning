@echo off
title SiangTTS VoxCPM2+SeedVC Studio (:8013)
cd /d "%~dp0voxcpm+vc"
echo ================================================================
echo  Starting VoxCPM2 + SeedVC Studio on http://localhost:8013
echo.
echo  Emotion : cloned from a donor recording by VoxCPM2
echo  Timbre  : swapped to the target voice by SeedVC
echo.
echo  Target Queue Gateway: http://127.0.0.1:8020  (GPU :8021)
echo  Target SeedVC worker: http://127.0.0.1:8022
echo ================================================================
REM --reload-include ".env" makes a .env edit (e.g. SEEDVC_F0_MODE, WEBHOOK_USE_LLM)
REM restart the server so the new value applies -- default reload only watches *.py.
REM We pass "*.py" too so python files keep triggering reload (some uvicorn versions
REM otherwise replace the default include). Dev only: a reload drops the in-memory
REM webhook queue.
py -m uvicorn app.main:app --reload --reload-include ".env" --reload-include "*.py" --port 8013
pause
