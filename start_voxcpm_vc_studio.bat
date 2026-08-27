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
py -m uvicorn app.main:app --reload --port 8013
pause
