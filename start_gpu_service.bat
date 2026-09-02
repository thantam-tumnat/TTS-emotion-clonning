@echo off
title [1/3] SiangTTS Python GPU Worker (:8021)
cd /d "%~dp0voice-cloning"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

rem --- VRAM (this process shares the card with the SeedVC worker on :8022) ---
rem SIANGTTS_MEM_FRACTION caps this process's share so an over-budget process
rem fails instead of starving whichever one allocates second. Pick the split for
rem your card (VoxCPM2 needs ~8 GB); leave unset to run uncapped.
rem   set "SIANGTTS_MEM_FRACTION=0.60"
rem VOXCPM_RECLAIM_BELOW is the free-VRAM fraction under which empty_cache() runs
rem after a job (default 0.15). VOXCPM_EMPTY_CACHE=always restores the old
rem reclaim-after-every-job behaviour.
rem   set "VOXCPM_RECLAIM_BELOW=0.15"

echo ================================================================
echo  🚀 Starting SiangTTS Python GPU Worker on port 8021...
echo ================================================================
uv run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8021
pause
