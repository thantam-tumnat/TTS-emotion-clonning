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

rem --- Idle policy (this is what keeps the card free for everything else) ---
rem The weights are not loaded at startup. The first request that needs them
rem loads them, and they are dropped again once the queue has been quiet --
rem handing ~6 GB back to the driver for the SeedVC worker, AI Live Studio, or
rem whatever else wants the card. "hot" restores the old always-resident
rem behaviour; use it if a reload turns out to cost more than the memory is worth.
rem   set "SIANGTTS_IDLE_MODE=hot"
rem Seconds of quiet before the weights are dropped (default 180). Keep it longer
rem than one generate-then-convert round trip, or every take pays for a reload.
rem   set "SIANGTTS_IDLE_TTL=180"
rem To take the card back right now, without waiting out the timer:
rem   curl -X POST http://127.0.0.1:8021/v2/gpu/release

echo ================================================================
echo  🚀 Starting SiangTTS Python GPU Worker on port 8021...
echo ================================================================
uv run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8021
pause
