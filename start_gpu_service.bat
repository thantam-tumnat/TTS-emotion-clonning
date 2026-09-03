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
rem
rem The normal release is no longer this timer: the Go gateway (:8020) calls
rem /v2/gpu/release the moment its queue drains, which hands the card back in
rem seconds. This is the BACKSTOP, for the work the gateway cannot see -- studio
rem renders straight to :8021, /v2/direct_render calls from anywhere else, voice
rem registration. Lower than a reload costs (~30 s) and a burst of those pays for
rem a reload between every take; the 90 s below is the compromise. Raise it back
rem to 180 if the studio path turns out to reload more than it saves.
set "SIANGTTS_IDLE_TTL=90"
rem To take the card back right now, without waiting out the timer:
rem   curl -X POST http://127.0.0.1:8021/v2/gpu/release

echo ================================================================
echo  🚀 Starting SiangTTS Python GPU Worker on port 8021...
echo ================================================================
uv run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8021
pause
