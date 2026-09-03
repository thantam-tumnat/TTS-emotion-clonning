@echo off
title [2/3] SiangTTS Go Fiber Queue Gateway (:8020)
echo ================================================================
echo  🚀 Starting Go Fiber Queue Gateway on port 8020...
echo  👉 Target Python GPU: http://127.0.0.1:8021
echo  👉 Dashboard UI:      http://127.0.0.1:8020/
echo ================================================================

rem --- Workspace-local Go toolchain (nothing is installed system-wide) ---
set "GOROOT=%~dp0.tools\go"
set "GOPATH=%~dp0.tools\gopath"
set "GOMODCACHE=%~dp0.tools\gopath\pkg\mod"
set "PATH=%GOROOT%\bin;%PATH%"

if not exist "%GOROOT%\bin\go.exe" (
    echo ERROR: no Go toolchain at "%GOROOT%".
    echo        Re-extract go1.27.0.windows-amd64.zip into .tools\ so that
    echo        .tools\go\bin\go.exe exists, then run this again.
    echo.
    pause
    exit /b 1
)

cd /d "%~dp0voice-clonning-queue-go"
set PORT=8020
set PYTHON_GPU_URL=http://127.0.0.1:8021

rem --- Handing the GPU back when the queue drains ---
rem This gateway is the only process that can tell "the batch is finished" from
rem "the next take is two seconds away" -- the Python service is dispatched via
rem /v2/direct_render, which bypasses its own queue, so it cannot. When the line
rem empties, the worker asks both GPU services to unload, which returns ~7.8 GB
rem within seconds instead of waiting out their idle timers. Fire-and-forget: a
rem service that is down or busy just keeps its weights until its timer fires.
rem SEEDVC_URL is used ONLY for that call, never to dispatch work.
set SEEDVC_URL=http://127.0.0.1:8022
rem Seconds to wait after the queue drains before asking, re-checked against the
rem queue afterwards. The default 2 exists to skip a release when the next take
rem is a heartbeat away -- but with the card wanted back for another service the
rem moment a render is done, 0 is the right call: release instantly when the line
rem empties. The cost is a ~30 s reload if a job lands within that gap; that is
rem the deliberate price of the card being free in between. Raise it back toward
rem 1-2 if reloads between closely-spaced takes turn out to hurt more than the
rem freed VRAM helps.
set "GPU_RELEASE_DELAY=0"

rem --- Name whoever already holds :8020, instead of Go's cryptic bind error ---
for /f "tokens=5" %%p in ('netstat -ano -p tcp ^| findstr /r /c:"LISTENING" ^| findstr /c:"127.0.0.1:8020"') do (
    echo ERROR: port 8020 is already held by PID %%p -- most likely an older
    echo        gateway still running the previous build. Stop it first:
    echo.
    echo            taskkill /PID %%p /F
    echo.
    pause
    exit /b 1
)

rem --- Hot-reload via air when present; a plain one-shot "go run" otherwise ---
if exist "%GOPATH%\bin\air.exe" (
    echo [dev] air detected -- every .go save rebuilds and restarts automatically.
    "%GOPATH%\bin\air.exe"
) else (
    echo [dev] air not found -- starting once, WITHOUT hot-reload.
    echo       Edits to .go files will not take effect until you restart.
    echo       Install it with: go install github.com/air-verse/air@latest
    go run main.go
)
pause
