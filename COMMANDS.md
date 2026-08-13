# Command reference

Every command runs from the project root:

```
C:\Users\opendream002\Desktop\SIANGTTS\VoxCPM-thai
```

`DEPLOY.md` explains *why*; this file is just the commands.

---

## Pre-flight

| check | command | want to see |
|---|---|---|
| ffmpeg | `ffmpeg -version` | version 7.x |
| GPU | `nvidia-smi` | ~8 GB free (the 3090 is shared with IndexTTS) |
| LoRA adapter | `dir checkpoints\siangtts-v1\lora_weights.safetensors` | 144,749,240 bytes |
| base model cached | `dir "%USERPROFILE%\.cache\huggingface\hub"` | a `models--openbmb--VoxCPM2` folder |
| reference voices | `dir ref` | one file per `voice_id` callers send |

Download the base model ahead of time (4.7 GB) so the first start isn't a surprise:

```
uv run hf download openbmb/VoxCPM2
```

---

## Install / update dependencies

```
uv sync --extra serve
```

Add `--extra dev` too if you want to run the tests on this machine.

---

## Environment

`cmd` — lasts only for that window:

```
set SIANGTTS_ADAPTER=checkpoints/siangtts-v1
set SIANGTTS_UPLOAD_TOKEN=<bearer>
set PYTHONIOENCODING=utf-8
```

PowerShell:

```
$env:SIANGTTS_ADAPTER = "checkpoints/siangtts-v1"
$env:SIANGTTS_UPLOAD_TOKEN = "<bearer>"
$env:PYTHONIOENCODING = "utf-8"
```

Permanent, for the machine (needs an elevated prompt, and a new window to take effect):

```
setx SIANGTTS_ADAPTER "checkpoints/siangtts-v1" /M
```

### All variables

| variable | default | what it does |
|---|---|---|
| `SIANGTTS_ADAPTER` | `checkpoints/siangtts-v1` | LoRA directory. **Missing = refuses to start.** `""` runs the base model with no Thai LoRA. |
| `SIANGTTS_BASE_MODEL` | `openbmb/VoxCPM2` | base HF id |
| `SIANGTTS_DEVICE` | auto | `cuda` / `cpu` |
| `SIANGTTS_UPLOAD_TOKEN` | — | bearer for the upload endpoint. Unset = every job fails at upload. |
| `SIANGTTS_UPLOAD_URL` | `https://looklike.ai/api/v1/live-gpt/upload` | where the merged mp3 goes |
| `SIANGTTS_DEFAULT_CALLBACK` | `https://test.looklike.ai/.../audio-callback` | used when the caller omits `callback_url` |
| `SIANGTTS_REF_DIR` | `ref` | reference clips, named `<voice_id>.mp3`. The old server kept these in `C:\temp\tts_jobs\voices\` — point here to reuse them in place. |
| `SIANGTTS_CACHE_DIR` | `voice_cache` | cached encodings (`.pt`), derived from `ref/`. Safe to delete; costs a re-encode. |
| `SIANGTTS_WORK_DIR` | `work` | job scratch. **Relative to the working directory** — set an absolute path when running as a service. |
| `SIANGTTS_KEEP_WORK` | — | `1` keeps `work/<queue_id>/` instead of deleting it |
| `SIANGTTS_NUM_STEP` | `10` | inference steps (was `num_step` in n8n, where it was `32` — see DEPLOY.md) |
| `SIANGTTS_GUIDANCE` | `2` | CFG scale (was `guidance_scale`) |
| `SIANGTTS_MAX_HISTORY` | `500` | finished jobs kept for `/jobs` |
| `SIANGTTS_HTTP_TIMEOUT` | `120` | seconds for upload + callback |
| `SIANGTTS_FFMPEG` | `ffmpeg` | path to the binary if it isn't on PATH |
| `PYTHONIOENCODING` | — | **set to `utf-8`** or Thai log lines crash the console on Windows |

---

## Run

Testing — localhost only, no one else can reach it:

```
uv run uvicorn src.webhook:app --host 127.0.0.1 --port 8010
```

Reachable from other machines (**the service has no authentication** — make sure the firewall blocks 8010 from the internet):

```
uv run uvicorn src.webhook:app --host 0.0.0.0 --port 8010
```

Ready when the console prints `[webhook] ready — sr=48000 work=work`. Model load takes 30–60 s; the port isn't listening before that. Stop with `Ctrl+C`.

Do **not** add `--workers N`: each worker loads its own copy of the model into VRAM. Concurrency is handled inside the process by the job queue.

### As a Windows service

```
nssm install SiangTTS "C:\Users\opendream002\.local\bin\uv.exe" "run uvicorn src.webhook:app --host 0.0.0.0 --port 8010"
```
```
nssm set SiangTTS AppDirectory "C:\Users\opendream002\Desktop\SIANGTTS\VoxCPM-thai"
```
```
nssm set SiangTTS AppEnvironmentExtra SIANGTTS_ADAPTER=checkpoints/siangtts-v1 SIANGTTS_UPLOAD_TOKEN=<bearer> PYTHONIOENCODING=utf-8
```
```
nssm start SiangTTS
```

`nssm restart SiangTTS` · `nssm stop SiangTTS` · `nssm edit SiangTTS`

`AppDirectory` matters: `work/`, `ref/`, `voice_cache/` and the adapter path are all relative to it.

---

## Test

Health:

```
curl http://localhost:8010/health
```

Create audio (cmd — inner quotes must be escaped):

```
curl -X POST http://localhost:8010/webhook/live-ai-create-new -H "Content-Type: application/json" -d "{\"queue_id\":\"smoke1\",\"prompt\":\"สนใจสินค้าตัวไหน กดที่ตะกร้าได้เลยนะคะ\",\"voice_id\":\"demo_female\",\"callback_url\":\"https://webhook.site/xxxx\"}"
```

PowerShell is easier for JSON:

```
Invoke-RestMethod -Method Post http://localhost:8010/webhook/live-ai-create-new -ContentType "application/json" -Body '{"queue_id":"smoke1","prompt":"สนใจสินค้าตัวไหน","voice_id":"demo_female","callback_url":"https://webhook.site/xxxx"}'
```

Or import `SiangTTS.postman_collection.json` into Postman — 7 requests, ready to go.

Browser: `http://localhost:8010/docs` is a full Swagger UI you can fire requests from.

---

## Watch the queue

Open <http://localhost:8010/> in a browser — live table, polls every 2 s, no
build step and no CDN (works on a box with no outbound internet). Everything on
it comes from `/jobs`, so the CLI below shows the same data.

```
curl http://localhost:8010/jobs
```
```
curl http://localhost:8010/jobs/smoke1
```
```
curl "http://localhost:8010/jobs?status=failed"
```

PowerShell, as a table:

```
(Invoke-RestMethod http://localhost:8010/jobs).jobs | Format-Table job_id,status,progress,position,waited_s,elapsed_s
```

Live view, refreshing every 3 s:

```
while ($true) { Clear-Host; Invoke-RestMethod http://localhost:8010/health | Format-List; Start-Sleep 3 }
```

Failures with their reasons:

```
(Invoke-RestMethod "http://localhost:8010/jobs?status=failed").jobs | Format-Table job_id,created,error -Wrap
```

---

## Voices

Register a voice — the filename *is* the `voice_id`:

```
copy C:\temp\tts_jobs\voices\<voice_id>.mp3 ref\
```

No restart needed; it is encoded on first use and cached.

List what's cached:

```
dir voices
```

Force a re-encode (after replacing a reference clip):

```
del voices\<voice_id>-*.pt
```

The cache key is `<voice_id>-<hash of voice_text>`, so the same voice sent with different `voice_text` produces more than one `.pt`. That is expected — a prompt cache is bound to the transcript it was built with.

---

## Output files

```
work\<queue_id>\<queue_id>_000.wav     chunk 1, 48 kHz
work\<queue_id>\<queue_id>_001.wav     chunk 2 …
work\<queue_id>\<queue_id>.mp3         merged, 192 kbps — this is what gets uploaded
```

Deleted after the callback fires, unless `SIANGTTS_KEEP_WORK=1`.

Listen to the newest result:

```
start work\smoke1\smoke1.mp3
```

Sweep leftovers if a run crashed hard:

```
rmdir /s /q work
```

---

## Development

```
uv run --extra dev pytest -q
```
```
uv run --extra dev pytest tests/test_thai_text.py -q
```
```
uvx ruff check src
```

Check text preparation without touching the GPU — how a script gets expanded and split:

```
uv run python -c "from src.thai_text import prepare_prompt, chunk_text; t=prepare_prompt('ราคา 250 บาท ลดเหลือ 199 บาทค่ะ','th'); print(t); [print(c.filename, len(c.text), c.text) for c in chunk_text(t,'demo')]"
```

---

## Update the code

With git:

```
git pull
```
```
uv sync --extra serve
```

Without git — extract `siangtts-patch.zip` over the project root, choosing Replace, then:

```
uv sync --extra serve
```

Restart the service afterwards either way. Confirm the new code is in place:

```
dir src\webhook.py src\thai_text.py src\pipeline.py tests\test_thai_text.py
```

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `RuntimeError: adapter ... not found` | LoRA missing at that path | check `dir checkpoints\siangtts-v1` |
| `voice 'x' has no reference clip in ref/` | no `ref\x.mp3` | copy the clip in, or send a different `voice_id` |
| `SIANGTTS_UPLOAD_TOKEN is not set` | env not set | set it, restart — or accept it while testing and use `SIANGTTS_KEEP_WORK=1` |
| `ffmpeg merge failed - no audio generated` | every chunk failed, or the prompt was empty | look further up the console for the real error |
| `status: "loading"` on /health | model still loading | wait 30–60 s |
| `UnicodeEncodeError` in the console | Windows cp1252 | `set PYTHONIOENCODING=utf-8`, restart |
| CUDA out of memory | IndexTTS is holding the GPU | `nvidia-smi`, free VRAM before starting |
| port 8010 in use | already running | `netstat -ano | findstr :8010` then `taskkill /PID <pid> /F` |
| queue stuck, nothing progressing | a job is wedged | restart — there is no per-job cancel |
