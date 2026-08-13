"""Webhook service — the whole n8n "LIveAI_Audio" flow as one process.

Same contract as the old webhook (POST a script, get `{"status":"success"}`
back immediately, receive the finished audio URL on `callback_url` later), but
the twenty nodes in between collapse into a single in-process job:

    prepare (src/thai_text) -> synth every chunk -> merge -> upload -> callback

What that buys over the n8n version: no 3-second polling granularity, no HTTP
hop per chunk, and the reference voice is encoded once per job instead of
re-read on every chunk.

Run:
    uv run uvicorn src.webhook:app --host 0.0.0.0 --port 8002

Config (env):
    SIANGTTS_ADAPTER        LoRA dir            (default checkpoints/siangtts-v1)
    SIANGTTS_BASE_MODEL     base HF id
    SIANGTTS_DEVICE         cuda / cpu          (default: auto)
    SIANGTTS_REF_DIR        reference clips     (default ref/)
                            NB: the old system kept these in
                            C:\\temp\\tts_jobs\\voices — point this there to
                            reuse them in place.
    SIANGTTS_CACHE_DIR      prompt-cache store  (default voice_cache/)
                            Derived .pt files, not audio. Deleting it only
                            costs a re-encode.
    SIANGTTS_WORK_DIR       job scratch         (default work/)
    SIANGTTS_UPLOAD_URL     upload endpoint
    SIANGTTS_UPLOAD_TOKEN   bearer for upload   (was n8n credential "VR_live Auth")
    SIANGTTS_DEFAULT_CALLBACK
    SIANGTTS_KEEP_WORK      "1" keeps job scratch dirs for debugging
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import pipeline
from .inference import DEFAULT_BASE_MODEL, Synthesizer
from .thai_text import Chunk, chunk_text, prepare_prompt

BASE_MODEL = os.environ.get("SIANGTTS_BASE_MODEL", DEFAULT_BASE_MODEL)
ADAPTER = os.environ.get("SIANGTTS_ADAPTER", "checkpoints/siangtts-v1")
DEVICE = os.environ.get("SIANGTTS_DEVICE") or None
REF_DIR = Path(os.environ.get("SIANGTTS_REF_DIR", "ref"))
CACHE_DIR = Path(os.environ.get("SIANGTTS_CACHE_DIR", "voice_cache"))
KEEP_WORK = os.environ.get("SIANGTTS_KEEP_WORK", "") == "1"

AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")

# Defaults lifted from the flow's `set` node and its send_prompt_new body.
DEFAULT_VOICE = "thai_female"
DEFAULT_CALLBACK = os.environ.get(
    "SIANGTTS_DEFAULT_CALLBACK",
    "https://test.looklike.ai/api/v1/live-gpt/n8n/audio-callback",
)
DEFAULT_REF_TEXT = (
    "ริ้วรอยลดเลือน จุดด่างดำจางลง ผิวไร้ปัญหาสิว "
    "บำรุงล้ำลึกจากภายในเซลล์ผิว ผิวกระจ่างใสเนียนสวยแบบนี้ได้ใจเลยค่ะ"
)
# n8n: num_step. The flow inherited 32 from IndexTTS, where a step is cheap.
# Here every LM step runs the flow-matching DiT this many times, doubled again by
# CFG — 32 costs 64 DiT forwards per step and put a single chunk at ~1s/step on an
# uncompiled build. voxcpm's own default is 10, and the difference is inaudible.
NUM_STEP = int(os.environ.get("SIANGTTS_NUM_STEP", "10"))
GUIDANCE = float(os.environ.get("SIANGTTS_GUIDANCE", "2"))         # n8n: guidance_scale

# Finished jobs kept for /jobs. State is in memory, so this is the only thing
# stopping a long-lived process from growing without bound.
MAX_HISTORY = int(os.environ.get("SIANGTTS_MAX_HISTORY", "500"))


# ---------------------------------------------------------------------------
# Request / job model
# ---------------------------------------------------------------------------

class WebhookBody(BaseModel):
    """Body the old webhook received. Unknown fields (session_id, section,
    label, and the IndexTTS-only knobs) are accepted and ignored."""

    prompt: str = ""
    job_id: str = ""
    queue_id: str = ""
    voice_id: str = ""
    voice_text: str = ""
    audio_speed: float = 1.0
    country_code: str = "th"
    callback_url: str = ""


@dataclass
class Job:
    job_id: str
    queue_id: str
    voice_id: str
    ref_text: str
    speed: float
    callback_url: str
    chunks: list[Chunk]
    status: str = "queued"              # queued | running | completed | failed
    error: str | None = None
    file_url: str | None = None
    done: int = 0                       # chunks synthesised so far
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def as_dict(self, position: int | None = None) -> dict:
        now = time.time()
        waited = (self.started or now) - self.created
        ran = ((self.finished or now) - self.started) if self.started else None
        return {
            "job_id": self.job_id,
            "queue_id": self.queue_id,
            "status": self.status,
            "voice_id": self.voice_id,
            "progress": f"{self.done}/{len(self.chunks)}",
            "position": position,       # place in line; None once it starts
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created)),
            "waited_s": round(waited, 1),
            "elapsed_s": round(ran, 1) if ran is not None else None,
            "file_url": self.file_url,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Voice cache
# ---------------------------------------------------------------------------

class VoiceCache:
    """Encoded reference clips, keyed by voice + ref text.

    VoxCPM2's prompt cache is bound to the transcript it was built with, so a
    voice used with two different `voice_text` values needs two caches. Built
    lazily on first use and persisted, because voice ids arrive from the caller
    and are not known at startup.
    """

    def __init__(self, synth: Synthesizer) -> None:
        self.synth = synth
        self.mem: dict[str, dict] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _ref_file(self, voice_id: str) -> Path:
        for ext in AUDIO_EXTS:
            p = REF_DIR / f"{voice_id}{ext}"
            if p.exists():
                return p
        raise RuntimeError(f"voice '{voice_id}' has no reference clip in {REF_DIR}/")

    def get(self, voice_id: str, ref_text: str) -> dict:
        digest = hashlib.sha1(ref_text.encode("utf-8")).hexdigest()[:8]
        key = f"{voice_id}-{digest}"
        if key in self.mem:
            return self.mem[key]

        path = CACHE_DIR / f"{key}.pt"
        ref_file = self._ref_file(voice_id)
        if path.exists() and path.stat().st_mtime >= ref_file.stat().st_mtime:
            cache = self.synth.load_voice(path)
        else:
            print(f"[voice] encoding '{voice_id}' from {ref_file.name} …")
            cache = self.synth.build_voice(str(ref_file), prompt_text=ref_text or None)
            self.synth.save_voice(cache, path)
        self.mem[key] = cache
        return cache


# ---------------------------------------------------------------------------
# App + worker
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    adapter = ADAPTER or None
    if adapter and not Path(adapter).exists():
        # Loud, because the service otherwise runs happily on the base model and
        # every clip comes out without the Thai LoRA.
        raise RuntimeError(
            f"adapter {adapter!r} not found — set SIANGTTS_ADAPTER, or '' for base only"
        )

    print(f"[webhook] loading {BASE_MODEL} adapter={adapter} …")
    synth = Synthesizer(base_model=BASE_MODEL, adapter_path=adapter, device=DEVICE)
    _state["synth"] = synth
    _state["voices"] = VoiceCache(synth)
    _state["jobs"] = {}
    _state["running"] = None
    _state["queue"] = asyncio.Queue()

    # First thaisum/newmm call loads its model; do it now so job #1 isn't slower.
    await asyncio.to_thread(prepare_prompt, "อุ่นเครื่องนะคะ", "th")

    worker = asyncio.create_task(_worker())
    pipeline.WORK_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[webhook] ready — sr={synth.sample_rate} work={pipeline.WORK_ROOT}")
    try:
        yield
    finally:
        worker.cancel()
        _state.clear()


app = FastAPI(title="SiangTTS webhook", version="1.0.0", lifespan=lifespan)


async def _worker() -> None:
    """Single consumer: one GPU, so jobs run strictly one at a time. Queueing
    rather than gating with a semaphore also gives the caller backpressure for
    free — bursts wait instead of thrashing VRAM."""
    queue: asyncio.Queue = _state["queue"]
    while True:
        job: Job = await queue.get()
        try:
            await _run_job(job)
        except Exception:                                        # noqa: BLE001
            traceback.print_exc()
        finally:
            queue.task_done()


async def _run_job(job: Job) -> None:
    synth: Synthesizer = _state["synth"]
    voices: VoiceCache = _state["voices"]
    work = pipeline.WORK_ROOT / job.queue_id
    job.status = "running"
    job.started = time.time()
    _state["running"] = job.job_id

    try:
        work.mkdir(parents=True, exist_ok=True)
        cache = await asyncio.to_thread(voices.get, job.voice_id, job.ref_text)

        wav_paths: list[Path] = []
        for ch in job.chunks:
            wav = await asyncio.to_thread(
                synth.synth_cached,
                ch.text,
                cache,
                cfg_value=GUIDANCE,
                inference_timesteps=NUM_STEP,
            )
            out = work / f"{ch.filename}.wav"
            await asyncio.to_thread(sf.write, str(out), wav, synth.sample_rate)
            wav_paths.append(out)
            job.done = ch.index
            print(f"[{job.queue_id}] chunk {ch.index}/{ch.total} ok")

        merged = work / f"{job.queue_id}.mp3"
        await asyncio.to_thread(
            pipeline.merge_chunks,
            wav_paths,
            merged,
            pipeline.MergeOptions(speed=job.speed),
        )

        # Absolute path, because WORK_ROOT is relative to the working directory
        # and "work/<id>.mp3" is not enough to find the file when the service was
        # started from somewhere else. Logged before the upload so the line is
        # there even when the upload is what fails.
        print(
            f"[{job.queue_id}] merged -> {merged.resolve()} "
            f"({merged.stat().st_size / 1024:.0f} KB)"
            # ASCII only: Windows consoles default to cp1252 and an em dash here
            # raises UnicodeEncodeError mid-job.
            + ("" if KEEP_WORK else "  [deleted after upload - SIANGTTS_KEEP_WORK=1 to keep]")
        )

        job.file_url = await pipeline.upload(merged)
        job.status = "completed"
        await pipeline.post_callback(
            job.callback_url,
            job_id=job.job_id,
            queue_id=job.queue_id,
            file_url=job.file_url,
            error=None,
        )
        print(f"[{job.queue_id}] done -> {job.file_url}")

    except Exception as exc:                                     # noqa: BLE001
        job.status = "failed"
        job.error = str(exc)
        traceback.print_exc()
        try:
            await pipeline.post_callback(
                job.callback_url,
                job_id=job.job_id,
                queue_id=job.queue_id,
                file_url=None,
                error=job.error,
            )
        except Exception as cb_exc:                               # noqa: BLE001
            print(f"[{job.queue_id}] error callback failed: {cb_exc}")

    finally:
        job.finished = time.time()
        _state["running"] = None
        # Only a delivered job is safe to delete. A failure usually means the
        # upload or callback broke *after* the GPU work was already done, and
        # wiping the scratch dir there throws away minutes of synthesis and the
        # only copy of the audio — leaving nothing to inspect or re-upload.
        if job.status == "completed" and not KEEP_WORK:
            pipeline.cleanup(work)
        elif job.status != "completed" and work.exists():
            print(f"[{job.queue_id}] failed - audio kept at {work.resolve()}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

async def _accept(body: WebhookBody) -> JSONResponse:
    """Validate + enqueue. Text prep runs here, not in the worker, so a bad
    script is rejected in the HTTP response instead of only via callback."""
    queue_id = body.queue_id or str(uuid.uuid4())
    callback_url = body.callback_url or DEFAULT_CALLBACK

    prompt = await asyncio.to_thread(prepare_prompt, body.prompt, body.country_code)
    if not prompt:
        # The n8n chunker returned [] here and nothing downstream fired, so the
        # webhook hung until timeout. Fail out loud instead.
        return JSONResponse(
            {"status": "error", "error": "prompt is empty after normalize"}, status_code=400
        )

    chunks = chunk_text(prompt, prefix=queue_id)
    job = Job(
        job_id=body.job_id or queue_id,
        queue_id=queue_id,
        voice_id=body.voice_id.strip() or DEFAULT_VOICE,
        ref_text=body.voice_text.strip() or DEFAULT_REF_TEXT,
        speed=body.audio_speed or 1.0,
        callback_url=callback_url,
        chunks=chunks,
    )
    jobs: dict[str, Job] = _state["jobs"]
    jobs[job.job_id] = job
    # Bounded history — the dict is insertion-ordered, so drop the oldest
    # finished jobs first and never evict anything still queued or running.
    if len(jobs) > MAX_HISTORY:
        for jid, j in list(jobs.items()):
            if len(jobs) <= MAX_HISTORY:
                break
            if j.status in ("completed", "failed"):
                del jobs[jid]

    await _state["queue"].put(job)
    print(f"[{queue_id}] queued — {len(chunks)} chunk(s), voice={job.voice_id}")
    return JSONResponse({"status": "success", "job_id": job.job_id, "chunks": len(chunks)})


@app.post("/webhook/live-ai-create-new")
async def webhook(body: WebhookBody) -> JSONResponse:
    return await _accept(body)


# The n8n webhook lived at /webhook/<path>; keep the bare path too so callers
# that were pointed straight at the node still resolve.
@app.post("/live-ai-create-new")
async def webhook_bare(body: WebhookBody) -> JSONResponse:
    return await _accept(body)


def _positions() -> dict[str, int]:
    """Place in line for everything still waiting, 1-based. The queue itself is
    opaque, but jobs are enqueued in creation order, so ordering the waiting
    ones by `created` reproduces it."""
    waiting = sorted(
        (j for j in _state.get("jobs", {}).values() if j.status == "queued"),
        key=lambda j: j.created,
    )
    return {j.job_id: i for i, j in enumerate(waiting, 1)}


@app.get("/jobs")
def list_jobs(status: str | None = None, limit: int = 50) -> JSONResponse:
    """Everything the service remembers, newest first. `?status=failed` to
    filter — this is the surface that replaces scrolling n8n's execution list."""
    jobs: list[Job] = list(_state.get("jobs", {}).values())
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.status] = counts.get(j.status, 0) + 1

    pos = _positions()
    jobs.sort(key=lambda j: j.created, reverse=True)
    if status:
        jobs = [j for j in jobs if j.status == status]
    return JSONResponse({
        "counts": counts,
        "running": _state.get("running"),
        "total": len(jobs),
        "jobs": [j.as_dict(pos.get(j.job_id)) for j in jobs[:limit]],
    })


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job: Job | None = _state.get("jobs", {}).get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return JSONResponse(job.as_dict(_positions().get(job_id)))


@app.get("/health")
def health() -> JSONResponse:
    synth = _state.get("synth")
    queue: asyncio.Queue | None = _state.get("queue")
    jobs: list[Job] = list(_state.get("jobs", {}).values())
    return JSONResponse({
        "status": "ok" if synth else "loading",
        "adapter": ADAPTER,
        "sample_rate": getattr(synth, "sample_rate", None),
        "waiting": queue.qsize() if queue else None,
        "running": _state.get("running"),
        "completed": sum(1 for j in jobs if j.status == "completed"),
        "failed": sum(1 for j in jobs if j.status == "failed"),
        "voices_cached": len(getattr(_state.get("voices"), "mem", {})),
        "upload_token": bool(pipeline.UPLOAD_TOKEN),
    })


# ---------------------------------------------------------------------------
# Queue page
# ---------------------------------------------------------------------------

# One self-contained file, no build step and no CDN — this runs on a LAN box
# that may have no outbound internet, and a dashboard that needs a network
# fetch to render is a dashboard that fails exactly when you need it. It polls
# /jobs, so everything shown here is also available as JSON.
QUEUE_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SiangTTS · queue</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #fbfbfa; --fg: #1c1b1a; --dim: #6d6a67;
    --line: #e3e1de; --card: #fff;
    --run: #2563eb; --ok: #15803d; --fail: #b91c1c; --wait: #a16207;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #191817; --fg: #eceae7; --dim: #9a9691;
      --line: #322f2c; --card: #201f1d;
      --run: #60a5fa; --ok: #4ade80; --fail: #f87171; --wait: #fbbf24;
    }
  }
  * { box-sizing: border-box }
  body {
    margin: 0; padding: 1.5rem; background: var(--bg); color: var(--fg);
    font: 15px/1.5 ui-sans-serif, system-ui, "Segoe UI", "Sarabun", sans-serif;
  }
  header { display: flex; align-items: baseline; gap: .75rem; flex-wrap: wrap;
           margin-bottom: 1.25rem }
  h1 { font-size: 1.15rem; margin: 0; font-weight: 600 }
  .meta { color: var(--dim); font-size: .85rem }
  .stats { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem }
  .stat { background: var(--card); border: 1px solid var(--line);
          border-radius: 8px; padding: .5rem .8rem; min-width: 5.5rem }
  .stat b { display: block; font-size: 1.35rem; font-weight: 600;
            font-variant-numeric: tabular-nums; line-height: 1.2 }
  .stat span { color: var(--dim); font-size: .75rem; text-transform: uppercase;
               letter-spacing: .04em }
  .wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px;
          background: var(--card) }
  table { border-collapse: collapse; width: 100%; font-size: .875rem }
  th, td { text-align: left; padding: .55rem .7rem; white-space: nowrap;
           border-bottom: 1px solid var(--line) }
  th { color: var(--dim); font-weight: 500; font-size: .75rem;
       text-transform: uppercase; letter-spacing: .04em }
  tr:last-child td { border-bottom: 0 }
  td.num { font-variant-numeric: tabular-nums }
  code { font: 12px/1 ui-monospace, "Cascadia Mono", Consolas, monospace;
         color: var(--dim) }
  .badge { font-size: .75rem; font-weight: 600 }
  .running { color: var(--run) } .completed { color: var(--ok) }
  .failed  { color: var(--fail) } .queued  { color: var(--wait) }
  .err { color: var(--fail); max-width: 28rem; white-space: normal }
  a { color: var(--run) }
  .empty { padding: 2.5rem; text-align: center; color: var(--dim) }
</style>
<header>
  <h1>SiangTTS queue</h1>
  <span class="meta" id="meta">connecting…</span>
</header>
<div class="stats" id="stats"></div>
<div class="wrap"><table>
  <thead><tr>
    <th>#</th><th>status</th><th>job</th><th>voice</th><th>chunks</th>
    <th>created</th><th>waited</th><th>elapsed</th><th>result</th>
  </tr></thead>
  <tbody id="rows"></tbody>
</table></div>
<script>
// Escape before interpolating: job ids and voice ids arrive from the caller's
// POST body, so they are untrusted input, not our own strings.
const esc = s => String(s ?? "").replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const secs = v => v == null ? "—" : v < 60 ? v.toFixed(1) + "s"
  : Math.floor(v / 60) + "m " + Math.round(v % 60) + "s";

// Long ids are noise at a glance but you still want the whole thing when
// grepping logs — show the head, keep the rest in the title attribute.
const shortId = s => !s ? "—" :
  `<code title="${esc(s)}">${esc(s.length > 10 ? s.slice(0, 8) + "…" : s)}</code>`;

const STATES = ["running", "queued", "completed", "failed"];

function render(d) {
  document.getElementById("stats").innerHTML = STATES.map(s =>
    `<div class="stat"><b class="${s}">${d.counts[s] || 0}</b><span>${s}</span></div>`
  ).join("");

  const rows = d.jobs.map(j => `<tr>
    <td class="num">${j.position ?? ""}</td>
    <td class="badge ${esc(j.status)}">${esc(j.status)}</td>
    <td>${shortId(j.job_id)}</td>
    <td>${shortId(j.voice_id)}</td>
    <td class="num">${esc(j.progress)}</td>
    <td class="num">${esc(j.created)}</td>
    <td class="num">${secs(j.waited_s)}</td>
    <td class="num">${secs(j.elapsed_s)}</td>
    <td>${j.error ? `<span class="err">${esc(j.error)}</span>`
        : j.file_url ? `<a href="${esc(j.file_url)}" target="_blank"
                          rel="noopener noreferrer">audio</a>` : "—"}</td>
  </tr>`).join("");

  document.getElementById("rows").innerHTML = rows ||
    `<tr><td colspan="9" class="empty">no jobs yet</td></tr>`;
  document.getElementById("meta").textContent =
    `${d.total} job(s) · updated ${new Date().toLocaleTimeString()}`;
}

async function tick() {
  try {
    const r = await fetch("jobs?limit=100", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    render(await r.json());
  } catch (e) {
    // Say so rather than freezing on stale numbers — a queue page that looks
    // alive while the service is down is worse than no page.
    document.getElementById("meta").textContent = "disconnected — " + e.message;
  }
}
tick();
setInterval(tick, 2000);
</script>
"""


@app.get("/", response_class=HTMLResponse)
def queue_page() -> HTMLResponse:
    """Live view of the queue — the thing n8n's execution list used to be."""
    return HTMLResponse(QUEUE_PAGE)
