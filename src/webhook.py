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
    SIANGTTS_VOICES_DIR     prompt-cache store  (default voices/)
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
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import pipeline
from .inference import DEFAULT_BASE_MODEL, Synthesizer
from .thai_text import Chunk, chunk_text, prepare_prompt

BASE_MODEL = os.environ.get("SIANGTTS_BASE_MODEL", DEFAULT_BASE_MODEL)
ADAPTER = os.environ.get("SIANGTTS_ADAPTER", "checkpoints/siangtts-v1")
DEVICE = os.environ.get("SIANGTTS_DEVICE") or None
REF_DIR = Path(os.environ.get("SIANGTTS_REF_DIR", "ref"))
VOICES_DIR = Path(os.environ.get("SIANGTTS_VOICES_DIR", "voices"))
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
NUM_STEP = int(os.environ.get("SIANGTTS_NUM_STEP", "32"))          # n8n: num_step
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
        VOICES_DIR.mkdir(parents=True, exist_ok=True)

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

        path = VOICES_DIR / f"{key}.pt"
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
        if not KEEP_WORK:
            pipeline.cleanup(work)


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
