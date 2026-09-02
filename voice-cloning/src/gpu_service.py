"""SiangTTS GPU Service — the only process that loads the model.

Split out of `src/webhook.py` so the model is loaded once and shared, instead of one
copy per pipeline. Both callers are HTTP clients of this service:

    n8n / LiveAI ──► :8010 webhook   ──┐
                                       ├──► :8020 (this)  VoxCPM2 + Thai LoRA ×1
    browser      ──► :8011 tone studio ┘

Nothing here knows about scripts, chunk sizing, ffmpeg, uploads, callbacks, emotion
tags or audio post-processing. Those stay with the pipeline that owns them; this
service takes a list of ready-to-speak strings and a voice, and gives back audio.

Run:
    uv run uvicorn src.gpu_service:app --host 127.0.0.1 --port 8020

Bind to localhost. There is no authentication, and a render job can name the folder
it writes into.

Config (env):
    SIANGTTS_GPU_STUB       "1" runs without the model — see src/stub_synth.py
    SIANGTTS_BASE_MODEL     base HF id
    SIANGTTS_ADAPTER        LoRA dir ("" for base only)
    SIANGTTS_DEVICE         cuda / cpu (default: auto)
    SIANGTTS_REF_DIR        reference clips              (default ref/)
    SIANGTTS_CACHE_DIR      prompt-cache store           (default voices/ or voice_cache/)
    SIANGTTS_WORK_DIR       where "files" output lands   (default work/)
    SIANGTTS_SEED_TEXT      line used to mint the neutral seed voice
    SIANGTTS_DEFAULT_LORA   lora mode when a job does not ask (default "shipped")
    SIANGTTS_IDLE_MODE      "unload" (default) drops the weights when the queue
                            has been quiet; "hot" keeps them resident forever
    SIANGTTS_IDLE_TTL       seconds of quiet before that happens (default 180)
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from pydantic import BaseModel, Field

from . import lora as lora_mod
from .engine import (
    IDLE_MODE,
    INTERACTIVE_BURST,
    LANES,
    Engine,
    ModelHolder,
    RenderJob,
    reclaim_vram,
)
from .env_file import load_env_file
from .voices import UnknownVoice, VoiceStore

# Before anything below reads os.environ. The adapter path and the reference
# directories used to be the webhook's business, and production keeps them in
# `.env`; they are this service's business now, so it has to read the same file or
# it comes up on the defaults with every production voice missing.
load_env_file()

STUB = os.environ.get("SIANGTTS_GPU_STUB", "") == "1"
BASE_MODEL = os.environ.get("SIANGTTS_BASE_MODEL", "openbmb/VoxCPM2")
ADAPTER = os.environ.get("SIANGTTS_ADAPTER", "checkpoints/siangtts-v1")
DEVICE = os.environ.get("SIANGTTS_DEVICE") or None

# One entry, or several separated by the platform's path separator (`;` on Windows).
# Several, because the two pipelines arrived with their own reference folders and the
# service now answers for both: the webhook's voices live in ref/ and the legacy
# C:\temp\tts_jobs\voices, the tone studio's in its own ref/. Listing only the first
# would make the others vanish from every voice picker in the system.
#
# The first entry is where newly registered clips are written.
REF_DIRS = [
    Path(p.strip())
    for p in os.environ.get("SIANGTTS_REF_DIR", "ref").split(os.pathsep)
    if p.strip()
] or [Path("ref")]
REF_DIR = REF_DIRS[0]
# Same default the webhook used, so the .pt files already in voices/ stay hits.
CACHE_DIR = Path(
    os.environ.get("SIANGTTS_CACHE_DIR")
    or os.environ.get("SIANGTTS_VOICES_DIR")
    or ("voices" if Path("voices").exists() else "voice_cache")
)
WORK_DIR = Path(os.environ.get("SIANGTTS_WORK_DIR", "work"))
SEED_TEXT = os.environ.get("SIANGTTS_SEED_TEXT", "วันนี้อากาศปกติ อุณหภูมิยี่สิบห้าองศา")
DEFAULT_LORA = os.environ.get("SIANGTTS_DEFAULT_LORA", lora_mod.DEFAULT_MODE)

# The old system kept reference clips here; the webhook checked it as a fallback and
# voice ids in production still resolve through it.
LEGACY_REF_DIR = Path("C:/temp/tts_jobs/voices")

MAX_WAIT_S = float(os.environ.get("SIANGTTS_MAX_WAIT", "600"))

# Hard ceiling on this process's share of the card, as a fraction. The SeedVC
# worker on :8022 is on the same GPU; without a cap the two race for whatever is
# free and whoever allocates second dies, so the OOM lands on an arbitrary victim
# instead of on the process that is actually over budget. "" disables it.
MEM_FRACTION = os.environ.get("SIANGTTS_MEM_FRACTION", "").strip()

_state: dict = {}


def _ref_dirs() -> list[Path]:
    """Every directory searched for reference clips, first match wins.

    Deduplicated by resolved path so naming the legacy directory explicitly does not
    make every voice in it appear twice in the listing.
    """
    dirs: list[Path] = []
    seen: set[str] = set()
    for d in [*REF_DIRS, LEGACY_REF_DIR]:
        try:
            key = str(d.resolve()).lower()
        except OSError:                                    # pragma: no cover
            key = str(d).lower()
        if key in seen:
            continue
        if d is LEGACY_REF_DIR and not d.exists():
            continue
        seen.add(key)
        dirs.append(d)
    return dirs


def _build_synth() -> Any:
    if STUB:
        from .stub_synth import StubSynthesizer

        print("[gpu] STUB MODE — no model, no GPU, output is a test tone")
        return StubSynthesizer()

    from .inference import Synthesizer

    adapter = ADAPTER or None
    if adapter and not Path(adapter).exists():
        # Loud, because the service otherwise runs happily on the base model and
        # every clip comes out without the Thai LoRA.
        raise RuntimeError(
            f"adapter {adapter!r} not found — set SIANGTTS_ADAPTER, or '' for base only"
        )
    print(f"[gpu] loading {BASE_MODEL} adapter={adapter} device={DEVICE or 'auto'} …")
    # The cap goes on before the weights, not after: set_per_process_memory_fraction
    # only bounds allocations made after the call, so applying it to a model that
    # is already resident caps nothing that matters. It lives here rather than at
    # startup for the same reason the load does -- it creates this process's CUDA
    # context, and an idle service should not be holding one for a model it has
    # not loaded.
    _apply_memory_fraction()
    import torch                                       # noqa: F401  (for the probe below)

    # Timed, and the VRAM delta reported, because both numbers are inputs to a
    # decision rather than trivia: how long a load costs sets the idle TTL that
    # is worth paying for, and what the weights actually occupy says whether
    # releasing the card between jobs buys the other GPU applications on this
    # box enough room to matter.
    t0 = time.time()
    before = _vram_stats(allow_init=True)
    synth = Synthesizer(base_model=BASE_MODEL, adapter_path=adapter, device=DEVICE)
    after = _vram_stats(allow_init=True)
    took = time.time() - t0
    if before and after:
        print(
            f"[gpu] loaded in {took:.1f}s — weights {after['reserved_gb'] - before['reserved_gb']:.2f} GB, "
            f"card {after['free_gb']:.2f}/{after['total_gb']:.2f} GB free"
        )
    else:
        print(f"[gpu] loaded in {took:.1f}s")
    return synth


def _vram_stats(allow_init: bool = False) -> Optional[dict]:
    """What this process is holding on the card, for /health.

    Reported because the two model processes on this box can only be sized
    against each other if both say what they are actually using -- `reserved` is
    the number that matters, since the caching allocator does not give it back.

    Answers None rather than initialising CUDA, unless the caller says otherwise.
    /health is polled every 1.5 s by the dashboard, and `mem_get_info()` creates
    this process's CUDA context -- several hundred MB of the very card the
    service has just handed back, taken for a model it is not even holding.
    """
    torch = sys.modules.get("torch")
    if torch is None:                       # nothing here has touched the GPU yet
        return None
    try:
        if not torch.cuda.is_available():
            return None
        if not allow_init and not torch.cuda.is_initialized():
            return None
        free, total = torch.cuda.mem_get_info()
        return {
            "free_gb": round(free / 1024**3, 2),
            "total_gb": round(total / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        }
    except Exception:                                              # noqa: BLE001
        return None


def _apply_memory_fraction() -> None:
    if not MEM_FRACTION:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(MEM_FRACTION))
            print(f"[gpu] VRAM capped at {float(MEM_FRACTION):.0%} of the card")
    except Exception as e:                                         # noqa: BLE001
        print(f"[gpu] could not cap VRAM ({e}); running uncapped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nothing is loaded here. The service comes up holding no VRAM at all and
    # builds the model on the first request that actually needs it, so the card
    # belongs to whatever else is running on this box until there is work.
    holder = ModelHolder(
        _build_synth,
        # A stub has no weights to give back and reloads for free, so releasing
        # one is all cost and no benefit.
        mode="hot" if STUB else IDLE_MODE,
        is_stub=STUB,
    )
    voices = VoiceStore(holder, CACHE_DIR, _ref_dirs(), seed_text=SEED_TEXT)
    engine = Engine(holder, voices, WORK_DIR, default_lora=DEFAULT_LORA)
    engine.start()
    _state["engine"] = engine
    print(
        f"[gpu] ready — no model loaded (idle_mode={holder.mode}"
        f"{'' if holder.mode == 'hot' else f' ttl={holder.ttl:.0f}s'}) "
        f"stub={STUB} cache={CACHE_DIR} work={WORK_DIR} "
        f"refs={[str(d) for d in _ref_dirs()]}"
    )
    try:
        yield
    finally:
        engine.stop()
        _state.clear()


app = FastAPI(title="SiangTTS GPU Service", version="1.0.0", lifespan=lifespan)


def _engine() -> Engine:
    return _state["engine"]


# ---------------------------------------------------------------------------
# Render jobs
# ---------------------------------------------------------------------------

class VoiceSpec(BaseModel):
    """How to condition the generation. At most one of these is meaningful.

    `handle`    a voice this service already holds
    `speaker_id` a named clip in the reference directories
    `seed`      the shared neutral voice, for unpinned multi-chunk requests
    (none)      unconditioned — VoxCPM2 picks a speaker per chunk
    """

    handle: Optional[str] = None
    speaker_id: Optional[str] = None
    ref_text: Optional[str] = None
    # Use a `<clip>.txt` beside the clip as its transcript. The caller decides,
    # because only it knows whether the ref_text it sent was real or a stand-in.
    allow_sidecar: bool = True
    seed: bool = False


class OutputSpec(BaseModel):
    """`files` writes WAVs into the shared work dir and returns their paths — for a
    caller that is going to hand them to ffmpeg anyway, which is every byte of audio
    the webhook path would otherwise push through HTTP for nothing.

    `npz` returns float32 arrays, which is what a caller assembling audio in memory
    wants."""

    mode: str = "npz"
    job_dir: Optional[str] = None       # a folder *name*, resolved under the work root
    names: list[str] = Field(default_factory=list)


class RenderRequest(BaseModel):
    raw_prompt: Optional[str] = None
    prompt: Optional[str] = None
    chunks: list[str]
    voice: Optional[VoiceSpec] = None
    cfg_value: float = Field(default=2.0, ge=0.5, le=5.0)
    timesteps: int = Field(default=10, ge=1, le=50)
    # "shipped" | "tones" | "off" | {"lm": 2.0, "dit": 0.0} — see src/lora.py
    lora: Any = None
    output: OutputSpec = Field(default_factory=OutputSpec)
    lane: str = "batch"
    client: str = ""
    job_id: Optional[str] = None


def _payload_response(job: RenderJob) -> Response:
    """Hand back the audio and free it — a delivered payload has no second reader."""
    data = job.payload or b""
    job.payload = None
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "X-Job-Id": job.job_id,
            "X-Sample-Rate": str((job.result or {}).get("sample_rate", "")),
            "X-Chunks": str((job.result or {}).get("chunks", "")),
        },
    )


@app.post("/v2/direct_render")
async def direct_render(req: RenderRequest) -> Response:
    """Execute a render job directly on the GPU without queuing.
    Used by the Go Queue Gateway (:8020) to dispatch scheduled jobs."""
    engine = _engine()
    chunks = [c for c in req.chunks if c and c.strip()]
    if not chunks:
        return JSONResponse({"error": "chunks is empty"}, status_code=400)
    if req.output.mode not in ("npz", "files"):
        return JSONResponse({"error": f"unknown output mode {req.output.mode!r}"}, status_code=400)

    job = RenderJob(
        chunks=chunks,
        voice=req.voice.model_dump() if req.voice else None,
        cfg_value=req.cfg_value,
        timesteps=req.timesteps,
        lora=req.lora,
        output=req.output.model_dump(),
        lane=req.lane,
        client=req.client,
        **({"job_id": req.job_id} if req.job_id else {}),
    )

    await engine._execute(job)
    if job.status == "failed":
        # 503 for an OOM: the request was never wrong, the card was full. The Go
        # gateway keys on `error_kind` to abandon the whole request instead of
        # handing the same full card this take's remaining chunks.
        code = 503 if job.error_kind == "oom" else 500
        return JSONResponse(job.as_dict(), status_code=code)
    if job.payload is not None:
        return _payload_response(job)
    return JSONResponse(job.as_dict(), status_code=200)


@app.post("/v2/jobs/render")
async def render(req: RenderRequest, wait: float = 0.0) -> Response:
    """Queue a render. With `?wait=N` the request blocks up to N seconds and returns
    the finished audio on the same connection; on timeout it returns 202 and the job
    id, and the caller polls."""
    engine = _engine()
    chunks = [c for c in req.chunks if c and c.strip()]
    if not chunks:
        return JSONResponse({"error": "chunks is empty"}, status_code=400)
    if req.output.mode not in ("npz", "files"):
        return JSONResponse({"error": f"unknown output mode {req.output.mode!r}"}, status_code=400)
    if req.lane not in LANES:
        return JSONResponse({"error": f"unknown lane {req.lane!r}"}, status_code=400)

    job = RenderJob(
        chunks=chunks,
        voice=req.voice.model_dump() if req.voice else None,
        cfg_value=req.cfg_value,
        timesteps=req.timesteps,
        lora=req.lora,
        output=req.output.model_dump(),
        lane=req.lane,
        client=req.client,
        **({"job_id": req.job_id} if req.job_id else {}),
    )
    engine.submit(job)

    if wait and wait > 0:
        finished = await engine.wait(job.job_id, min(wait, MAX_WAIT_S))
        if finished is not None:
            if finished.status == "failed":
                return JSONResponse(finished.as_dict(), status_code=500)
            if finished.payload is not None:
                return _payload_response(finished)
            return JSONResponse(finished.as_dict(), status_code=200)

    return JSONResponse(job.as_dict(engine.positions().get(job.job_id)), status_code=202)


@app.get("/v2/jobs")
def list_jobs(status: Optional[str] = None, limit: int = 100) -> JSONResponse:
    engine = _engine()
    jobs = list(engine.jobs.values())
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.status] = counts.get(j.status, 0) + 1
    pos = engine.positions()
    jobs.sort(key=lambda j: j.created, reverse=True)
    if status:
        jobs = [j for j in jobs if j.status == status]
    return JSONResponse({
        "counts": counts,
        "running": engine.running,
        "waiting": {lane: engine.queues[lane].qsize() for lane in LANES},
        "total": len(jobs),
        "jobs": [j.as_dict(pos.get(j.job_id)) for j in jobs[:limit]],
    })


@app.get("/v2/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    engine = _engine()
    job = engine.jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return JSONResponse(job.as_dict(engine.positions().get(job_id)))


@app.get("/v2/jobs/{job_id}/result")
def get_result(job_id: str) -> Response:
    job = _engine().jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    if job.status != "completed":
        return JSONResponse(
            {"error": f"job is {job.status}", "status": job.status, "detail": job.error},
            status_code=409,
        )
    if job.payload is None:
        if (job.result or {}).get("mode") == "files":
            return JSONResponse(job.result)
        return JSONResponse({"error": "result already delivered"}, status_code=410)
    return _payload_response(job)


@app.delete("/v2/jobs/{job_id}")
def cancel_job(job_id: str) -> JSONResponse:
    engine = _engine()
    if job_id not in engine.jobs:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    ok = engine.cancel(job_id)
    return JSONResponse({"job_id": job_id, "cancelled": ok, "status": engine.jobs[job_id].status})


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

class ResolveRequest(BaseModel):
    speaker_id: str
    ref_text: str = ""
    allow_sidecar: bool = True


@app.post("/v2/voices/resolve")
async def resolve_voice(req: ResolveRequest) -> JSONResponse:
    """Handle for a named reference clip, encoding it on first use."""
    engine = _engine()
    voices: VoiceStore = engine.voices
    try:
        # Encoding a reference clip is GPU work, and this endpoint is proxied
        # straight through by the queue gateway -- so without the lock it runs on
        # top of whatever generation is already in flight.
        async with engine.gpu_lock:
            handle = await asyncio.to_thread(
                voices.resolve_speaker, req.speaker_id, req.ref_text, req.allow_sidecar
            )
    except UnknownVoice as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return JSONResponse({"voice_handle": handle, "speaker_id": req.speaker_id})


@app.post("/v2/voices")
async def register_voice(
    clip: Annotated[UploadFile, File(description="Reference clip (3–10 s)")],
    speaker_id: Annotated[Optional[str], Form()] = None,
    ref_text: Annotated[Optional[str], Form()] = None,
    save_as_speaker: Annotated[bool, Form(description="Also keep the clip in ref/")] = False,
) -> JSONResponse:
    """Encode an uploaded clip and return a handle.

    Without `speaker_id` the handle is throwaway and expires — that is the studio's
    "synthesize with this file" path. With `save_as_speaker` the clip is filed in the
    reference directory and becomes a named voice both pipelines can use.
    """
    engine = _engine()
    voices: VoiceStore = engine.voices
    data = await clip.read()
    suffix = Path(clip.filename or "ref.wav").suffix or ".wav"

    kept: Optional[Path] = None
    if save_as_speaker and speaker_id:
        REF_DIR.mkdir(parents=True, exist_ok=True)
        kept = REF_DIR / f"{speaker_id}{suffix}"
        kept.write_bytes(data)
        src = kept
        tmp = None
    else:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(data)
            tmp = tf.name
        src = Path(tmp)

    try:
        async with engine.gpu_lock:                # encoding is GPU work
            handle = await asyncio.to_thread(
                voices.register_clip, src, speaker_id, ref_text, bool(speaker_id)
            )
    finally:
        if not (save_as_speaker and speaker_id) and tmp and os.path.exists(tmp):
            os.remove(tmp)

    return JSONResponse({
        "voice_handle": handle,
        "speaker_id": speaker_id,
        "saved_to": str(kept) if kept else None,
    })


@app.post("/v2/voices/seed")
async def seed_voice() -> JSONResponse:
    engine = _engine()
    # Minting the seed voice is a full generation, not just an encode -- it was
    # the one GPU workload that could run with no lane, no queue and no lock.
    async with engine.gpu_lock:
        handle = await asyncio.to_thread(
            engine.voices.seed,
            lambda text: engine.synth.synth(text, cfg_value=2.0, inference_timesteps=10),
        )
    if handle is None:
        return JSONResponse({"error": "seed voice unavailable"}, status_code=503)
    return JSONResponse({"voice_handle": handle})


@app.delete("/v2/voices/seed")
def reset_seed() -> JSONResponse:
    return JSONResponse({"rerolled": True, "cache_removed": _engine().voices.reset_seed()})


@app.get("/v2/voices")
def list_voices() -> JSONResponse:
    voices: VoiceStore = _engine().voices
    return JSONResponse({"voices": voices.list_voices(), **voices.stats()})


@app.get("/v2/voices/{speaker_id}/audio")
def get_voice_audio(speaker_id: str):
    voices: VoiceStore = _engine().voices
    try:
        ref_path = voices.ref_file(speaker_id)
        suffix = ref_path.suffix.lower()
        media_type = "audio/mpeg" if suffix == ".mp3" else ("audio/ogg" if suffix == ".ogg" else "audio/wav")
        return FileResponse(
            ref_path,
            media_type=media_type,
            headers={"Content-Disposition": f'inline; filename="{ref_path.name}"'}
        )
    except UnknownVoice as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@app.delete("/v2/voices/{speaker_id}")
def delete_voice(speaker_id: str) -> JSONResponse:
    voices: VoiceStore = _engine().voices
    for key in [k for k in list(voices.mem) if k == speaker_id or k.startswith(f"{speaker_id}-")]:
        voices.mem.pop(key, None)
        voices.meta.pop(key, None)
    removed = 0
    for p in voices.cache_dir.glob(f"{speaker_id}-*.pt"):
        p.unlink(missing_ok=True)
        removed += 1
    for d in _ref_dirs():
        for f in d.glob(f"{speaker_id}.*"):
            f.unlink(missing_ok=True)
    return JSONResponse({"speaker_id": speaker_id, "deleted": True, "caches_removed": removed})


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> JSONResponse:
    engine = _state.get("engine")
    if engine is None:
        return JSONResponse({"status": "loading", "stub": STUB}, status_code=503)
    jobs = list(engine.jobs.values())
    holder = engine.holder
    return JSONResponse({
        "status": "ok",
        "stub": engine.is_stub,
        # Residency, not health: a released model is a service working exactly as
        # configured, so this stays 200 and the state is a field. Reading it must
        # never load anything -- see ModelHolder.
        "model_loaded": holder.loaded if holder else True,
        "idle_mode": holder.mode if holder else "hot",
        "idle_ttl_s": holder.ttl if holder else None,
        "idle_s": round(holder.idle_seconds, 1) if holder else None,
        "model_loads": holder.loads if holder else None,
        "model_releases": holder.releases if holder else None,
        "load_seconds_total": round(holder.load_seconds, 1) if holder else None,
        "model": BASE_MODEL,
        "adapter": ADAPTER,
        "device": DEVICE or "auto",
        "sample_rate": engine.sample_rate,
        "default_lora": DEFAULT_LORA,
        "lora_now": engine._lora_state,
        "interactive_burst": INTERACTIVE_BURST,
        "waiting": {lane: engine.queues[lane].qsize() for lane in LANES},
        "running": engine.running,
        "completed": sum(1 for j in jobs if j.status == "completed"),
        "failed": sum(1 for j in jobs if j.status == "failed"),
        "oom": sum(1 for j in jobs if j.error_kind == "oom"),
        "total": len(jobs),
        "voices": engine.voices.stats(),
        "work_dir": str(engine.work_root.resolve()),
        "gpu_busy": engine.gpu_lock.locked(),
        "vram": _vram_stats(),
        "mem_fraction": MEM_FRACTION or None,
    })


@app.post("/v2/gpu/reclaim")
def gpu_reclaim() -> JSONResponse:
    """Drop this process's cached-but-unused VRAM so the SeedVC worker can have it.

    Manual, because the automatic policy only fires when the card is already
    tight; an operator about to start a big SeedVC run should not have to wait for
    that threshold to be crossed the hard way.
    """
    before = _vram_stats()
    freed = reclaim_vram(force=True)
    return JSONResponse({"reclaimed": freed, "before": before, "after": _vram_stats()})


@app.post("/v2/gpu/release")
async def gpu_release() -> JSONResponse:
    """Drop the weights now rather than waiting out the idle timer.

    For the operator about to start another GPU application. The timer exists so
    that nobody has to think about this in the normal case; when the card is
    wanted *now*, "wait three minutes for a threshold to expire" is the wrong
    answer. Waits for a running generation to finish rather than interrupting it.
    """
    engine = _engine()
    before = _vram_stats()
    released = await engine.release_model("requested")
    return JSONResponse({
        "released": released,
        "model_loaded": engine.holder.loaded if engine.holder else True,
        "before": before,
        "after": _vram_stats(),
    })


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>SiangTTS GPU Service</title>
<style>
 body{background:#090d16;color:#f1f5f9;font:14px/1.5 "Segoe UI",system-ui,sans-serif;margin:0;padding:24px}
 h1{font-size:18px;margin:0 0 4px} .sub{color:#64748b;margin-bottom:20px}
 .cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}
 .card{background:#111726;border:1px solid #263352;border-radius:10px;padding:12px 16px;min-width:120px}
 .card b{display:block;font-size:22px;font-weight:600} .card span{color:#94a3b8;font-size:12px}
 table{width:100%;border-collapse:collapse;background:#111726;border:1px solid #263352;border-radius:10px;overflow:hidden}
 th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #1c2740;font-size:13px}
 th{color:#94a3b8;font-weight:500;text-transform:uppercase;font-size:11px;letter-spacing:.04em}
 code{font-family:ui-monospace,Consolas,monospace;color:#94a3b8}
 .s{padding:2px 8px;border-radius:99px;font-size:11px}
 .queued{background:rgba(251,191,36,.12);color:#fbbf24}
 .running{background:rgba(56,189,248,.12);color:#38bdf8}
 .completed{background:rgba(52,211,153,.12);color:#34d399}
 .failed{background:rgba(248,113,113,.12);color:#f87171}
 .cancelled{background:rgba(100,116,139,.15);color:#64748b}
 .lane-interactive{color:#38bdf8}.lane-batch{color:#94a3b8}
 .stub{background:#f87171;color:#111;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
 .card.model b{font-size:15px} .held{color:#38bdf8} .free{color:#34d399}
</style></head><body>
<h1>SiangTTS GPU Service <span id="stub"></span></h1>
<div class="sub">Shared VoxCPM2 engine · webhook (:8010) and tone studio (:8011) are clients</div>
<div class="cards" id="cards"></div>
<table><thead><tr><th>job</th><th>client</th><th>lane</th><th>status</th><th>progress</th>
<th>pos</th><th>voice</th><th>lora</th><th>wait</th><th>run</th></tr></thead>
<tbody id="rows"></tbody></table>
<script>
async function tick(){
 const [h,j] = await Promise.all([fetch('/health').then(r=>r.json()),fetch('/v2/jobs?limit=40').then(r=>r.json())]);
 document.getElementById('stub').innerHTML = h.stub ? '<span class="stub">STUB — no model</span>' : '';
 document.getElementById('cards').innerHTML = [
  ['waiting (interactive)', h.waiting.interactive], ['waiting (batch)', h.waiting.batch],
  ['running', h.running ? 1 : 0], ['completed', h.completed], ['failed', h.failed],
  ['voices cached', h.voices.in_memory],
 ].map(([k,v])=>`<div class="card"><b>${v}</b><span>${k}</span></div>`).join('')
 + `<div class="card model"><b class="${h.model_loaded?'held':'free'}">${
     h.model_loaded ? 'holding VRAM' : 'card released'}</b><span>model · ${
     h.idle_mode}${h.model_loaded&&h.idle_mode!=='hot'?` · idle ${Math.round(h.idle_s)}s/${Math.round(h.idle_ttl_s)}s`:''}</span></div>`;
 document.getElementById('rows').innerHTML = (j.jobs||[]).map(r=>`<tr>
  <td><code>${r.job_id}</code></td><td>${r.client||'—'}</td>
  <td class="lane-${r.lane}">${r.lane}</td>
  <td><span class="s ${r.status}">${r.status}</span></td>
  <td>${r.progress}</td><td>${r.position??'—'}</td>
  <td><code>${r.voice_handle||'—'}</code></td>
  <td><code>${r.lora?`${r.lora.lm}/${r.lora.dit}`:'—'}</code></td>
  <td>${r.waited_s}s</td><td>${r.elapsed_s??'—'}${r.elapsed_s!=null?'s':''}</td></tr>`).join('');
}
tick(); setInterval(tick, 1500);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_PAGE)
