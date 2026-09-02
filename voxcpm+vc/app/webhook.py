"""n8n LiveAI webhook — the :8010 async contract, hosted inside the :8013 studio.

The production webhook (voice-cloning/src/webhook.py, :8010) accepts a script, answers
``{"status":"success"}`` immediately, and delivers the finished audio URL on a
``callback_url`` later. It synthesizes through the plain Thai-LoRA path on the shared
GPU service.

This module reproduces that contract exactly, but the synthesis runs through *this*
studio's pipeline instead: each chunk's emotion is auto-annotated, cloned from a donor
recording by VoxCPM2, and the timbre swapped onto the target voice by SeedVC. So a
caller pointed here gets emotional voice cloning over the same webhook it already uses,
with no second process and no second port — the endpoints mount under /webhook/* on
8013.

    POST /webhook/live-ai-create-new   accept a script, enqueue, return success
    POST /live-ai-create-new           bare alias (callers pointed at the n8n node path)
    GET  /webhook                      monitoring dashboard
    GET  /webhook/jobs[ /{job_id} ]    job state as JSON
    GET  /webhook/voices               target voices available
    GET  /webhook/audio/{queue_id}     locally-rendered take, for dashboard preview
    GET  /webhook/health               this queue + the studio behind it

One GPU means one job at a time, so a single asyncio worker drains the queue in FIFO
order — the same shape as :8010, which also gives callers backpressure for free.
"""

from __future__ import annotations

import asyncio
import random
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.annotator import annotator
from app.models import Segment, Tone
from app.segmenter import segment_text
from app.renderers import get_renderer
from app.renderers.voxcpm import split_style_chunk_specs
from app.services.siangtts_service import siangtts_service
from app.services.voxcpm_vc_service import voxcpm_vc_service

router = APIRouter()

WORK_ROOT = Path(settings.webhook_work_dir)

# The Go queue gateway (:8020). Every accepted webhook request is mirrored there as a
# visibility-only "meta" job the instant it arrives, so the admin dashboard shows the
# whole backlog rather than only the one job this studio happens to be working. The
# real per-emotion generations still flow through the same gateway, tagged with the
# *-internal client below so the dashboard collapses them behind their meta row.
QUEUE_BASE_URL = settings.voxcpm_service_url.rstrip("/")
META_LANE = "batch"
META_CLIENT = "voxcpm-vc"
INTERNAL_CLIENT = "voxcpm-vc-internal"


async def _meta_register(job: "Job") -> None:
    """Register this request's row on the queue gateway at accept time.

    Best-effort: a gateway that is down or slow must never block accepting the job or
    running the pipeline — the dashboard is a view, not the source of truth.
    """
    payload = {
        "job_id": job.queue_id,
        "raw_prompt": job.prompt,
        "lane": META_LANE,
        "client": META_CLIENT,
        # The body as it arrived, alongside what this studio read it as. voice_id,
        # sex and donor_set are all silently defaulted when the caller omits them
        # (see _accept and _pick_donor_set), and a take in the house voice looks
        # exactly like a take in the requested one -- so the dashboard gets both
        # halves and can say which fields the caller actually sent.
        "request": {"received": job.request, "resolved": job.resolved},
        # Put the resolved target on the row itself too: the per-emotion render
        # jobs behind it are conditioned on *donor* clips, so the card would
        # otherwise label this request with a donor handle.
        "voice": {"speaker_id": job.voice_id} if job.voice_id else None,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{QUEUE_BASE_URL}/v2/jobs/external", json=payload)
        job.meta_registered = resp.status_code < 400
    except Exception as exc:                                          # noqa: BLE001
        print(f"[{job.queue_id}] meta register failed (dashboard only): {exc}")


# Substrings that identify a CUDA out-of-memory failure in an error string. The
# GPU service and the SeedVC worker both label their own OOMs, but a failure can
# also arrive here as the text of a 5xx body, which is what this covers.
_OOM_MARKERS = ("out of memory", "outofmemoryerror", "alloc_failed", '"oom"', "'oom'")


def _looks_like_oom(message: str) -> bool:
    low = (message or "").lower()
    return any(m in low for m in _OOM_MARKERS)


async def _meta_patch(job: "Job", **fields: object) -> None:
    """Advance this request's dashboard row (status/chunks/result). No-op when the row
    was never registered; best-effort so a dashboard hiccup can't fail synthesis."""
    if not job.meta_registered:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.patch(f"{QUEUE_BASE_URL}/v2/jobs/{job.queue_id}", json=fields)
    except Exception as exc:                                          # noqa: BLE001
        print(f"[{job.queue_id}] meta patch failed (dashboard only): {exc}")


# ---------------------------------------------------------------------------
# Request / job model  (same wire shape the :8010 webhook receives from n8n)
# ---------------------------------------------------------------------------

class WebhookBody(BaseModel):
    """Body the n8n LiveAI flow posts. Unknown fields are accepted and ignored so
    the same caller works against either service."""

    prompt: str = ""
    job_id: str = ""
    queue_id: str = ""
    voice_id: str = ""
    voice_text: str = ""
    ref_text: str = ""
    audio_speed: float = 1.0
    country_code: str = "th"
    callback_url: str = ""
    # Extensions beyond the :8010 body (both optional, so the original n8n payload
    # still validates). `sex` picks which donor gender clones the emotion; `donor_set`
    # pins one specific actor. Omit donor_set and a random set of the chosen sex is
    # used per job, so takes vary instead of always cloning the same actor.
    sex: str = ""
    donor_set: str = ""


@dataclass
class Job:
    job_id: str
    queue_id: str
    voice_id: str
    callback_url: str
    # Planned in the worker (see _run_job), not at accept time, so the HTTP response
    # comes back immediately like the :8010 webhook instead of blocking on the LLM.
    parts: List[str] = field(default_factory=list)
    tones: List[Optional[str]] = field(default_factory=list)
    breaks: List[bool] = field(default_factory=list)
    prompt: str = ""
    status: str = "queued"              # queued | running | completed | failed
    error: Optional[str] = None
    file_url: Optional[str] = None
    donor_set: Optional[str] = None     # the actor whose emotion is cloned
    gender: Optional[str] = None        # sex used to pick/validate the donor
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    # True once this job's visibility-only row is registered on the queue gateway, so
    # progress PATCHes know there is something to update (see _meta_register).
    meta_registered: bool = False
    # The caller's JSON body verbatim, and what this studio resolved it into. Kept
    # for the dashboards (this one and :8020), which is the only place a silent
    # default -- an omitted voice_id, an unreadable sex -- is visible at all.
    request: dict = field(default_factory=dict)
    resolved: dict = field(default_factory=dict)

    def as_dict(self, position: Optional[int] = None) -> dict:
        now = time.time()
        waited = (self.started or now) - self.created
        ran = ((self.finished or now) - self.started) if self.started else None
        local = WORK_ROOT / self.queue_id / f"{self.queue_id}.wav"
        has_audio = local.exists()
        audio_src = self.file_url or (f"/webhook/audio/{self.queue_id}" if has_audio else None)
        return {
            "job_id": self.job_id,
            "queue_id": self.queue_id,
            "status": self.status,
            "voice_id": self.voice_id or "auto",
            "prompt": self.prompt,
            "callback_url": self.callback_url,
            "chunks_total": len(self.parts),
            "tones": self.tones,
            "donor_set": self.donor_set,
            "sex": self.gender,
            "position": position,
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created)),
            "created_ts": self.created,
            "waited_s": round(waited, 1),
            "elapsed_s": round(ran, 1) if ran is not None else None,
            "file_url": self.file_url,
            "audio_src": audio_src,
            "has_local_audio": has_audio,
            "error": self.error,
            "request": self.request,
            "resolved": self.resolved,
        }


# ---------------------------------------------------------------------------
# In-process queue state
# ---------------------------------------------------------------------------

_state: dict = {}


async def start_worker() -> None:
    """Called from the app lifespan once the studio is up."""
    _state["jobs"] = {}
    _state["running"] = None
    _state["queue"] = asyncio.Queue()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    _state["worker"] = asyncio.create_task(_worker())
    print(f"[webhook] ready — queue on /webhook, work={WORK_ROOT.resolve()}")


async def stop_worker() -> None:
    task = _state.get("worker")
    if task is not None:
        task.cancel()
    _state.clear()


async def _worker() -> None:
    queue: asyncio.Queue = _state["queue"]
    while True:
        job: Job = await queue.get()
        try:
            await _run_job(job)
        except Exception:                                            # noqa: BLE001
            traceback.print_exc()
        finally:
            queue.task_done()


async def _run_job(job: Job) -> None:
    work = WORK_ROOT / job.queue_id
    job.status = "running"
    job.started = time.time()
    _state["running"] = job.job_id
    await _meta_patch(job, status="running")

    try:
        work.mkdir(parents=True, exist_ok=True)

        # Plan the chunks here, not at accept time: emotion annotation may call the LLM
        # (when enabled), and doing it in the worker keeps the accept response instant,
        # exactly like the :8010 webhook. A script that normalizes to nothing fails the
        # job (reported via callback) instead of a synchronous HTTP 400.
        job.parts, job.tones, job.breaks = await asyncio.to_thread(
            _plan_chunks, job.prompt, settings.webhook_use_llm
        )
        if not any(p.strip() for p in job.parts):
            raise RuntimeError("prompt is empty after normalize")

        # Publish the planned chunks onto the dashboard row now that they exist.
        await _meta_patch(job, chunks=list(job.parts), total_chunks=len(job.parts))

        # Generation is blocking (HTTP to the GPU service + SeedVC), so keep it off
        # the event loop or the accept endpoint stalls behind it.
        debug: List[dict] = []
        wav_bytes = await asyncio.to_thread(
            voxcpm_vc_service.synthesize_many,
            job.parts,
            speaker_id=(job.voice_id or None),
            tones=job.tones,
            breaks=job.breaks,
            donor_set=job.donor_set,
            gender=job.gender,
            raw_prompt=job.prompt,
            client=INTERNAL_CLIENT,
            # Attach every generation job to this request's existing dashboard
            # row, so the operator sees one card with its chunks under it instead
            # of one loose row per emotion -- and so an OOM on one of them takes
            # the rest of this take down with it rather than grinding on.
            request_id=job.queue_id,
            debug_out=debug,
        )
        # Keep the pinned/random pick; if none was resolvable up front, record what
        # the synth actually fell back to.
        job.donor_set = job.donor_set or (debug[0]["donor_set"] if debug else None)

        out = work / f"{job.queue_id}.wav"
        out.write_bytes(wav_bytes)
        size_kb = round(out.stat().st_size / 1024)
        print(f"[{job.queue_id}] rendered -> {out.resolve()} ({size_kb} KB)")

        job.file_url = await _upload(out)
        job.status = "completed"
        # Mark the dashboard row done only now — after generate + SeedVC + assemble +
        # upload — so "completed" there means the whole take is delivered, not just that
        # the GPU generation finished.
        # Only the uploaded URL is worth publishing: `work` is wiped a few lines
        # below (see the finally block), so a local path here would be a dead link
        # by the time anyone clicks Play on the dashboard.
        await _meta_patch(job, status="completed", result={"file_url": job.file_url})
        await _post_callback(job.callback_url, job, error=None)
        print(f"[{job.queue_id}] done -> {job.file_url}")

    except Exception as exc:                                          # noqa: BLE001
        job.status = "failed"
        job.error = str(exc)
        traceback.print_exc()
        # Name a VRAM failure on the dashboard. SeedVC conversion happens in this
        # process, not on the queue gateway, so an OOM there would otherwise reach
        # the operator as an unlabelled traceback.
        kind = "oom" if _looks_like_oom(job.error) else None
        await _meta_patch(job, status="failed", error=job.error, **({"error_kind": kind} if kind else {}))
        try:
            await _post_callback(job.callback_url, job, error=job.error)
        except Exception as cb_exc:                                  # noqa: BLE001
            print(f"[{job.queue_id}] error callback failed: {cb_exc}")

    finally:
        job.finished = time.time()
        _state["running"] = None
        # Only a delivered job is safe to wipe — a failed one usually broke at upload
        # or callback, after the (expensive) synthesis, so its audio is worth keeping.
        if job.status == "completed" and not settings.webhook_keep_work:
            _cleanup(work)
        elif job.status != "completed" and work.exists():
            print(f"[{job.queue_id}] failed - audio kept at {work.resolve()}")


# ---------------------------------------------------------------------------
# Delivery — upload the take, POST the result back  (mirrors :8010 pipeline.py)
# ---------------------------------------------------------------------------

async def _upload(path: Path) -> str:
    """POST the merged audio as multipart `file`, return the `file_url` field."""
    token = settings.siangtts_upload_token.strip()
    if not token:
        raise RuntimeError("SIANGTTS_UPLOAD_TOKEN is not set")
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        with open(path, "rb") as fh:
            resp = await client.post(
                settings.siangtts_upload_url.strip(),
                headers=headers,
                files={"file": (path.name, fh, "audio/wav")},
            )
    if resp.status_code >= 400:
        raise RuntimeError(f"upload failed {resp.status_code} - {resp.text[:300]}")
    data = resp.json()
    file_url = data.get("file_url") or data.get("url")
    if not file_url:
        raise RuntimeError(f"upload response has no file_url - {resp.text[:300]}")
    return file_url


async def _post_callback(url: str, job: Job, *, error: Optional[str]) -> None:
    payload = {
        "job_id": job.job_id,
        "queue_id": job.queue_id,
        "file_url": job.file_url if job.file_url else "none",
        "error": error,
    }
    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        await client.post(url, json=payload)


def _cleanup(work_dir: Path) -> None:
    import shutil

    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception as exc:                                         # noqa: BLE001
        print(f"[cleanup] ignored: {exc}")


# ---------------------------------------------------------------------------
# Chunk planning — self-contained copy of main._plan_chunks (kept here to avoid
# importing app.main, which would be a circular import)
# ---------------------------------------------------------------------------

# Values of `sex` this webhook recognises. Anything else is *not* rejected -- the
# n8n flow has been sending free text for a long time -- but it does not silently
# pass for a deliberate choice either: it resolves to the configured default and is
# reported as unrecognised, because "female" and "ชาย" would otherwise both produce
# a female donor and look identical afterwards.
_MALE_WORDS = {"m", "male", "man", "ชาย", "ผู้ชาย"}
_FEMALE_WORDS = {"f", "female", "woman", "หญิง", "ผู้หญิง"}


def _normalize_sex(sex: str) -> tuple[str, str]:
    """`sex` from the body -> ("male"|"female", where that answer came from)."""
    raw = (sex or "").strip()
    if not raw:
        default = (settings.default_gender or "female").strip().lower()
        want = "male" if default in _MALE_WORDS else "female"
        return want, f"not sent — DEFAULT_GENDER={default or 'female'}"

    low = raw.lower()
    if low in _MALE_WORDS:
        return "male", "request"
    if low in _FEMALE_WORDS:
        return "female", "request"
    # Historical rule: anything starting with "m" was male and everything else was
    # female, which quietly turned a typo into a female donor. Same outcome, but
    # named as the guess it is.
    want = "male" if low.startswith("m") else "female"
    return want, f"unrecognised {raw!r} — guessed {want}"


def _pick_donor_set(sex: str, donor_set: str):
    """Choose the donor whose emotion is cloned.

    Returns ``(donor_set, gender, sources)``, where ``sources`` says where the sex
    and the donor came from so the dashboards can show a default standing in for
    something the caller never sent.

    An explicit ``donor_set`` is pinned as-is (the synth validates it). Otherwise a
    random complete set of the requested sex is drawn, so successive jobs vary the
    actor instead of always cloning the top-ranked one. ``sex`` defaults to the
    configured gender when blank. When no set matches, donor_set comes back None and
    the gender is handed on so the synth can resolve or raise a clear error.
    """
    want, sex_source = _normalize_sex(sex)

    ds = donor_set.strip()
    if ds:
        # A pinned actor already carries its own sex; the `sex` field is not read.
        return ds, None, {"sex_source": "not used — donor_set pinned", "donor_set_source": "request"}

    sets = voxcpm_vc_service.list_donor_sets()
    pool = [s["id"] for s in sets if s.get("gender") == want and s.get("complete")]
    if not pool:
        pool = [s["id"] for s in sets if s.get("gender") == want]
    if pool:
        return random.choice(pool), want, {
            "sex_source": sex_source,
            "donor_set_source": f"random {want} donor ({len(pool)} available)",
        }
    return None, want, {
        "sex_source": sex_source,
        "donor_set_source": f"no {want} donor set — resolved at render time",
    }


def _plan_chunks(text: str, use_llm: bool = True):
    """Split into per-emotion chunks. Hand-written style tags always win. Un-tagged text
    is auto-annotated by the LLM so each chunk carries the tone that selects its donor --
    unless ``use_llm`` is off (WEBHOOK_USE_LLM=false), in which case the whole script is
    rendered neutral and no LLM is contacted."""
    specs = split_style_chunk_specs(text, use_llm=use_llm)
    if specs:
        return (
            [s.text for s in specs],
            [s.tone for s in specs],
            [s.break_before for s in specs],
        )

    renderer = get_renderer("voxcpm")
    if use_llm:
        clauses = segment_text(text)
        segments = annotator.annotate(original_text=text, clauses=clauses).segments
    else:
        # No LLM: one neutral segment. synthesize_many still splits it into safe
        # generation-sized pieces internally, so long scripts are fine here.
        segments = [Segment(text=text, tone=Tone.NEUTRAL, intensity=2)]

    rendered = renderer.render(segments)
    if rendered.chunks:
        return (
            [c.text for c in rendered.chunks],
            [c.tone for c in rendered.chunks],
            [c.break_before for c in rendered.chunks],
        )
    return ([rendered.text], [None], [False])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

async def _accept(body: WebhookBody, received: Optional[dict] = None) -> JSONResponse:
    """Enqueue and return success immediately, like the :8010 webhook — the script is
    planned and synthesized in the worker, and any problem is reported via callback.
    Only a trivially-empty prompt is rejected inline (no LLM, no GPU touched here)."""
    queue_id = body.queue_id or str(uuid.uuid4())
    callback_url = body.callback_url or settings.siangtts_default_callback

    if not body.prompt.strip():
        return JSONResponse(
            {"status": "error", "error": "prompt is empty"},
            status_code=400,
        )

    # A missing voice_id is a broken request, not a request for "any voice". SeedVC
    # converts *into* a named target, so with nothing pinned this used to fall
    # through to whatever clip sorted first in ref/ -- delivering a take in a
    # stranger's voice that is indistinguishable from a correct one until someone
    # listens. Refuse it here, where the caller still gets an answer it can act on;
    # WEBHOOK_DEFAULT_VOICE nominates a house voice for callers that really do not
    # care. Rejected inline (like an empty prompt) because it costs no LLM and no GPU.
    asked_voice = body.voice_id.strip()
    default_voice = settings.webhook_default_voice.strip()
    if not asked_voice and not default_voice:
        return JSONResponse(
            {
                "status": "error",
                "error": (
                    "voice_id is required: this pipeline converts every take into a "
                    "named target voice. Send voice_id, or set WEBHOOK_DEFAULT_VOICE "
                    "on the service to nominate a default."
                ),
            },
            status_code=400,
        )

    # Pick the donor now (fast, local — no network) so the choice is fixed when the job
    # is queued and visible on the dashboard, and a random pick is stable across retries.
    donor_set, gender, sources = await asyncio.to_thread(
        _pick_donor_set, body.sex, body.donor_set
    )

    # Every one of these three can be a default the caller never asked for, and the
    # resulting take sounds like a normal one either way. Record which is which here,
    # where the body is still in hand, and ship it to both dashboards.
    voice_id = asked_voice or default_voice
    voice_source = "request" if asked_voice else "not sent — WEBHOOK_DEFAULT_VOICE"

    job = Job(
        job_id=body.job_id or queue_id,
        queue_id=queue_id,
        voice_id=voice_id,
        callback_url=callback_url,
        prompt=body.prompt,
        donor_set=donor_set,
        gender=gender,
        request=received if received is not None else body.model_dump(),
        resolved={
            "voice_id": voice_id or None,
            "voice_id_source": voice_source,
            "sex": gender,
            "sex_source": sources["sex_source"],
            "donor_set": donor_set,
            "donor_set_source": sources["donor_set_source"],
        },
    )

    jobs: dict = _state["jobs"]
    jobs[job.job_id] = job
    # Bounded history: dict is insertion-ordered, so drop the oldest finished jobs
    # first and never evict one still queued or running.
    if len(jobs) > settings.webhook_max_history:
        for jid, j in list(jobs.items()):
            if len(jobs) <= settings.webhook_max_history:
                break
            if j.status in ("completed", "failed"):
                del jobs[jid]

    await _state["queue"].put(job)
    # Mirror the request onto the queue dashboard immediately, so the admin sees the
    # full backlog now rather than only when this studio's serial worker reaches it.
    await _meta_register(job)
    print(
        f"[{queue_id}] queued — voice={job.voice_id or 'auto'} ({voice_source}), "
        f"donor={donor_set or 'auto'} ({sources['donor_set_source']}), "
        f"sex={gender or '-'} ({sources['sex_source']})"
    )
    return JSONResponse({"status": "success", "job_id": job.job_id})


async def _received_body(request: Request) -> Optional[dict]:
    """The caller's JSON exactly as posted, for the dashboards.

    Deliberately the *raw* body rather than the parsed model: a field n8n spelled
    wrong is dropped by the model and is precisely what an operator staring at a
    take in the wrong voice needs to see. Returns None if it is not a JSON object,
    in which case the parsed model stands in.
    """
    try:
        data = await request.json()
    except Exception:                                                 # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


@router.post("/webhook/live-ai-create-new")
async def webhook(body: WebhookBody, request: Request) -> JSONResponse:
    return await _accept(body, await _received_body(request))


@router.post("/live-ai-create-new")
async def webhook_bare(body: WebhookBody, request: Request) -> JSONResponse:
    """Bare alias for callers pointed straight at the old n8n node path."""
    return await _accept(body, await _received_body(request))


def _positions() -> dict:
    waiting = sorted(
        (j for j in _state.get("jobs", {}).values() if j.status == "queued"),
        key=lambda j: j.created,
    )
    return {j.job_id: i for i, j in enumerate(waiting, 1)}


@router.get("/webhook/jobs")
def list_jobs(status: Optional[str] = None, limit: int = 100) -> JSONResponse:
    jobs: List[Job] = list(_state.get("jobs", {}).values())
    counts: dict = {}
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


@router.get("/webhook/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job: Optional[Job] = _state.get("jobs", {}).get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return JSONResponse(job.as_dict(_positions().get(job_id)))


@router.get("/webhook/voices")
def list_voices() -> JSONResponse:
    speakers = siangtts_service.list_speakers()
    return JSONResponse({
        "voices": speakers,
        "default": settings.webhook_default_voice or "auto",
    })


@router.get("/webhook/health")
def webhook_health() -> JSONResponse:
    queue: Optional[asyncio.Queue] = _state.get("queue")
    jobs: List[Job] = list(_state.get("jobs", {}).values())
    return JSONResponse({
        "status": "ok",
        "pipeline": "donor -> VoxCPM2 (continuation) -> SeedVC",
        "waiting": queue.qsize() if queue else 0,
        "running": _state.get("running"),
        "completed": sum(1 for j in jobs if j.status == "completed"),
        "failed": sum(1 for j in jobs if j.status == "failed"),
        "total": len(jobs),
        "upload_token": bool(settings.siangtts_upload_token.strip()),
        "default_callback": settings.siangtts_default_callback,
    })


@router.get("/webhook/audio/{queue_id}")
def get_audio(queue_id: str):
    if "/" in queue_id or "\\" in queue_id or ".." in queue_id:
        return JSONResponse({"error": "invalid id"}, status_code=400)
    wav = WORK_ROOT / queue_id / f"{queue_id}.wav"
    if wav.exists():
        return FileResponse(wav, media_type="audio/wav", filename=f"{queue_id}.wav")
    return JSONResponse({"error": "audio not found in local scratch"}, status_code=404)


@router.get("/webhook", include_in_schema=False)
def dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# Monitoring dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoxCPM2+VC Webhook Queue</title>
<style>
  :root {
    color-scheme: light dark;
    --bg:#090d16; --surface:#111726; --elev:#1a2236; --hover:#222d47;
    --border:#263352; --border2:#1c2740; --text:#f1f5f9; --muted:#94a3b8; --dim:#64748b;
    --primary:#a78bfa; --run:#38bdf8; --run-bg:rgba(56,189,248,.12);
    --ok:#34d399; --ok-bg:rgba(52,211,153,.12); --fail:#f87171; --fail-bg:rgba(248,113,113,.12);
    --wait:#fbbf24; --wait-bg:rgba(251,191,36,.12);
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:-apple-system,"Segoe UI","Sarabun",Roboto,sans-serif;
         font-size:14px; line-height:1.5; padding:1.5rem 2rem 3rem; min-height:100vh; }
  header { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;
           padding-bottom:1.25rem; border-bottom:1px solid var(--border2); margin-bottom:1.5rem; }
  .brand { display:flex; align-items:center; gap:.75rem; }
  .logo { width:38px; height:38px; border-radius:10px; background:linear-gradient(135deg,#7c3aed,#a78bfa);
          display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800; font-size:1.1rem; }
  h1 { font-size:1.2rem; font-weight:700; display:flex; align-items:center; gap:.5rem; }
  .sub { font-size:.8rem; color:var(--muted); }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--ok); box-shadow:0 0 8px var(--ok);
         display:inline-block; animation:pulse 2s infinite; }
  .dot.off { background:var(--fail); box-shadow:0 0 8px var(--fail); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .pills { display:flex; gap:.5rem; flex-wrap:wrap; }
  .pill { background:var(--surface); border:1px solid var(--border); padding:.3rem .65rem; border-radius:9999px;
          font-size:.75rem; color:var(--muted); }
  .pill b { color:var(--text); }
  .pill.ok { border-color:var(--ok); color:var(--ok); }
  .pill.warn { border-color:var(--wait); color:var(--wait); }
  .actions { display:flex; gap:.6rem; align-items:center; }
  .btn { background:var(--elev); border:1px solid var(--border); color:var(--text); padding:.45rem .85rem;
         border-radius:8px; font-size:.825rem; cursor:pointer; font-family:inherit; }
  .btn:hover { background:var(--hover); border-color:var(--primary); }
  .btn-primary { background:var(--primary); color:#0b0714; border-color:var(--primary); font-weight:600; }
  select { background:var(--elev); border:1px solid var(--border); color:var(--text); padding:.45rem .7rem;
           border-radius:8px; font-size:.825rem; font-family:inherit; }
  .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.85rem; margin-bottom:1.5rem; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1rem 1.15rem;
          cursor:pointer; transition:.15s; }
  .card:hover { transform:translateY(-2px); border-color:var(--dim); }
  .card.active { border-color:var(--primary); box-shadow:0 0 0 1px var(--primary); }
  .card .lbl { font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
  .card .val { font-size:1.7rem; font-weight:700; font-variant-numeric:tabular-nums; margin-top:.25rem; }
  .card.running .val{color:var(--run)} .card.queued .val{color:var(--wait)}
  .card.completed .val{color:var(--ok)} .card.failed .val{color:var(--fail)}
  .toolbar { display:flex; justify-content:space-between; align-items:center; gap:.75rem; flex-wrap:wrap; margin-bottom:1rem; }
  .tabs { display:flex; gap:.35rem; background:var(--surface); padding:.25rem; border-radius:10px; border:1px solid var(--border); }
  .tab { padding:.35rem .75rem; border-radius:6px; font-size:.8rem; color:var(--muted); background:transparent;
         border:none; cursor:pointer; font-family:inherit; }
  .tab.active { background:var(--elev); color:var(--text); font-weight:600; }
  input[type=text] { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:.4rem .75rem;
                     color:var(--text); font-size:.825rem; font-family:inherit; min-width:220px; }
  .wrap { background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; }
  table { width:100%; border-collapse:collapse; font-size:.825rem; }
  th { background:var(--elev); color:var(--muted); font-weight:600; font-size:.72rem; text-transform:uppercase;
       letter-spacing:.05em; padding:.7rem 1rem; border-bottom:1px solid var(--border); text-align:left; white-space:nowrap; }
  td { padding:.7rem 1rem; border-bottom:1px solid var(--border2); vertical-align:middle; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:var(--hover); }
  .badge { display:inline-flex; align-items:center; padding:.2rem .55rem; border-radius:9999px; font-size:.72rem;
           font-weight:600; text-transform:uppercase; }
  .badge.running{background:var(--run-bg);color:var(--run);border:1px solid var(--run)}
  .badge.queued{background:var(--wait-bg);color:var(--wait);border:1px solid var(--wait)}
  .badge.completed{background:var(--ok-bg);color:var(--ok);border:1px solid var(--ok)}
  .badge.failed{background:var(--fail-bg);color:var(--fail);border:1px solid var(--fail)}
  .mono { font-family:ui-monospace,Consolas,monospace; font-size:.78rem; color:var(--muted); }
  .num { font-variant-numeric:tabular-nums; }
  .snippet { max-width:260px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .tag { display:inline-block; background:var(--elev); border:1px solid var(--border); border-radius:6px;
         padding:.15rem .45rem; font-size:.75rem; color:var(--muted); }
  .empty { padding:3rem; text-align:center; color:var(--muted); }
  .modal-bg { display:none; position:fixed; inset:0; background:rgba(0,0,0,.65); z-index:100;
              align-items:center; justify-content:center; padding:1.5rem; }
  .modal-bg.on { display:flex; }
  .modal { background:var(--surface); border:1px solid var(--border); border-radius:16px; max-width:560px; width:100%;
           display:flex; flex-direction:column; max-height:90vh; }
  .modal h2 { padding:1.2rem 1.5rem; border-bottom:1px solid var(--border); font-size:1.05rem; }
  .modal .body { padding:1.5rem; display:flex; flex-direction:column; gap:1rem; overflow:auto; }
  .modal .foot { padding:1rem 1.5rem; border-top:1px solid var(--border); display:flex; justify-content:flex-end; gap:.5rem; }
  label { font-size:.72rem; font-weight:600; text-transform:uppercase; color:var(--muted); display:block; margin-bottom:.35rem; }
  textarea, .fc { width:100%; background:var(--elev); border:1px solid var(--border); border-radius:8px; padding:.55rem .75rem;
                  color:var(--text); font-size:.85rem; font-family:inherit; }
  textarea { min-height:90px; resize:vertical; }
  .toast { position:fixed; bottom:2rem; right:2rem; background:var(--elev); border:1px solid var(--border);
           border-radius:10px; padding:.75rem 1.25rem; display:none; z-index:200; }
  .toast.on { display:block; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="logo">V</div>
    <div>
      <h1>VoxCPM2+VC Webhook <span class="dot" id="dot"></span></h1>
      <div class="sub">n8n LiveAI queue · emotion cloned from donor → SeedVC · :8013</div>
    </div>
  </div>
  <div class="pills">
    <div class="pill">Pipeline: <b>donor→VoxCPM2→SeedVC</b></div>
    <div class="pill" id="pill-upload">Upload: <b id="v-upload">—</b></div>
  </div>
  <div class="actions">
    <button class="btn btn-primary" onclick="openTest()">+ New Test Job</button>
    <select id="interval" onchange="setInterval2(this.value)">
      <option value="2000" selected>Refresh 2s</option>
      <option value="5000">Refresh 5s</option>
      <option value="0">Paused</option>
    </select>
    <button class="btn" onclick="tick()">Refresh</button>
  </div>
</header>

<div class="kpis">
  <div class="card running" onclick="setFilter('running')"><div class="lbl">Running</div><div class="val" id="k-running">0</div></div>
  <div class="card queued" onclick="setFilter('queued')"><div class="lbl">In Queue</div><div class="val" id="k-queued">0</div></div>
  <div class="card completed" onclick="setFilter('completed')"><div class="lbl">Completed</div><div class="val" id="k-completed">0</div></div>
  <div class="card failed" onclick="setFilter('failed')"><div class="lbl">Failed</div><div class="val" id="k-failed">0</div></div>
  <div class="card active" onclick="setFilter('all')"><div class="lbl">Total</div><div class="val" id="k-total">0</div></div>
</div>

<div class="toolbar">
  <div class="tabs">
    <button class="tab active" id="t-all" onclick="setFilter('all')">All</button>
    <button class="tab" id="t-running" onclick="setFilter('running')">Running</button>
    <button class="tab" id="t-queued" onclick="setFilter('queued')">Queued</button>
    <button class="tab" id="t-completed" onclick="setFilter('completed')">Completed</button>
    <button class="tab" id="t-failed" onclick="setFilter('failed')">Failed</button>
  </div>
  <input type="text" id="q" placeholder="Search job id, voice, prompt, error…" oninput="render()">
</div>

<div class="wrap">
  <table>
    <thead><tr>
      <th>#</th><th>Status</th><th>Job ID</th><th>Prompt</th><th>Voice</th><th>Emotion</th>
      <th>Chunks</th><th>Created</th><th>Waited</th><th>Elapsed</th><th>Audio</th>
    </tr></thead>
    <tbody id="rows"><tr><td colspan="11" class="empty">Loading…</td></tr></tbody>
  </table>
</div>

<div class="modal-bg" id="test-bg" onclick="if(event.target===this)closeTest()">
  <div class="modal">
    <h2>Create Test Job</h2>
    <div class="body">
      <div><label>Thai Prompt</label>
        <textarea id="m-prompt">โปรโมชั่นวันนี้ลดสูงสุดห้าสิบเปอร์เซ็นต์ รีบสั่งเลยนะคะ เดี๋ยวของหมด</textarea></div>
      <div><label>Voice ID (required unless WEBHOOK_DEFAULT_VOICE is set)</label><input class="fc" id="m-voice" placeholder="e.g. 02af962c-96f3-4d48-9195-fdb3715abfae"></div>
      <div><label>Donor sex (blank = default)</label>
        <select class="fc" id="m-sex"><option value="">default</option><option value="female">female</option><option value="male">male</option></select></div>
      <div><label>Donor set (blank = random of that sex)</label><input class="fc" id="m-donor" placeholder="random"></div>
      <div><label>Callback URL (blank = default)</label><input class="fc" id="m-cb" placeholder="default"></div>
    </div>
    <div class="foot">
      <button class="btn" onclick="closeTest()">Cancel</button>
      <button class="btn btn-primary" onclick="submitTest()">Submit</button>
    </div>
  </div>
</div>

<audio id="audio" style="display:none"></audio>
<div class="toast" id="toast"></div>

<script>
let data = { counts:{}, running:null, jobs:[] };
let filter = 'all', timer = null, ms = 2000, playing = null;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const secs = v => v==null ? '—' : v<60 ? v.toFixed(1)+'s' : Math.floor(v/60)+'m '+Math.round(v%60)+'s';
function toast(m){ const t=document.getElementById('toast'); t.textContent=m; t.classList.add('on'); setTimeout(()=>t.classList.remove('on'),2200); }

function setFilter(f){
  filter=f;
  ['all','running','queued','completed','failed'].forEach(s=>{
    const t=document.getElementById('t-'+s); if(t) t.classList.toggle('active', s===f);
  });
  render();
}

function render(){
  const q=(document.getElementById('q').value||'').trim().toLowerCase();
  let list=data.jobs||[];
  if(filter!=='all') list=list.filter(j=>j.status===filter);
  if(q) list=list.filter(j=>(j.job_id||'').toLowerCase().includes(q)||(j.voice_id||'').toLowerCase().includes(q)||(j.prompt||'').toLowerCase().includes(q)||(j.error||'').toLowerCase().includes(q));
  const tb=document.getElementById('rows');
  if(!list.length){ tb.innerHTML='<tr><td colspan="11" class="empty">No jobs</td></tr>'; return; }
  tb.innerHTML=list.map((j,i)=>{
    const url=j.audio_src;
    const emo=(j.tones||[]).filter(Boolean).join(', ')||'—';
    const donor=j.donor_set?` · ${esc(j.donor_set)}`:'';
    const sj=j.job_id&&j.job_id.length>18?j.job_id.slice(0,16)+'…':j.job_id;
    return `<tr>
      <td class="num" style="color:var(--dim)">${j.position?'#'+j.position:i+1}</td>
      <td><span class="badge ${esc(j.status)}">${esc(j.status)}</span></td>
      <td><span class="mono" title="${esc(j.job_id)}">${esc(sj||'—')}</span></td>
      <td><div class="snippet" title="${esc(j.prompt||'')}">${esc(j.prompt||'—')}</div></td>
      <td><span class="tag">${esc(j.voice_id||'auto')}</span></td>
      <td><span class="mono" title="donor: ${esc(j.donor_set||'auto')} (${esc(j.sex||'—')})">${esc(emo)}${donor}</span></td>
      <td class="num">${j.chunks_total||0}</td>
      <td class="num" style="color:var(--muted)">${esc((j.created||'').split(' ')[1]||'')}</td>
      <td class="num">${secs(j.waited_s)}</td>
      <td class="num">${secs(j.elapsed_s)}</td>
      <td>${url?`<button class="btn" style="padding:.25rem .55rem" onclick="play('${esc(url)}')">▶</button> <a class="btn" style="padding:.25rem .55rem" href="${esc(url)}" target="_blank">URL</a>`:(j.error?`<span style="color:var(--fail)" title="${esc(j.error)}">Error</span>`:'—')}</td>
    </tr>`;
  }).join('');
}

function play(u){ const a=document.getElementById('audio'); if(playing===u){a.pause();playing=null;return;} a.src=u; a.play(); playing=u; }

function update(d){
  data=d;
  const c=d.counts||{};
  document.getElementById('k-running').textContent=c.running||0;
  document.getElementById('k-queued').textContent=c.queued||0;
  document.getElementById('k-completed').textContent=c.completed||0;
  document.getElementById('k-failed').textContent=c.failed||0;
  document.getElementById('k-total').textContent=d.total||0;
  render();
}

async function tick(){
  try{
    const r=await fetch('/webhook/jobs'); const d=await r.json();
    document.getElementById('dot').classList.remove('off'); update(d);
    const h=await (await fetch('/webhook/health')).json();
    const up=document.getElementById('v-upload'); up.textContent=h.upload_token?'configured':'missing';
    document.getElementById('pill-upload').className='pill '+(h.upload_token?'ok':'warn');
  }catch(e){ document.getElementById('dot').classList.add('off'); }
}

function setInterval2(v){ ms=+v; if(timer) clearInterval(timer); if(ms) timer=setInterval(tick,ms); }
function openTest(){ document.getElementById('test-bg').classList.add('on'); }
function closeTest(){ document.getElementById('test-bg').classList.remove('on'); }
async function submitTest(){
  const body={ prompt:document.getElementById('m-prompt').value,
               voice_id:document.getElementById('m-voice').value.trim(),
               sex:document.getElementById('m-sex').value,
               donor_set:document.getElementById('m-donor').value.trim(),
               callback_url:document.getElementById('m-cb').value.trim() };
  try{
    const r=await fetch('/webhook/live-ai-create-new',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.status==='success'){ toast('Queued: '+d.job_id); closeTest(); tick(); }
    else toast('Error: '+(d.error||'failed'));
  }catch(e){ toast('Request failed'); }
}

tick(); setInterval2(2000);
</script>
</body>
</html>"""
