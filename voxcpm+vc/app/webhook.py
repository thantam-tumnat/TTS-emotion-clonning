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
from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.annotator import annotator
from app.segmenter import segment_text
from app.renderers import get_renderer
from app.renderers.voxcpm import split_style_chunk_specs
from app.services.siangtts_service import siangtts_service
from app.services.voxcpm_vc_service import voxcpm_vc_service

router = APIRouter()

WORK_ROOT = Path(settings.webhook_work_dir)


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
    parts: List[str]
    tones: List[Optional[str]]
    breaks: List[bool]
    prompt: str = ""
    status: str = "queued"              # queued | running | completed | failed
    error: Optional[str] = None
    file_url: Optional[str] = None
    donor_set: Optional[str] = None     # the actor whose emotion is cloned
    gender: Optional[str] = None        # sex used to pick/validate the donor
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None

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

    try:
        work.mkdir(parents=True, exist_ok=True)

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
        await _post_callback(job.callback_url, job, error=None)
        print(f"[{job.queue_id}] done -> {job.file_url}")

    except Exception as exc:                                          # noqa: BLE001
        job.status = "failed"
        job.error = str(exc)
        traceback.print_exc()
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

def _pick_donor_set(sex: str, donor_set: str):
    """Choose the donor whose emotion is cloned, returning (donor_set, gender).

    An explicit ``donor_set`` is pinned as-is (the synth validates it). Otherwise a
    random complete set of the requested sex is drawn, so successive jobs vary the
    actor instead of always cloning the top-ranked one. ``sex`` defaults to the
    configured gender when blank. When no set matches, donor_set comes back None and
    the gender is handed on so the synth can resolve or raise a clear error.
    """
    ds = donor_set.strip()
    if ds:
        return ds, None
    g = (sex or settings.default_gender or "female").strip().lower()
    want = "male" if g.startswith("m") else "female"
    sets = voxcpm_vc_service.list_donor_sets()
    pool = [s["id"] for s in sets if s.get("gender") == want and s.get("complete")]
    if not pool:
        pool = [s["id"] for s in sets if s.get("gender") == want]
    if pool:
        return random.choice(pool), want
    return None, want


def _plan_chunks(text: str):
    """Split into per-emotion chunks. Hand-written style tags win; otherwise the
    text is auto-annotated so each chunk carries the tone that selects its donor."""
    specs = split_style_chunk_specs(text, use_llm=True)
    if specs:
        return (
            [s.text for s in specs],
            [s.tone for s in specs],
            [s.break_before for s in specs],
        )

    clauses = segment_text(text)
    annotated = annotator.annotate(original_text=text, clauses=clauses)
    renderer = get_renderer("voxcpm")
    rendered = renderer.render(annotated.segments)
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

async def _accept(body: WebhookBody) -> JSONResponse:
    """Validate + enqueue. Chunk planning runs here (not in the worker) so a bad
    script is rejected in the HTTP response instead of only via callback."""
    queue_id = body.queue_id or str(uuid.uuid4())
    callback_url = body.callback_url or settings.siangtts_default_callback

    parts, tones, breaks = await asyncio.to_thread(_plan_chunks, body.prompt)
    if not any(p.strip() for p in parts):
        return JSONResponse(
            {"status": "error", "error": "prompt is empty after normalize"},
            status_code=400,
        )

    # Pick the donor now (not in the worker) so the choice is fixed when the job is
    # queued and visible on the dashboard, and a random pick is stable across retries.
    donor_set, gender = await asyncio.to_thread(_pick_donor_set, body.sex, body.donor_set)

    job = Job(
        job_id=body.job_id or queue_id,
        queue_id=queue_id,
        voice_id=body.voice_id.strip() or settings.webhook_default_voice.strip(),
        callback_url=callback_url,
        parts=parts,
        tones=tones,
        breaks=breaks,
        prompt=body.prompt,
        donor_set=donor_set,
        gender=gender,
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
    print(f"[{queue_id}] queued — {len(parts)} chunk(s), voice={job.voice_id or 'auto'}")
    return JSONResponse({"status": "success", "job_id": job.job_id, "chunks": len(parts)})


@router.post("/webhook/live-ai-create-new")
async def webhook(body: WebhookBody) -> JSONResponse:
    return await _accept(body)


@router.post("/live-ai-create-new")
async def webhook_bare(body: WebhookBody) -> JSONResponse:
    """Bare alias for callers pointed straight at the old n8n node path."""
    return await _accept(body)


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
      <div><label>Voice ID (leave blank = auto)</label><input class="fc" id="m-voice" placeholder="auto"></div>
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
    if(d.status==='success'){ toast('Queued: '+d.job_id+' ('+d.chunks+' chunk)'); closeTest(); tick(); }
    else toast('Error: '+(d.error||'failed'));
  }catch(e){ toast('Request failed'); }
}

tick(); setInterval2(2000);
</script>
</body>
</html>"""
