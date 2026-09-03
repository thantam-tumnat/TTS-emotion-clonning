"""The one process that owns the model: queue, worker, and job execution.

Everything that needs the GPU happens here and nowhere else. The webhook service
(:8010) and the tone studio (:8011) are clients over HTTP — see `src/gpu_service.py`
for the wire format and `src/gpu_client.py` for the caller side.

Two things make a single worker the right shape rather than a semaphore:

* there is one GPU, so generation is serial no matter how it is expressed; and
* LoRA strength is *global mutable state* on the model (`src/lora.py`), and the two
  pipelines want different values. Serialising jobs is what makes it safe to set the
  scale per job instead of freezing one pipeline's preference into the process.

Lanes exist because the two clients have different patience. A webhook script is
minutes of batch work nobody is watching; a studio click is a person waiting. The
interactive lane jumps the queue, with a cap so a busy studio cannot starve
production traffic.
"""

from __future__ import annotations

import asyncio
import gc
import io
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import lora as lora_mod
from .voices import UnknownVoice, VoiceStore

LANES = ("interactive", "batch")

# How many interactive jobs may jump ahead of a waiting batch job before one batch
# job goes through regardless. Production traffic is never starved; the studio still
# gets served first in the common case where the batch lane is empty or shallow.
INTERACTIVE_BURST = int(os.environ.get("SIANGTTS_INTERACTIVE_BURST", "3"))

# Finished jobs kept for /v2/jobs. State is in memory, so this is the only thing
# stopping a long-lived process from growing without bound.
MAX_HISTORY = int(os.environ.get("SIANGTTS_GPU_MAX_HISTORY", "500"))

# --------------------------------------------------------------------------- #
# VRAM housekeeping
#
# This process shares one card with the SeedVC worker on :8022. PyTorch's caching
# allocator never returns a freed block to the driver by itself, so whatever peak
# this process reaches is memory the other one can never have -- which is why the
# reclaim policy below is a threshold rather than "always" or "never".
# --------------------------------------------------------------------------- #

# Reclaim when free VRAM drops below this fraction of the card. `empty_cache()`
# forces a full device sync, so paying for it after every job was pure latency on
# a healthy host; never paying for it strands this process's high-water mark.
RECLAIM_BELOW = float(os.environ.get("VOXCPM_RECLAIM_BELOW", "0.15"))

# --------------------------------------------------------------------------- #
# Idle policy
#
# The weights are ~6 GB of a 12 GB card that other GPU applications on this box
# also want, and a service that has finished its queue is holding them for
# nobody. "unload" drops them once the queue has been quiet for a while and
# reloads on the next job; "hot" is the old always-resident behaviour, kept as
# the escape hatch for when the reload costs more than the memory is worth.
#
# Deliberately not offered: parking the weights in system RAM. It frees the same
# VRAM, but it takes the RAM from the very applications this is meant to make
# room for -- and on a box that is already paging, Windows would put them on
# disk anyway, which makes the "fast" path slower than a clean reload.
# --------------------------------------------------------------------------- #

IDLE_MODE = os.environ.get("SIANGTTS_IDLE_MODE", "unload").strip().lower()

# Quiet time before the weights are dropped. The floor is set by the pipeline
# rather than by taste: a take generates every chunk here, then converts every
# chunk on the SeedVC worker, so this process sits idle through a whole
# conversion phase in the middle of work that is not finished. Too short and
# every take pays for a reload it did not need.
IDLE_TTL = float(os.environ.get("SIANGTTS_IDLE_TTL", "180"))

# How often the watcher looks. Cheap: it takes no lock unless it means to act.
IDLE_CHECK_S = float(os.environ.get("SIANGTTS_IDLE_CHECK", "10"))


def _free_fraction():
    """Free VRAM as a fraction of the card's total, or None when off-GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return (free / total) if total else None
    except Exception:                                                # noqa: BLE001
        return None


def reclaim_vram(force: bool = False) -> bool:
    """Hand cached-but-unused VRAM back to the driver. See RECLAIM_BELOW."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        if os.environ.get("VOXCPM_EMPTY_CACHE") == "always":
            force = True
        if not force:
            frac = _free_fraction()
            if frac is None or frac > RECLAIM_BELOW:
                return False
        torch.cuda.empty_cache()
        return True
    except Exception:                                                # noqa: BLE001
        return False


def is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory failure, however it surfaced.

    Matched on the message as well as the type, because an OOM raised inside a
    fused kernel or a cuBLAS workspace allocation arrives as a plain RuntimeError.
    Getting this wrong in the permissive direction is cheap (one extra reclaim);
    getting it wrong in the strict direction means the queue keeps feeding a card
    that has no memory left.
    """
    try:
        import torch

        oom_types = tuple(
            t for t in (
                getattr(torch.cuda, "OutOfMemoryError", None),
                getattr(torch, "OutOfMemoryError", None),
            )
            if isinstance(t, type)
        )
        if oom_types and isinstance(exc, oom_types):
            return True
    except Exception:                                                # noqa: BLE001
        pass
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in ("out of memory", "outofmemoryerror", "alloc_failed"))


class ModelHolder:
    """The model, present or absent, behind one reference that never changes.

    Everything that used to hold a `Synthesizer` holds one of these instead and
    goes on calling `.synth(...)`, `.build_voice(...)`, `.tts_model` as before:
    the attribute lookup is what loads the weights. That is the point -- there is
    no list of call sites to keep in sync, and a path that reaches the model
    without meaning to gets a load rather than an AttributeError.

    Two attributes deliberately do *not* load. `sample_rate` and `is_stub` are
    read by /health, which a dashboard polls every 1.5 s; answering those from
    the model would pin the weights in memory forever and defeat the whole
    exercise. They answer from cached values instead.

    Loading is serialised on its own lock. Callers already hold the engine's
    `gpu_lock`, but they reach this from worker threads, and two threads arriving
    together must not each build a model.
    """

    def __init__(
        self,
        build,
        *,
        mode: str = IDLE_MODE,
        ttl: float = IDLE_TTL,
        sample_rate: int = 48000,
        is_stub: bool = False,
    ) -> None:
        self._build = build
        self._model: Any = None
        self._lock = threading.Lock()
        self.mode = mode if mode in ("hot", "unload") else "unload"
        self.ttl = ttl
        # Cached so /health can answer with no model resident. The sample rate is
        # a property of the checkpoint, so the first load turns this default into
        # the real value and it never changes again.
        self._sample_rate = int(sample_rate)
        self._is_stub = bool(is_stub)
        self.last_used = time.time()
        self.loads = 0
        self.releases = 0
        self.load_seconds = 0.0

    # -- state ------------------------------------------------------------ #

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_stub(self) -> bool:
        return self._is_stub

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used

    def touch(self) -> None:
        """Mark the model as wanted now. Called when a job finishes as well as
        when one starts, so the idle clock runs from the end of the work."""
        self.last_used = time.time()

    # -- the model itself -------------------------------------------------- #

    def get(self) -> Any:
        """The model, loading it if it is not resident."""
        self.last_used = time.time()
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:                # else: lost the race, it is built
                t0 = time.time()
                model = self._build()
                took = time.time() - t0
                self._model = model
                self.loads += 1
                self.load_seconds += took
                self._sample_rate = int(getattr(model, "sample_rate", self._sample_rate))
                self._is_stub = bool(getattr(model, "is_stub", self._is_stub))
                if self.loads > 1:
                    print(f"[gpu] model back in {took:.1f}s (load #{self.loads})", flush=True)
            self.last_used = time.time()
            return self._model

    # -- prompt-cache I/O, which is not model work ------------------------- #
    #
    # `Synthesizer` hosts these two, but neither touches the model: a prompt
    # cache is not model state, it is a file, and `load_voice` reads it with
    # map_location="cpu" precisely so it never becomes one (inference.py, and
    # tests/test_inference_device.py pins the invariant). Letting them fall
    # through to `__getattr__` would load six gigabytes of weights to read a
    # file -- which is exactly what happened: the voice cache hits on nearly
    # every resolve, so every resolve woke a model that had just been released.
    # Seventeen loads, zero renders.

    # The shortcut is only taken for the real model. `StubSynthesizer` keeps its
    # own cache format (src/stub_synth.py), so borrowing `Synthesizer`'s
    # implementation there would read a file written by something else -- and a
    # stub costs nothing to build anyway, so there is nothing to save.

    def _cache_io(self, name: str):
        if self._model is None and not self._is_stub:
            from .inference import Synthesizer

            # The real implementation rather than a copy of it, called with this
            # holder standing in for an instance it never reads.
            return getattr(Synthesizer, name).__get__(self, ModelHolder)
        return getattr(self.get(), name)

    def load_voice(self, path):
        return self._cache_io("load_voice")(path)

    def save_voice(self, prompt_cache, path):
        return self._cache_io("save_voice")(prompt_cache, path)

    def __getattr__(self, name: str) -> Any:
        """Anything not defined above is the model's own API, so a load is what
        the caller is asking for. Private names are excluded: they arrive from
        copy and pickle protocols during construction, before there is anything
        to forward to, and answering those with a model load would be absurd."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.get(), name)

    # -- releasing --------------------------------------------------------- #

    def should_release(self) -> bool:
        """Cheap check -- no lock, no CUDA call. The watcher runs this every tick
        and only reaches for the GPU lock when it comes back true."""
        return self.mode == "unload" and self.loaded and self.idle_seconds >= self.ttl

    def release(self, reason: str = "idle") -> bool:
        """Drop the weights and hand the VRAM back to the driver.

        The caller must hold the engine's `gpu_lock`. Releasing underneath a
        running generation would pull the model out from under it.
        """
        with self._lock:
            model = self._model
            if model is None:
                return False
            self._model = None

        idle_for = self.idle_seconds
        del model
        # collect() before empty_cache(), and both are needed: the tensors only
        # become unreachable once the reference graph is collected, and the
        # caching allocator only returns blocks no tensor still owns. Skip the
        # collect and the weights stay on the card, which is the whole exercise.
        gc.collect()
        freed = _release_cuda()
        self.releases += 1
        print(
            f"[gpu] model released ({reason}) after {idle_for:.0f}s idle — "
            f"{'VRAM returned to the driver' if freed else 'nothing on the GPU to reclaim'}",
            flush=True,
        )
        return True


def _release_cuda() -> bool:
    """`empty_cache()` unconditionally. The RECLAIM_BELOW threshold that guards
    the per-job reclaim does not apply here: giving the memory up is the point,
    not a side effect worth deferring until the card is tight."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        # `empty_cache()` initialises this process's CUDA context if there is not
        # one yet -- so calling it on a process holding nothing takes several
        # hundred MB of the card in the name of giving memory back. If CUDA was
        # never initialised there is, by definition, nothing here to reclaim.
        if not torch.cuda.is_initialized():
            return False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return True
    except Exception:                                                # noqa: BLE001
        return False


@dataclass
class RenderJob:
    chunks: list[str]
    voice: Optional[dict] = None
    cfg_value: float = 2.0
    timesteps: int = 10
    lora: Any = None
    output: dict = field(default_factory=lambda: {"mode": "npz"})
    lane: str = "batch"
    client: str = ""
    job_id: str = field(default_factory=lambda: f"g_{uuid.uuid4().hex[:10]}")

    status: str = "queued"              # queued | running | completed | failed | cancelled
    done: int = 0
    error: Optional[str] = None
    # "oom" when the failure was CUDA out-of-memory. The queue gateway keys on it
    # to cancel the rest of the request rather than feeding the same full card
    # with the sibling chunks of a take that can no longer be assembled.
    error_kind: Optional[str] = None
    result: Optional[dict] = None       # metadata; bytes live in `payload`
    payload: Optional[bytes] = None
    voice_handle: Optional[str] = None
    lora_applied: Optional[dict] = None
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    def as_dict(self, position: Optional[int] = None) -> dict:
        now = time.time()
        ran = ((self.finished or now) - self.started) if self.started else None
        return {
            "job_id": self.job_id,
            "client": self.client,
            "lane": self.lane,
            "status": self.status,
            "position": position,
            "chunks_total": len(self.chunks),
            "chunks_done": self.done,
            "chunks": self.chunks,
            "progress": f"{self.done}/{len(self.chunks)}",
            "voice_handle": self.voice_handle,
            "lora": self.lora_applied,
            "created_ts": self.created,
            "started_ts": self.started,
            "finished_ts": self.finished,
            "waited_s": round((self.started or now) - self.created, 1),
            "elapsed_s": round(ran, 1) if ran is not None else None,
            "result": self.result,
            "error": self.error,
            "error_kind": self.error_kind,
        }


class Engine:
    """Model + voice store + queue. One instance per process."""

    def __init__(
        self,
        synth: Any,
        voices: VoiceStore,
        work_root: Path,
        *,
        default_lora: Any = None,
    ) -> None:
        self.synth = synth
        # The same object under a second name, when it is one: `synth` is what
        # every generation path calls, `holder` is what the lifecycle paths need.
        # None when a plain Synthesizer was passed in -- the tests do that, and a
        # model that cannot be released simply never is.
        self.holder = synth if isinstance(synth, ModelHolder) else None
        self.voices = voices
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.default_lora = default_lora

        self.jobs: dict[str, RenderJob] = {}
        # Every path that touches the GPU takes this, so "one GPU, one job at a
        # time" is enforced by the model owner rather than by whoever happens to
        # be calling. The queue worker was never the only caller:
        # `/v2/direct_render` executes jobs straight off the event loop, and the
        # voice endpoints encode (and `/v2/voices/seed` generates) outside the
        # queue entirely -- so a voice registration could run a second workload on
        # a card that was already mid-generation.
        self.gpu_lock = asyncio.Lock()
        self.queues: dict[str, asyncio.Queue] = {lane: asyncio.Queue() for lane in LANES}
        self.running: Optional[str] = None
        self._worker: Optional[asyncio.Task] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._interactive_streak = 0
        # What the model is currently scaled to, so an unchanged scale costs nothing.
        self._lora_state: Optional[tuple[float, float]] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_forever())
        if self._idle_task is None and self.holder is not None and self.holder.mode != "hot":
            self._idle_task = asyncio.create_task(self._idle_loop())

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.synth, "sample_rate", 48000))

    @property
    def is_stub(self) -> bool:
        return bool(getattr(self.synth, "is_stub", False))

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #

    def submit(self, job: RenderJob) -> RenderJob:
        if job.lane not in self.queues:
            job.lane = "batch"
        self.jobs[job.job_id] = job
        self._evict_history()
        self.queues[job.lane].put_nowait(job)
        return job

    def cancel(self, job_id: str) -> bool:
        """Cancel a job that has not started. A running job is left alone — there is
        no way to interrupt a generation mid-chunk, and killing the worker would take
        every other queued job with it."""
        job = self.jobs.get(job_id)
        if job is None or job.status != "queued":
            return False
        job.status = "cancelled"
        job.finished = time.time()
        job._event.set()
        return True

    async def wait(self, job_id: str, timeout: float) -> Optional[RenderJob]:
        """Block until the job finishes, or return None on timeout.

        This is what makes the studio's synchronous UI work without a polling loop:
        it posts a job, waits out the generation on the same connection, and gets
        audio back. A timeout is not an error — the job keeps running and the caller
        falls back to polling.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.terminal:
            return job
        try:
            await asyncio.wait_for(job._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return job

    def positions(self) -> dict[str, int]:
        """Place in line, 1-based, in the order the worker will actually take them.

        Reproduced from the job records rather than read off the queues, which are
        opaque — jobs are enqueued in creation order within a lane, and the worker
        prefers interactive, so interleaving the two by that rule gives the true
        order barring an anti-starvation swap.
        """
        waiting = [j for j in self.jobs.values() if j.status == "queued"]
        inter = sorted((j for j in waiting if j.lane == "interactive"), key=lambda j: j.created)
        batch = sorted((j for j in waiting if j.lane == "batch"), key=lambda j: j.created)
        return {j.job_id: i for i, j in enumerate(inter + batch, 1)}

    def _queued(self) -> bool:
        """Anything waiting in either lane. Read directly off the queues rather
        than off the job records: this decides whether the model may go away, and
        a job that has been enqueued but not yet marked is still a job."""
        return any(not self.queues[lane].empty() for lane in LANES)

    async def release_model(self, reason: str = "manual", *, only_if_idle: bool = False) -> bool:
        """Give the card back. The one path that drops the weights.

        Takes `gpu_lock`, so it waits out a running generation rather than
        pulling the model out from under it, and re-checks the idle conditions
        once it has the lock -- a job can arrive in the gap between the watcher
        deciding and the lock being granted.
        """
        if self.holder is None:
            return False
        async with self.gpu_lock:
            if only_if_idle and (self._queued() or not self.holder.should_release()):
                return False
            freed = await asyncio.to_thread(self.holder.release, reason)
            if freed:
                # A reloaded model comes back at its default LoRA scale, so the
                # cached "what the model is scaled to now" is now a lie -- and a
                # lie that makes `_apply_lora` skip `set_lora_strength` outright,
                # shipping a take rendered without the Thai LoRA and no error to
                # show for it. Resetting it here, in the only place that drops
                # the model, is what keeps the two from drifting apart.
                self._lora_state = None
            return freed

    async def _idle_loop(self) -> None:
        """Hand the card back once nothing has needed it for `IDLE_TTL`.

        The cheap checks run first and take no lock at all, so a busy service
        pays nothing for this beyond a comparison every `IDLE_CHECK_S`.
        """
        while True:
            try:
                await asyncio.sleep(IDLE_CHECK_S)
                if self.holder is None or not self.holder.should_release():
                    continue
                if self.running is not None or self._queued():
                    continue
                await self.release_model("idle", only_if_idle=True)
            except asyncio.CancelledError:
                raise
            except Exception:                                        # noqa: BLE001
                traceback.print_exc()

    def _evict_history(self) -> None:
        """Bounded history — insertion-ordered, so drop the oldest finished jobs
        first and never evict anything still queued or running."""
        if len(self.jobs) <= MAX_HISTORY:
            return
        for jid, j in list(self.jobs.items()):
            if len(self.jobs) <= MAX_HISTORY:
                break
            if j.terminal:
                del self.jobs[jid]

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #

    async def _next_job(self) -> RenderJob:
        while True:
            inter, batch = self.queues["interactive"], self.queues["batch"]

            if not inter.empty() and (batch.empty() or self._interactive_streak < INTERACTIVE_BURST):
                self._interactive_streak += 1
                return inter.get_nowait()
            if not batch.empty():
                self._interactive_streak = 0
                return batch.get_nowait()
            if not inter.empty():
                self._interactive_streak += 1
                return inter.get_nowait()

            # Both empty: wait on whichever fills first. No lane preference to apply
            # — with nothing queued there is nothing to jump ahead of. If both fire
            # at once, take one and put the other back for the next round.
            getters = [asyncio.ensure_future(q.get()) for q in (inter, batch)]
            done, pending = await asyncio.wait(getters, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
            first = done.pop()
            for extra in done:                    # both fired at once; re-queue the loser
                job = extra.result()
                self.queues[job.lane].put_nowait(job)
            return first.result()

    async def _run_forever(self) -> None:
        while True:
            job = await self._next_job()
            if job.status == "cancelled":
                continue
            try:
                await self._execute(job)
            except Exception:                                        # noqa: BLE001
                traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def _apply_lora(self, spec: Any) -> dict:
        lm, dit = lora_mod.resolve(spec if spec is not None else self.default_lora)
        if self._lora_state == (lm, dit):
            return {"lm": lm, "dit": dit, "unchanged": True}
        applied = lora_mod.set_lora_strength(getattr(self.synth, "tts_model", None), lm, dit)
        self._lora_state = (lm, dit)
        return applied

    def _resolve_voice(self, spec: Optional[dict], client: str = "") -> Optional[str]:
        """Voice spec -> handle. Returns None for an unconditioned generation."""
        if not spec:
            return None
        if spec.get("handle"):
            self.voices.get(spec["handle"])        # raises UnknownVoice -> 410
            return spec["handle"]
        if spec.get("speaker_id"):
            allow_sidecar = spec.get("allow_sidecar")
            if allow_sidecar is None:
                allow_sidecar = False if client == "tone-studio" else True
            return self.voices.resolve_speaker(
                spec["speaker_id"],
                spec.get("ref_text") or "",
                allow_sidecar=bool(allow_sidecar),
            )
        if spec.get("seed"):
            return self.voices.seed(
                lambda text: self.synth.synth(text, cfg_value=2.0, inference_timesteps=10)
            )
        return None

    def _generate(self, text: str, cache: Any, job: RenderJob):
        if cache is None:
            return self.synth.synth(
                text, cfg_value=job.cfg_value, inference_timesteps=job.timesteps
            )
        return self.synth.synth_cached(
            text, cache, cfg_value=job.cfg_value, inference_timesteps=job.timesteps
        )

    async def _execute(self, job: RenderJob) -> None:
        async with self.gpu_lock:
            await self._execute_locked(job)

    async def _execute_locked(self, job: RenderJob) -> None:
        job.status = "running"
        job.started = time.time()
        self.running = job.job_id
        try:
            job.voice_handle = await asyncio.to_thread(self._resolve_voice, job.voice, job.client)
            cache = self.voices.get(job.voice_handle) if job.voice_handle else None
            job.lora_applied = await asyncio.to_thread(self._apply_lora, job.lora)

            client_label = job.client or "unknown"
            print(f"\n[gpu] >>> Running Job {job.job_id} from '{client_label}' (lane={job.lane}, voice={job.voice_handle or 'unpinned'}, {len(job.chunks)} chunk(s)):")
            for idx, chunk_text in enumerate(job.chunks):
                print(f"[gpu]     [{idx+1}/{len(job.chunks)}] {chunk_text!r}")

            # `files` mode writes each chunk as it is generated rather than at the
            # end. Two reasons: a long script would otherwise hold every chunk's
            # audio in memory until the last one lands, and a job that dies partway
            # leaves nothing behind to inspect or re-merge — which is what the
            # in-process version used to give the webhook for free.
            sink = self._open_sink(job)
            for i, text in enumerate(job.chunks):
                wav = await self._generate_with_oom_retry(text, cache, job, i)
                await asyncio.to_thread(sink.write, i, wav)
                job.done = i + 1

            job.result, job.payload = await asyncio.to_thread(sink.finish)
            job.status = "completed"

        except UnknownVoice as exc:
            job.status = "failed"
            job.error = f"unknown voice: {exc}"
        except Exception as exc:                                     # noqa: BLE001
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            if is_oom(exc):
                job.error_kind = "oom"
                # The whole job dies here, not just this chunk: a take missing a
                # chunk cannot be assembled, and the chunks after it would be
                # asking the same card for the memory it just refused.
                print(
                    f"[gpu] job {job.job_id} failed with CUDA OOM at chunk "
                    f"{job.done + 1}/{len(job.chunks)} — abandoning the whole job",
                    flush=True,
                )
            traceback.print_exc()
        finally:
            job.finished = time.time()
            self.running = None
            job._event.set()
            # From the end of the work, not the start: a job that ran for four
            # minutes should not come out of it already eligible for a release.
            if self.holder is not None:
                self.holder.touch()
            # Reclaim after a failure (a partial run leaves the allocator
            # fragmented) and whenever the card is genuinely tight; skip it on the
            # healthy path, where the sync costs more than the memory is worth.
            reclaim_vram(force=job.status == "failed")

    async def _generate_with_oom_retry(self, text: str, cache: Any, job: RenderJob, index: int):
        """One generation, with a single retry if it OOMs.

        A partial generation leaves the allocator fragmented, so the same chunk
        frequently fits once the cache is dropped -- and a retry here is far
        cheaper than failing a take that is otherwise done. Only OOM is retried:
        anything else would fail identically the second time.
        """
        try:
            return await asyncio.to_thread(self._generate, text, cache, job)
        except Exception as exc:                                     # noqa: BLE001
            if not is_oom(exc):
                raise
            print(
                f"[gpu] job {job.job_id} chunk {index + 1}: CUDA OOM — "
                f"reclaiming and retrying once",
                flush=True,
            )
            await asyncio.to_thread(reclaim_vram, True)
            await asyncio.sleep(0.5)
            return await asyncio.to_thread(self._generate, text, cache, job)

    # -- delivery -------------------------------------------------------- #

    def _job_dir(self, name: str) -> Path:
        """Resolve a client-supplied directory *name* under the work root.

        A name, never a path: the client picks what the folder is called, the service
        decides where it lives. Otherwise `job_dir` would be an instruction to write
        anywhere on the host's disk.

        Anything with a separator or a `..` in it is rejected rather than trimmed
        down to its last component. Silently writing to a different directory than
        the one asked for would leave the caller looking for files that are not
        there, which is a worse failure than a clear one.
        """
        safe = str(name).strip()
        if not safe or safe in (".", "..") or safe != Path(safe).name:
            raise ValueError(f"invalid job_dir {name!r} — must be a plain folder name")
        out = (self.work_root / safe).resolve()
        if not str(out).startswith(str(self.work_root.resolve())):
            raise ValueError(f"job_dir {name!r} escapes the work root")
        return out

    def _open_sink(self, job: RenderJob) -> "_Sink":
        mode = (job.output or {}).get("mode", "npz")
        if mode == "files":
            return _FileSink(
                self._job_dir(job.output.get("job_dir") or job.job_id),
                job.output.get("names") or [],
                self.sample_rate,
            )
        if mode == "npz":
            return _NpzSink(self.sample_rate)
        raise ValueError(f"unknown output mode {mode!r}")


class _Sink:
    """Where a job's chunks go as they are generated."""

    def write(self, index: int, wav) -> None:                        # pragma: no cover
        raise NotImplementedError

    def finish(self) -> tuple[dict, Optional[bytes]]:                # pragma: no cover
        raise NotImplementedError


class _FileSink(_Sink):
    """WAVs in a directory both services can see. The caller's next step is ffmpeg,
    so putting the audio on disk saves pushing it through HTTP only to be written
    out again at the other end."""

    def __init__(self, out_dir: Path, names: list[str], sample_rate: int) -> None:
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.names = names
        self.sample_rate = sample_rate
        self.paths: list[str] = []

    def write(self, index: int, wav) -> None:
        import soundfile as sf

        stem = (
            Path(self.names[index]).name
            if index < len(self.names)
            else f"{self.dir.name}_{index:03d}"
        )
        path = self.dir / f"{stem}.wav"
        sf.write(str(path), wav, self.sample_rate)
        self.paths.append(str(path))

    def finish(self) -> tuple[dict, Optional[bytes]]:
        return {
            "mode": "files",
            "dir": str(self.dir),
            "files": self.paths,
            "sample_rate": self.sample_rate,
        }, None


class _NpzSink(_Sink):
    """float32 arrays in one bundle, for a caller assembling audio in memory."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.wavs: list = []

    def write(self, index: int, wav) -> None:
        self.wavs.append(wav)

    def finish(self) -> tuple[dict, Optional[bytes]]:
        import numpy as np

        buf = io.BytesIO()
        np.savez(
            buf,
            sample_rate=np.asarray(self.sample_rate),
            count=np.asarray(len(self.wavs)),
            **{f"chunk_{i:03d}": np.asarray(w, dtype="float32") for i, w in enumerate(self.wavs)},
        )
        data = buf.getvalue()
        return {
            "mode": "npz",
            "chunks": len(self.wavs),
            "sample_rate": self.sample_rate,
            "bytes": len(data),
        }, data


__all__ = [
    "Engine",
    "RenderJob",
    "LANES",
    "INTERACTIVE_BURST",
    "MAX_HISTORY",
    "is_oom",
    "reclaim_vram",
]
