"""Persistent SeedVC voice-conversion worker (runs in the seed-vc venv).

The studio (:8011) generates emotional Thai speech with Thonburian, then posts each
chunk here to be re-timbred onto the target speaker's voice. SeedVC's own
``inference.py`` reloads the whole model on every call; ``SeedVCWrapper`` loads it
once, so this wraps that in a tiny HTTP server the studio can call per chunk.

Runs in seed-vc's own virtualenv (torch 2.4), which cannot coexist with the
studio's env — hence a separate process, not an import. Start it with that venv's
python and the seed-vc checkout on the path:

    <seedvc-venv>/python tools/seedvc_server.py \
        --seedvc-repo <path-to-seed-vc> --port 8022

POST /convert  {source, target, output, f0_condition, auto_f0_adjust,
                diffusion_steps, semi_tone_shift}  -> writes `output`, returns
                {"output":..., "sample_rate":...}. Paths are absolute and local;
this binds to localhost and is trusted, like the sibling GPU service.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# VRAM housekeeping
#
# This process does not own the GPU. The VoxCPM2 service on :8021 holds ~8 GB of
# it, and PyTorch's caching allocator never hands a freed block back to the
# driver on its own -- so whatever peak this worker reaches, it keeps, and the
# other process can never borrow it. Everything below exists to keep that peak
# bounded and predictable rather than to squeeze the last megabyte.
# --------------------------------------------------------------------------- #

# Reclaim only when the device is actually tight. `empty_cache()` forces a full
# device sync: after every convert it stalled the diffusion hot path for memory
# the allocator would have reused on the next call anyway, but never calling it
# means this process sits on its high-water mark forever. A threshold gets the
# reclaim without paying for it on the common path.
RECLAIM_BELOW = float(os.getenv("SEEDVC_RECLAIM_BELOW", "0.15"))

# Hard ceiling on this process's share of the card, as a fraction. Without it the
# two processes race for whatever is free and whoever allocates second dies --
# which makes the failure land on an arbitrary victim rather than on the process
# that is actually over budget. "" disables the cap.
MEM_FRACTION = os.getenv("SEEDVC_MEM_FRACTION", "").strip()


def _free_fraction():
    """Free VRAM as a fraction of the card's total, or None when off-GPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return (free / total) if total else None


def _reclaim(force: bool = False) -> bool:
    """Return cached-but-unused VRAM to the driver. See RECLAIM_BELOW."""
    import torch

    if not torch.cuda.is_available():
        return False
    if os.getenv("SEEDVC_EMPTY_CACHE") == "always":
        force = True
    if not force:
        frac = _free_fraction()
        if frac is None or frac > RECLAIM_BELOW:
            return False
    torch.cuda.empty_cache()
    return True


def _is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory failure, however it surfaced.

    Matched on the message as well as the type: an OOM inside a fused kernel or a
    cuBLAS workspace allocation arrives as a RuntimeError, and treating that as an
    ordinary failure would retry it forever instead of tearing the request down.
    """
    import torch

    oom_types = tuple(
        t for t in (getattr(torch.cuda, "OutOfMemoryError", None), getattr(torch, "OutOfMemoryError", None))
        if isinstance(t, type)
    )
    if oom_types and isinstance(exc, oom_types):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in ("out of memory", "outofmemoryerror", "alloc_failed"))


class ConvertRequest(BaseModel):
    source: str
    target: str
    output: str
    f0_condition: bool = True
    auto_f0_adjust: bool = True
    diffusion_steps: int = 25
    semi_tone_shift: int = 0
    inference_cfg_rate: float = 0.7


def _load_env_file(path: str = ".env") -> None:
    """Load `KEY=value` lines from `.env` in the cwd. Real env vars always win."""
    env_path = Path(path)
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'").strip('"')
            if k and k not in os.environ:
                os.environ[k] = v


def main() -> int:
    _load_env_file()  # picks up HF_TOKEN etc. from voxcpm+vc/.env (this script's cwd)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seedvc-repo", required=True, help="path to a seed-vc checkout")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8022)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    repo = Path(args.seedvc_repo).resolve()
    if not (repo / "seed_vc_wrapper.py").exists():
        print(f"[seedvc] {repo} is not a seed-vc checkout", file=sys.stderr)
        return 2
    # seed_vc_wrapper imports `modules.commons` etc. by relative name.
    sys.path.insert(0, str(repo))

    import numpy as np
    import soundfile as sf
    import torch
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from seed_vc_wrapper import SeedVCWrapper

    print(f"[seedvc] loading SeedVCWrapper (device={args.device or 'auto'}) …", flush=True)
    t0 = time.time()
    wrapper = SeedVCWrapper(device=args.device)
    print(f"[seedvc] ready in {time.time()-t0:.0f}s on {wrapper.device}", flush=True)

    if MEM_FRACTION and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(float(MEM_FRACTION))
            print(f"[seedvc] VRAM capped at {float(MEM_FRACTION):.0%} of the card", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"[seedvc] could not cap VRAM ({e}); running uncapped", file=sys.stderr)

    # One GPU, one conversion at a time.
    #
    # `/convert` is a plain `def`, so Starlette runs it in the anyio threadpool --
    # up to 40 concurrent calls into ONE shared SeedVCWrapper. Nothing upstream
    # gates it either: the Go queue serialises *generation*, not conversion, so a
    # take converting can overlap with the next take converting. Two diffusion
    # runs that each fit alone do not fit together, and the second one dies with
    # an OOM that looks like a capacity problem rather than a scheduling one.
    # Queueing here costs latency under load and nothing at all when idle.
    gpu_lock = threading.Lock()

    app = FastAPI(title="SeedVC worker", version="1.0.0")

    @app.get("/health")
    def health() -> JSONResponse:
        body = {
            "status": "ok",
            "device": str(wrapper.device),
            "busy": gpu_lock.locked(),
        }
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            body["vram"] = {
                "free_gb": round(free / 1024**3, 2),
                "total_gb": round(total / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            }
        return JSONResponse(body)

    @app.post("/convert")
    def convert(req: ConvertRequest) -> JSONResponse:
        for p in (req.source, req.target):
            if not Path(p).exists():
                return JSONResponse({"error": f"missing file: {p}"}, status_code=400)

        with gpu_lock:
            for attempt in (1, 2):
                try:
                    # inference_mode, not just no_grad: SeedVC's own wrapper does
                    # not set either, so without it every diffusion step keeps
                    # autograd bookkeeping for a backward pass nobody will run.
                    with torch.inference_mode():
                        # stream_output=False returns one numpy array (the whole take).
                        audio = wrapper.convert_voice(
                            source=req.source,
                            target=req.target,
                            diffusion_steps=req.diffusion_steps,
                            inference_cfg_rate=req.inference_cfg_rate,
                            f0_condition=req.f0_condition,
                            auto_f0_adjust=req.auto_f0_adjust,
                            pitch_shift=req.semi_tone_shift,
                            stream_output=False,
                        )

                    # f0-conditioned models run at 44.1k, the base models at 22.05k.
                    sr = 44100 if req.f0_condition else 22050
                    audio = np.asarray(audio, dtype="float32").squeeze()
                    Path(req.output).parent.mkdir(parents=True, exist_ok=True)
                    sf.write(req.output, audio, sr, subtype="PCM_16")
                    _reclaim()
                    return JSONResponse(
                        {"output": req.output, "sample_rate": sr, "frames": int(audio.size)}
                    )

                except Exception as e:                            # noqa: BLE001
                    oom = _is_oom(e)

                    # One retry, and only for an OOM: a partial run leaves the
                    # allocator fragmented, and a full reclaim often makes the
                    # same request fit on the second try. Retrying anything else
                    # would just burn the GPU on a request that cannot succeed.
                    if oom and attempt == 1:
                        print(
                            f"[seedvc] CUDA OOM on convert; reclaiming and retrying once "
                            f"(free={_free_fraction() or 0:.0%})",
                            file=sys.stderr, flush=True,
                        )
                        _reclaim(force=True)
                        time.sleep(0.5)
                        continue

                    import traceback
                    traceback.print_exc()
                    _reclaim(force=True)
                    if oom:
                        # 503, not 500: the request was never wrong, the GPU was
                        # full. `error_kind` is what the queue gateway keys on to
                        # tear down the whole request instead of this chunk alone.
                        return JSONResponse(
                            {"error": f"{type(e).__name__}: {e}", "error_kind": "oom"},
                            status_code=503,
                        )
                    return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

        return JSONResponse({"error": "convert fell through without a result"}, status_code=500)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
