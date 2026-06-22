"""FastAPI inference server for SiangTTS (VoxCPM2 + Thai LoRA).

Loads the base model + adapter once at startup and serves:

    GET  /health            → liveness + model info
    POST /tts               → text → speech in the model's default voice
    POST /clone             → text + reference wav → speech in that voice
    GET  /                  → tiny HTML form for manual testing

Run:
    uv run uvicorn src.serve:app --host 0.0.0.0 --port 8000
    # base-only (no adapter):
    SIANGTTS_ADAPTER="" uv run uvicorn src.serve:app

Config via env:
    SIANGTTS_BASE_MODEL   default openbmb/VoxCPM2
    SIANGTTS_ADAPTER      default checkpoints/siangtts-lora-v0/latest ("" = base only)
    SIANGTTS_DEVICE       default auto
"""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from .inference import DEFAULT_BASE_MODEL, Synthesizer

_BASE_MODEL = os.environ.get("SIANGTTS_BASE_MODEL", DEFAULT_BASE_MODEL)
_ADAPTER = os.environ.get("SIANGTTS_ADAPTER", "checkpoints/siangtts-lora-v0/latest")
_DEVICE = os.environ.get("SIANGTTS_DEVICE") or None

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    adapter = _ADAPTER or None
    if adapter and not Path(adapter).exists():
        print(f"[serve] adapter path {adapter!r} not found — loading base only")
        adapter = None
    _state["synth"] = Synthesizer(
        base_model=_BASE_MODEL, adapter_path=adapter, device=_DEVICE
    )
    _state["adapter"] = adapter
    print(f"[serve] ready — base={_BASE_MODEL} adapter={adapter}")
    yield
    _state.clear()


app = FastAPI(title="SiangTTS", version="1.0", lifespan=lifespan)


def _wav_response(wav, sample_rate: int) -> Response:
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.get("/health")
def health() -> dict:
    synth: Synthesizer = _state.get("synth")
    return {
        "status": "ok" if synth else "loading",
        "base_model": _BASE_MODEL,
        "adapter": _state.get("adapter"),
        "sample_rate": getattr(synth, "sample_rate", None),
    }


@app.post("/tts")
def tts(
    text: str = Form(...),
    cfg_value: float = Form(2.5),
    timesteps: int = Form(10),
) -> Response:
    """Synthesize `text` in the model's default voice (no reference)."""
    synth: Synthesizer = _state["synth"]
    if not text.strip():
        raise HTTPException(400, "text must be non-empty")
    wav = synth.synth(text, cfg_value=cfg_value, inference_timesteps=timesteps)
    return _wav_response(wav, synth.sample_rate)


@app.post("/clone")
async def clone(
    text: str = Form(...),
    reference: UploadFile = Form(...),
    cfg_value: float = Form(2.5),
    timesteps: int = Form(10),
) -> Response:
    """Synthesize `text` in the voice of the uploaded `reference` wav (3–10 s)."""
    synth: Synthesizer = _state["synth"]
    if not text.strip():
        raise HTTPException(400, "text must be non-empty")
    suffix = Path(reference.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await reference.read())
        ref_path = tmp.name
    try:
        wav = synth.synth(
            text,
            ref_audio=ref_path,
            cfg_value=cfg_value,
            inference_timesteps=timesteps,
        )
    finally:
        os.unlink(ref_path)
    return _wav_response(wav, synth.sample_rate)


_INDEX_HTML = """<!doctype html><meta charset="utf-8"><title>SiangTTS</title>
<h2>SiangTTS — Thai TTS + Voice Cloning</h2>
<h3>Text-to-speech</h3>
<form action="/tts" method="post" enctype="multipart/form-data" target="_blank">
  <input name="text" size="60" placeholder="พิมพ์ข้อความภาษาไทย" required>
  <button>Speak</button>
</form>
<h3>Voice cloning</h3>
<form action="/clone" method="post" enctype="multipart/form-data" target="_blank">
  <input name="text" size="60" placeholder="พิมพ์ข้อความภาษาไทย" required><br>
  reference wav (3–10s): <input type="file" name="reference" accept="audio/*" required>
  <button>Clone</button>
</form>
<p>API: <code>POST /tts</code>, <code>POST /clone</code>, <code>GET /health</code></p>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML
