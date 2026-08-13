"""FastAPI inference server for SiangTTS (VoxCPM2 + Thai LoRA).

Interface modelled on KhongkhunAPI: drop reference clips in `ref/`, their
encodings are precomputed at startup and cached in `voice_cache/`, and you synthesize
either zero-shot (upload a reference) or by a registered `speaker_id`.

VoxCPM2's "embedding" is a **prompt cache** (encoded reference latents), not a
d-vector — persisted with torch.save rather than np.save.

    GET  /health
    POST /tts                      text [+ reference] → wav  (reference optional → default voice)
    POST /tts/speaker/{speaker_id} text in a registered voice (reuses cached encoding)
    POST /speakers                 register a voice (saves clip to ref/ + caches it)
    GET  /speakers                 list registered voices
    DELETE /speakers/{speaker_id}  remove a voice
    GET  /                         small HTML test form · /docs for Swagger

Run:
    uv run uvicorn src.serve:app --host 0.0.0.0 --port 8000
Config via env: SIANGTTS_BASE_MODEL, SIANGTTS_ADAPTER ("" = base only), SIANGTTS_DEVICE.
"""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .inference import DEFAULT_BASE_MODEL, Synthesizer

_BASE_MODEL = os.environ.get("SIANGTTS_BASE_MODEL", DEFAULT_BASE_MODEL)
_ADAPTER = os.environ.get("SIANGTTS_ADAPTER", "checkpoints/siangtts-lora-v0/latest")
_DEVICE = os.environ.get("SIANGTTS_DEVICE") or None

# Two different things, so two different names. REF_DIR holds the audio a human
# supplied — losing it loses the voice. CACHE_DIR holds .pt files derived from
# it — deleting it only costs a re-encode. The old n8n system called its *ref*
# folder "voices" (C:\temp\tts_jobs\voices), so don't reuse that word here.
REF_DIR = Path(os.environ.get("SIANGTTS_REF_DIR", "ref"))
CACHE_DIR = Path(os.environ.get("SIANGTTS_CACHE_DIR", "voice_cache"))
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

_state: dict = {}


def _init_ref_speakers(synth: Synthesizer) -> None:
    """Precompute + cache a prompt cache for every clip in ref/ (skip if cached)."""
    REF_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    for ref_file in sorted(REF_DIR.iterdir()):
        if ref_file.suffix.lower() not in AUDIO_EXTS:
            continue
        sid = ref_file.stem
        cache_path = CACHE_DIR / f"{sid}.pt"
        if cache_path.exists() and cache_path.stat().st_mtime >= ref_file.stat().st_mtime:
            _state["voices"][sid] = synth.load_voice(cache_path)
            print(f"  [skip] {sid} (cached)")
            continue
        print(f"  [init] encoding voice '{sid}' …")
        cache = synth.build_voice(str(ref_file))
        synth.save_voice(cache, cache_path)
        _state["voices"][sid] = cache
        print(f"  [init] '{sid}' ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    adapter = _ADAPTER or None
    if adapter and not Path(adapter).exists():
        print(f"[serve] adapter {adapter!r} not found — base only")
        adapter = None
    synth = Synthesizer(base_model=_BASE_MODEL, adapter_path=adapter, device=_DEVICE)
    _state["synth"] = synth
    _state["adapter"] = adapter
    _state["voices"] = {}
    print("[serve] initialising voices from ref/ …")
    _init_ref_speakers(synth)
    print(f"[serve] ready — base={_BASE_MODEL} adapter={adapter} voices={list(_state['voices'])}")
    yield
    _state.clear()


app = FastAPI(
    title="SiangTTS API",
    description="Thai Text-to-Speech + zero-shot voice cloning (VoxCPM2 + Thai LoRA). "
    "Clone from an uploaded clip or a registered speaker whose reference is "
    "pre-encoded and cached.",
    version="1.0.0",
    lifespan=lifespan,
)


def _wav_response(wav, sample_rate: int) -> Response:
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


async def _save_upload(upload: UploadFile, dest: Path | None = None) -> str:
    suffix = Path(upload.filename or "ref.wav").suffix or ".wav"
    if dest is not None:
        dest.write_bytes(await upload.read())
        return str(dest)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await upload.read())
        return tmp.name


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

@app.post("/tts", summary="TTS (optional zero-shot cloning)",
          responses={200: {"content": {"audio/wav": {}}}})
async def tts(
    text: Annotated[str, Form(description="Thai text to synthesize")],
    reference: Annotated[UploadFile | None,
                         File(description="Optional reference clip (3–10 s) to clone")] = None,
    cfg_value: Annotated[float, Form(description="CFG guidance scale (1.0–3.0)")] = 2.5,
    timesteps: Annotated[int, Form(description="Inference steps (4–30)")] = 10,
) -> Response:
    synth: Synthesizer = _state["synth"]
    if not text.strip():
        raise HTTPException(400, "text must be non-empty")
    ref_path = None
    try:
        if reference is not None:
            ref_path = await _save_upload(reference)
        wav = synth.synth(text, ref_audio=ref_path, cfg_value=cfg_value,
                          inference_timesteps=timesteps)
    finally:
        if ref_path:
            os.unlink(ref_path)
    return _wav_response(wav, synth.sample_rate)


@app.post("/tts/speaker/{speaker_id}", summary="TTS with a registered speaker",
          responses={200: {"content": {"audio/wav": {}}}})
async def tts_with_speaker(
    speaker_id: str,
    text: Annotated[str, Form(description="Thai text to synthesize")],
    cfg_value: Annotated[float, Form(description="CFG guidance scale (1.0–3.0)")] = 2.5,
    timesteps: Annotated[int, Form(description="Inference steps (4–30)")] = 10,
) -> Response:
    synth: Synthesizer = _state["synth"]
    if not text.strip():
        raise HTTPException(400, "text must be non-empty")
    cache = _state["voices"].get(speaker_id)
    if cache is None:
        raise HTTPException(404, f"speaker '{speaker_id}' not found (see GET /speakers)")
    wav = synth.synth_cached(text, cache, cfg_value=cfg_value, inference_timesteps=timesteps)
    return _wav_response(wav, synth.sample_rate)


# ---------------------------------------------------------------------------
# Speaker management
# ---------------------------------------------------------------------------

@app.post("/speakers", summary="Register a speaker (caches its reference encoding)")
async def register_speaker(
    speaker_id: Annotated[str, Form(description="Unique name/ID for this voice")],
    reference: Annotated[UploadFile, File(description="Reference clip (3–10 s)")],
) -> JSONResponse:
    synth: Synthesizer = _state["synth"]
    REF_DIR.mkdir(exist_ok=True)
    suffix = Path(reference.filename or "ref.wav").suffix or ".wav"
    ref_dest = REF_DIR / f"{speaker_id}{suffix}"
    await _save_upload(reference, dest=ref_dest)       # persist clip in ref/
    cache = synth.build_voice(str(ref_dest))
    synth.save_voice(cache, CACHE_DIR / f"{speaker_id}.pt")
    _state["voices"][speaker_id] = cache
    return JSONResponse({"speaker_id": speaker_id, "status": "registered"})


@app.get("/speakers", summary="List registered speakers")
def list_speakers() -> JSONResponse:
    return JSONResponse({"speakers": sorted(_state.get("voices", {}))})


@app.delete("/speakers/{speaker_id}", summary="Remove a registered speaker")
def delete_speaker(speaker_id: str) -> JSONResponse:
    if speaker_id not in _state.get("voices", {}):
        raise HTTPException(404, f"speaker '{speaker_id}' not found")
    _state["voices"].pop(speaker_id, None)
    (CACHE_DIR / f"{speaker_id}.pt").unlink(missing_ok=True)
    for ref_file in REF_DIR.glob(f"{speaker_id}.*"):
        ref_file.unlink(missing_ok=True)
    return JSONResponse({"speaker_id": speaker_id, "status": "deleted"})


@app.get("/health", summary="Health check")
def health() -> JSONResponse:
    synth: Synthesizer = _state.get("synth")
    return JSONResponse({
        "status": "ok" if synth else "loading",
        "base_model": _BASE_MODEL,
        "adapter": _state.get("adapter"),
        "sample_rate": getattr(synth, "sample_rate", None),
        "speakers": sorted(_state.get("voices", {})),
    })


_INDEX_HTML = """<!doctype html><meta charset="utf-8"><title>SiangTTS</title>
<h2>SiangTTS — Thai TTS + Voice Cloning</h2>
<h3>Text-to-speech (optional reference to clone)</h3>
<form action="/tts" method="post" enctype="multipart/form-data" target="_blank">
  <input name="text" size="55" placeholder="พิมพ์ข้อความภาษาไทย" required><br>
  reference (optional): <input type="file" name="reference" accept="audio/*"><button>Speak</button>
</form>
<h3>Register a reusable voice</h3>
<form action="/speakers" method="post" enctype="multipart/form-data" target="_blank">
  id: <input name="speaker_id" required>
  clip: <input type="file" name="reference" accept="audio/*" required><button>Register</button>
</form>
<p>Or drop clips in <code>ref/</code> and restart. Cached speakers:
<a href="/speakers">/speakers</a> · API docs: <a href="/docs">/docs</a></p>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML
