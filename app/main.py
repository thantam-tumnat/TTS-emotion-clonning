import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    AnnotateRequest,
    AnnotateResponse,
    RenderRequest,
    RenderResponse,
    SpeakRequest,
    SpeakResponse,
    SpeakerListResponse,
    SynthesizeRequest,
)
from app.segmenter import segment_text
from app.annotator import annotator
from app.renderers import get_renderer
from app.services.siangtts_service import siangtts_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize speaker audio clips & cache on startup
    try:
        siangtts_service.init_speakers()
    except Exception as e:
        print(f"[Startup] Warning: Could not initialize speaker cache: {e}")
    yield


app = FastAPI(
    title="Thai TTS Tone Annotation & Voice Cloning Studio",
    description="LLM-based emotional tone annotation and SiangTTS (VoxCPM2) Voice Cloning pipeline for Thai speech.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder if exists
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def root_ui():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Thai TTS Tone Annotation & Voice Cloning API is running. Visit /docs for API documentation."}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "thai-tts-tone-annotation",
        "version": "2.0.0",
        "speakers_count": len(siangtts_service.list_speakers()),
    }


@app.post("/annotate", response_model=AnnotateResponse)
def annotate_endpoint(req: AnnotateRequest):
    """
    Segment input Thai text into clauses, query LLM for tone & intensity labels,
    validate, merge, and return annotated segments.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    clauses = segment_text(text)
    response = annotator.annotate(original_text=text, clauses=clauses, guidance=req.guidance)
    return response


@app.post("/render", response_model=RenderResponse)
def render_endpoint(req: RenderRequest):
    """
    Render annotated segments for a specific TTS engine (voxcpm/siangtts, elevenlabs, gemini).
    """
    try:
        renderer = get_renderer(req.engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return renderer.render(req.segments)


@app.post("/speak", response_model=SpeakResponse)
def speak_endpoint(req: SpeakRequest):
    """
    End-to-end preparation pipeline:
    Receives raw text -> annotates tone -> renders engine payload.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    clauses = segment_text(text)
    annotated = annotator.annotate(original_text=text, clauses=clauses, guidance=req.guidance)
    
    renderer = get_renderer(req.engine)
    rendered = renderer.render(annotated.segments)

    return SpeakResponse(
        engine=req.engine,
        text=rendered.text,
        prompt=rendered.prompt,
        segments=annotated.segments,
        model_used=annotated.model_used,
        fallback=annotated.fallback
    )


# ---------------------------------------------------------------------------
# Voice Cloning & Speaker Management Endpoints
# ---------------------------------------------------------------------------

@app.get("/speakers", response_model=SpeakerListResponse)
def list_speakers_endpoint():
    """List all registered voice cloning profiles in ref/ directory."""
    speakers = siangtts_service.list_speakers()
    return SpeakerListResponse(speakers=speakers)


@app.post("/speakers")
async def register_speaker_endpoint(
    file: UploadFile = File(...),
    speaker_id: Optional[str] = Form(None),
):
    """Upload a reference audio clip (.wav/.mp3) to register a new voice profile."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    sid = speaker_id.strip() if speaker_id and speaker_id.strip() else os.path.splitext(file.filename)[0]
    result = siangtts_service.register_speaker(
        speaker_id=sid,
        audio_bytes=content,
        filename=file.filename or "voice.wav",
    )
    return result


@app.delete("/speakers/{speaker_id}")
def delete_speaker_endpoint(speaker_id: str):
    """Remove a voice cloning profile."""
    deleted = siangtts_service.delete_speaker(speaker_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"deleted": True, "speaker_id": speaker_id}


# ---------------------------------------------------------------------------
# Audio Synthesis Endpoint
# ---------------------------------------------------------------------------

@app.post("/synthesize")
async def synthesize_endpoint(req: SynthesizeRequest):
    """
    Synthesizes speech using SiangTTS (VoxCPM2) or mapped engines.
    If auto_annotate is True, extracts emotions and injects control instructions automatically.
    Returns 48kHz WAV audio stream.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # If text has brackets or auto-annotate requested, prepare rendered text
    if req.auto_annotate and not text.startswith("("):
        clauses = segment_text(text)
        annotated = annotator.annotate(original_text=text, clauses=clauses, guidance=req.guidance)
        renderer = get_renderer(req.engine if req.engine in ("voxcpm", "siangtts") else "voxcpm")
        rendered = renderer.render(annotated.segments)
        prompt_text = rendered.text
    else:
        prompt_text = text

    # Perform synthesis via SiangTTSService
    try:
        wav_bytes = siangtts_service.synthesize(
            text=prompt_text,
            speaker_id=req.speaker_id,
            cfg_value=req.cfg_value,
            inference_timesteps=req.inference_timesteps,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="synthesized.wav"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.post("/synthesize/upload")
async def synthesize_with_upload_endpoint(
    text: str = Form(...),
    file: Optional[UploadFile] = File(None),
    guidance: Optional[str] = Form(None),
    cfg_value: float = Form(2.5),
    inference_timesteps: int = Form(10),
    auto_annotate: bool = Form(True),
):
    """
    Synthesizes speech with a direct one-off uploaded reference audio file.
    Returns 48kHz WAV audio stream.
    """
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if auto_annotate and not clean_text.startswith("("):
        clauses = segment_text(clean_text)
        annotated = annotator.annotate(original_text=clean_text, clauses=clauses, guidance=guidance)
        renderer = get_renderer("voxcpm")
        rendered = renderer.render(annotated.segments)
        prompt_text = rendered.text
    else:
        prompt_text = clean_text

    audio_bytes = await file.read() if file else None
    filename = file.filename if file else None

    try:
        wav_bytes = siangtts_service.synthesize(
            text=prompt_text,
            ref_audio_bytes=audio_bytes,
            ref_filename=filename,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="synthesized_custom.wav"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")
