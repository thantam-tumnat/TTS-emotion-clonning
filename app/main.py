import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.models import (
    AnnotateRequest,
    AnnotateResponse,
    RenderRequest,
    RenderResponse,
    SpeakRequest,
    SpeakResponse,
)
from app.segmenter import segment_text
from app.annotator import annotator
from app.renderers import get_renderer

app = FastAPI(
    title="Thai TTS Tone Annotation Layer",
    description="LLM-based emotional tone and intensity annotation for Thai Text-to-Speech synthesis.",
    version="1.0.0"
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
    return {"message": "Thai TTS Tone Annotation API is running. Visit /docs for API documentation."}



@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "thai-tts-tone-annotation",
        "version": "1.0.0"
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
    Render annotated segments for a specific TTS engine (elevenlabs or gemini).
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
    Ready to pass directly to TTS engine adapters in future phases.
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
