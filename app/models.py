from enum import Enum
from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class Tone(str, Enum):
    NEUTRAL = "neutral"
    SAD = "sad"
    HAPPY = "happy"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    NERVOUS = "nervous"
    SARCASTIC = "sarcastic"
    SCARED = "scared"
    TIRED = "tired"


class Segment(BaseModel):
    text: str
    tone: Tone
    intensity: int = Field(default=2, ge=1, le=3)
    # Free-form style word as written, e.g. "appalled". `tone` stays the coarse
    # family used for colour and for the ElevenLabs/Gemini renderers.
    style: Optional[str] = None
    # The source put a line break before this segment's tag, which earns a longer
    # pause than an inline tone change. Kept on the segment so the short script form
    # can round-trip it.
    break_before: bool = False


class AnnotateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    guidance: Optional[str] = Field(default=None, description="Optional custom emotion/tone guidance")
    model: Optional[str] = Field(default=None, description="Optional specific LLM model to use")


class AnnotateResponse(BaseModel):
    original: str
    segments: list[Segment]
    model_used: str
    fallback: bool  # True = validate failed, fallback to all neutral
    error: Optional[str] = None
    error_detail: Optional[str] = None
    attempts: Optional[list[dict]] = None
    warnings: List[str] = Field(default_factory=list)


class RenderRequest(BaseModel):
    segments: list[Segment]
    engine: Literal["elevenlabs", "gemini", "voxcpm", "siangtts"]


class RenderedChunk(BaseModel):
    """One synthesis unit: a style instruction plus the text it applies to."""
    text: str  # instruction + body, ready to hand to the engine
    instruction: Optional[str] = None
    body: str = ""
    # Carried through to the audio assembler, which sets this chunk's loudness and
    # the pause in front of it from the tone. Without it every chunk lands at
    # whatever level the model happened to pick.
    tone: Optional[str] = None
    # True when the source had a line break before this chunk, which earns a longer
    # pause than an inline tone change does.
    break_before: bool = False


class RenderResponse(BaseModel):
    text: str  # text ready for TTS / instruction prompt
    prompt: Optional[str] = None  # for engines using separate field (Gemini/VoxCPM summary)
    # The short, editable form: "[sad] ... [happy] ...". This is what the studio puts
    # in the editable box, because `text` is a single-shot rendering that carries only
    # the FIRST instruction -- sending that back collapsed a multi-emotion script into
    # one tone.
    script: Optional[str] = None
    # Per-segment units. VoxCPM2 only honours a style parenthetical at the start of
    # the text it is given, so multi-tone input must be synthesized chunk by chunk.
    chunks: List[RenderedChunk] = Field(default_factory=list)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    guidance: Optional[str] = Field(default=None, description="Optional custom emotion/tone guidance")
    engine: Literal["elevenlabs", "gemini", "voxcpm", "siangtts"] = "voxcpm"
    model: Optional[str] = Field(default=None, description="Optional specific LLM model to use")


class SpeakResponse(BaseModel):
    engine: Literal["elevenlabs", "gemini", "voxcpm", "siangtts"]
    text: str
    prompt: Optional[str] = None
    segments: list[Segment]
    model_used: str
    fallback: bool
    error: Optional[str] = None
    error_detail: Optional[str] = None
    attempts: Optional[list[dict]] = None
    chunks: Optional[list[RenderedChunk]] = None
    script: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class PronunciationResponse(BaseModel):
    """The custom pronunciation overrides applied just before synthesis."""
    entries: dict[str, str]
    path: str


class PronunciationUpdateRequest(BaseModel):
    # Replaces the whole dictionary. Keys are matched on word boundaries, so
    # respelling "ไฟล์" leaves "โปรไฟล์" -- a genuinely different vowel -- alone.
    entries: dict[str, str]


class SpeakerInfo(BaseModel):
    id: str
    name: str
    filename: str
    cached: bool


class SpeakerListResponse(BaseModel):
    speakers: List[SpeakerInfo]


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    speaker_id: Optional[str] = None
    guidance: Optional[str] = None
    engine: Literal["voxcpm", "siangtts", "elevenlabs", "gemini"] = "voxcpm"
    model: Optional[str] = Field(default=None, description="Optional specific LLM model to use")
    cfg_value: float = Field(default=2.5, ge=1.0, le=10.0)
    inference_timesteps: int = Field(default=10, ge=4, le=50)
    auto_annotate: bool = True
    lora_mode: Optional[Literal["on", "off", "legacy"]] = Field(
        default="on", description="LoRA mode: 'on' (Thai optimized), 'off' (Base model), or 'legacy' (shipped 2.0/2.0)"
    )


class LLMClauseItem(BaseModel):
    i: int
    text: str


class LLMClauseLabel(BaseModel):
    i: int
    tone: Tone
    intensity: int = Field(default=2, ge=1, le=3)


class LLMAnnotationResult(BaseModel):
    labels: list[LLMClauseLabel]
