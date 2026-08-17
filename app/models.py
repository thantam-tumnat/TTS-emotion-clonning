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


class Segment(BaseModel):
    text: str
    tone: Tone
    intensity: int = Field(default=2, ge=1, le=3)


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


class RenderRequest(BaseModel):
    segments: list[Segment]
    engine: Literal["elevenlabs", "gemini", "voxcpm", "siangtts"]


class RenderedChunk(BaseModel):
    """One synthesis unit: a style instruction plus the text it applies to."""
    text: str  # instruction + body, ready to hand to the engine
    instruction: Optional[str] = None
    body: str = ""


class RenderResponse(BaseModel):
    text: str  # text ready for TTS / instruction prompt
    prompt: Optional[str] = None  # for engines using separate field (Gemini/VoxCPM summary)
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


class LLMClauseItem(BaseModel):
    i: int
    text: str


class LLMClauseLabel(BaseModel):
    i: int
    tone: Tone
    intensity: int = Field(default=2, ge=1, le=3)


class LLMAnnotationResult(BaseModel):
    labels: list[LLMClauseLabel]
