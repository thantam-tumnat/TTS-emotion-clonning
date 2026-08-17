from enum import Enum
from typing import Literal, Optional
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


class AnnotateResponse(BaseModel):
    original: str
    segments: list[Segment]
    model_used: str
    fallback: bool  # True = validate failed, fallback to all neutral


class RenderRequest(BaseModel):
    segments: list[Segment]
    engine: Literal["elevenlabs", "gemini"]


class RenderResponse(BaseModel):
    text: str  # text ready for TTS
    prompt: Optional[str] = None  # for engines using separate field (Gemini)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    guidance: Optional[str] = Field(default=None, description="Optional custom emotion/tone guidance")
    engine: Literal["elevenlabs", "gemini"] = "elevenlabs"


class SpeakResponse(BaseModel):
    engine: Literal["elevenlabs", "gemini"]
    text: str
    prompt: Optional[str] = None
    segments: list[Segment]
    model_used: str
    fallback: bool


class LLMClauseItem(BaseModel):
    i: int
    text: str


class LLMClauseLabel(BaseModel):
    i: int
    tone: Tone
    intensity: int = Field(default=2, ge=1, le=3)


class LLMAnnotationResult(BaseModel):
    labels: list[LLMClauseLabel]
