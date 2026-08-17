from typing import Literal
from app.renderers.base import BaseRenderer
from app.renderers.elevenlabs import ElevenLabsRenderer
from app.renderers.gemini import GeminiRenderer


def get_renderer(engine: Literal["elevenlabs", "gemini"]) -> BaseRenderer:
    """Factory to get the appropriate renderer instance."""
    if engine == "elevenlabs":
        return ElevenLabsRenderer()
    elif engine == "gemini":
        return GeminiRenderer()
    else:
        raise ValueError(f"Unknown engine: {engine}")


__all__ = ["BaseRenderer", "ElevenLabsRenderer", "GeminiRenderer", "get_renderer"]
