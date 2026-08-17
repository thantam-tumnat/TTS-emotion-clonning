from typing import Literal
from app.renderers.base import BaseRenderer
from app.renderers.elevenlabs import ElevenLabsRenderer
from app.renderers.gemini import GeminiRenderer
from app.renderers.voxcpm import VoxCPMRenderer


def get_renderer(engine: Literal["elevenlabs", "gemini", "voxcpm", "siangtts"]) -> BaseRenderer:
    """Factory to get the appropriate renderer instance."""
    if engine == "elevenlabs":
        return ElevenLabsRenderer()
    elif engine == "gemini":
        return GeminiRenderer()
    elif engine in ("voxcpm", "siangtts"):
        return VoxCPMRenderer()
    else:
        raise ValueError(f"Unknown engine: {engine}")


__all__ = ["BaseRenderer", "ElevenLabsRenderer", "GeminiRenderer", "VoxCPMRenderer", "get_renderer"]
