from typing import List, Optional
from app.models import Segment, Tone, RenderResponse
from app.renderers.base import BaseRenderer
from app.config import settings

ELEVENLABS_TAG_MAP = {
    Tone.NEUTRAL: None,
    Tone.SAD: "sad",
    Tone.HAPPY: "happily",
    Tone.ANGRY: "angry",
    Tone.EXCITED: "excited",
    Tone.CALM: "calm",
    Tone.NERVOUS: "nervous",
    Tone.SARCASTIC: "sarcastic",
    Tone.SCARED: "scared",
    Tone.TIRED: "tired",
}


def format_tag(tone: Tone, intensity: int) -> Optional[str]:
    """Format ElevenLabs audio tag with intensity modifier."""
    base_tag = ELEVENLABS_TAG_MAP.get(tone)
    if not base_tag or tone == Tone.NEUTRAL:
        return None

    if intensity == 1:
        return f"[slightly {base_tag}] "
    elif intensity == 3:
        return f"[very {base_tag}] "
    else:  # intensity == 2
        return f"[{base_tag}] "


class ElevenLabsRenderer(BaseRenderer):
    def __init__(self, reanchor_chars: Optional[int] = None):
        self.reanchor_chars = reanchor_chars if reanchor_chars is not None else settings.reanchor_chars

    def render(self, segments: List[Segment]) -> RenderResponse:
        if not segments:
            return RenderResponse(text="", prompt=None)

        out = []
        prev_tone: Optional[Tone] = None

        for seg in segments:
            # Tag only inserted when tone changes and tone is not neutral
            if seg.tone != prev_tone and seg.tone != Tone.NEUTRAL:
                tag_str = format_tag(seg.tone, seg.intensity)
                if tag_str:
                    out.append(tag_str)
            
            # Optional re-anchoring for very long segments if configured
            if self.reanchor_chars and len(seg.text) > self.reanchor_chars and seg.tone != Tone.NEUTRAL:
                # If segment is longer than reanchor_chars, insert periodic anchors
                tag_str = format_tag(seg.tone, seg.intensity)
                text_part = seg.text
                sub_parts = []
                while len(text_part) > self.reanchor_chars:
                    sub_parts.append(text_part[:self.reanchor_chars])
                    text_part = text_part[self.reanchor_chars:]
                sub_parts.append(text_part)
                out.append(f"{tag_str}".join(sub_parts))
            else:
                out.append(seg.text)

            prev_tone = seg.tone

        return RenderResponse(
            text="".join(out),
            prompt=None
        )
