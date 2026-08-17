import re
from typing import List, Optional
from app.models import Segment, Tone, RenderResponse, RenderedChunk
from app.renderers.base import BaseRenderer

# A style parenthetical is ASCII-only -- "(sad)", "(Excited and energetic tone)".
# Requiring that keeps Thai text in brackets treated as spoken content, not a tag.
STYLE_TAG_RE = re.compile(r"\([a-zA-Z][a-zA-Z\s,.\-]*\)")


def split_style_chunks(text: str) -> List[str]:
    """Split hand-written text into chunks that each *lead* with a style parenthetical.

    VoxCPM2 only honours a parenthetical at position 0, so a tag typed mid-text would
    otherwise be read aloud. Splitting at every tag turns "(a)one(b)two" into two
    separately-synthesized chunks, which is what the user meant by writing it.

    Returns [] when the text carries no style tag, so callers can fall back to the
    LLM annotation path.
    """
    matches = list(STYLE_TAG_RE.finditer(text))
    if not matches:
        return []

    chunks: List[str] = []
    # Text before the first tag has no instruction of its own.
    lead = text[:matches[0].start()].strip()
    if lead:
        chunks.append(lead)

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            chunks.append(f"{match.group(0)}{body}")

    return chunks

VOXCPM_INSTRUCTION_MAP = {
    Tone.NEUTRAL: {
        1: None,
        2: None,
        3: None,
    },
    Tone.CALM: {
        1: "(Slightly calm and gentle tone)",
        2: "(Calm and soothing voice, speaking softly)",
        3: "(Deeply calm and relaxing tone, very slow pace)",
    },
    Tone.SAD: {
        1: "(Slightly sad tone)",
        2: "(Sad and melancholic voice, slight sighs)",
        3: "(Deeply sorrowful and crying voice, trembling)",
    },
    Tone.HAPPY: {
        1: "(Pleasant tone, slight smile)",
        2: "(Happy and cheerful voice, smiling while speaking)",
        3: "(Extremely joyful and laughing voice)",
    },
    Tone.ANGRY: {
        1: "(Annoyed and sharp voice)",
        2: "(Angry, firm and aggressive tone)",
        3: "(Furious and yelling tone, very loud and harsh)",
    },
    Tone.EXCITED: {
        1: "(Eager voice)",
        2: "(Excited and energetic tone)",
        3: "(Thrilled and loud energetic voice)",
    },
    Tone.NERVOUS: {
        1: "(Slightly hesitant voice)",
        2: "(Nervous and trembling voice, hesitant)",
        3: "(Extremely anxious and panicking voice)",
    },
    Tone.SARCASTIC: {
        1: "(Slightly sarcastic tone)",
        2: "(Sarcastic and mocking tone)",
        3: "(Heavy sarcastic and cynical tone)",
    },
}


def format_voxcpm_instruction(tone: Tone, intensity: int = 2) -> Optional[str]:
    """Format VoxCPM emotion control instruction."""
    intensity = max(1, min(3, intensity))
    tone_map = VOXCPM_INSTRUCTION_MAP.get(tone, {})
    return tone_map.get(intensity)


class VoxCPMRenderer(BaseRenderer):
    """
    Renders segments for VoxCPM2 / SiangTTS with natural-language control instructions.
    Example: '(Calm and soothing voice, speaking softly)หายใจเข้าลึกๆ ผ่อนคลาย'

    VoxCPM2 reads a style parenthetical only when it leads the text it is given; one
    appearing mid-text is spoken aloud instead. So each run of same-tone segments
    becomes its own chunk with the instruction at position 0, and the caller
    synthesizes the chunks separately. ``text`` remains a single-shot rendering that
    only carries the opening instruction.
    """

    def render(self, segments: List[Segment]) -> RenderResponse:
        if not segments:
            return RenderResponse(text="", prompt=None, chunks=[])

        chunks: List[RenderedChunk] = []
        instructions_used: List[str] = []
        prev_tone: Optional[Tone] = None

        for seg in segments:
            # Same tone as the previous segment: extend it rather than re-stating.
            if seg.tone == prev_tone and chunks:
                chunks[-1].body += seg.text
                chunks[-1].text += seg.text
                continue

            instruction = (
                format_voxcpm_instruction(seg.tone, seg.intensity)
                if seg.tone != Tone.NEUTRAL
                else None
            )
            if instruction:
                instructions_used.append(f"{seg.tone.value} (lvl {seg.intensity})")

            # No space after ')' -- that is the documented VoxCPM2 format.
            chunks.append(
                RenderedChunk(
                    text=f"{instruction}{seg.text}" if instruction else seg.text,
                    instruction=instruction,
                    body=seg.text,
                )
            )
            prev_tone = seg.tone

        for c in chunks:
            c.body = c.body.strip()
            c.text = c.text.strip()
        chunks = [c for c in chunks if c.body]

        # Single-shot form: the leading instruction applies, the rest is plain body
        # text so no parenthetical ever lands mid-utterance.
        lead = chunks[0].instruction if chunks else None
        joined = "".join(c.body for c in chunks)
        rendered_text = f"{lead}{joined}" if lead else joined

        return RenderResponse(
            text=rendered_text.strip(),
            prompt=", ".join(instructions_used) if instructions_used else "neutral",
            chunks=chunks,
        )
