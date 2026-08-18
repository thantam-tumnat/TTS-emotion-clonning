import re
from typing import List, Optional, Tuple
from app.models import Segment, Tone, RenderResponse, RenderedChunk
from app.renderers.base import BaseRenderer

# A style tag is ASCII-only and may use either bracket style -- the UI documents
# "[sad]" while VoxCPM2's own format is "(sad)". Requiring ASCII keeps Thai inside
# brackets treated as spoken content, not direction. An optional ":N" sets intensity.
STYLE_TAG_RE = re.compile(r"[\[(]\s*([A-Za-z][A-Za-z\s,.\-]*?)\s*(?::\s*([123]))?\s*[\])]")

_TONE_BY_NAME = {t.value: t for t in Tone}


def _tone_from_body(body: str) -> Optional[Tone]:
    """Best-effort tone for display, e.g. '(Sad and melancholic voice...)' -> SAD."""
    for word in re.findall(r"[a-z]+", body.lower())[:2]:
        if word in _TONE_BY_NAME:
            return _TONE_BY_NAME[word]
    return None


def resolve_style_tag(body: str, level: Optional[str] = None) -> Tuple[Optional[str], Tone, int]:
    """Turn a raw tag body into (instruction, tone, intensity).

    A bare tone name expands to this module's canonical instruction rather than
    being passed through: measured against a pinned speaker, "(sad)"/"(happy)" gave
    dF0 -15.0 where the full phrasing gave +28.6. Free-form English direction is
    kept verbatim so hand-written prompts still reach the model untouched.
    """
    body = body.strip()
    intensity = int(level) if level else 2
    tone = _TONE_BY_NAME.get(body.lower())
    if tone is not None:
        return format_voxcpm_instruction(tone, intensity), tone, intensity
    return f"({body})", _tone_from_body(body) or Tone.NEUTRAL, intensity


def _tagged_spans(text: str) -> List[Tuple[Optional[str], Tone, int, str]]:
    """Split text into (instruction, tone, intensity, body) runs at each style tag."""
    matches = list(STYLE_TAG_RE.finditer(text))
    if not matches:
        return []

    spans: List[Tuple[Optional[str], Tone, int, str]] = []
    lead = text[:matches[0].start()].strip()
    if lead:
        spans.append((None, Tone.NEUTRAL, 2, lead))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if not body:
            continue
        instruction, tone, intensity = resolve_style_tag(m.group(1), m.group(2))
        spans.append((instruction, tone, intensity, body))
    return spans


def split_style_chunks(text: str) -> List[str]:
    """Split hand-written text into chunks that each *lead* with a style instruction.

    VoxCPM2 only honours a parenthetical at position 0, so a tag typed mid-text would
    otherwise be read aloud. Splitting at every tag turns "[a]one[b]two" into two
    separately-synthesized chunks, which is what the user meant by writing it.

    Returns [] when the text carries no style tag, so callers can fall back to the
    LLM annotation path.
    """
    return [
        f"{instruction}{body}" if instruction else body
        for instruction, _tone, _lvl, body in _tagged_spans(text)
    ]


def parse_tagged_segments(text: str) -> List[Segment]:
    """Read hand-written tags as annotated segments, bypassing the LLM.

    Lets explicitly tagged input show up in the Segments panel as what the user
    actually wrote instead of coming back as one NEUTRAL blob with the raw markers
    still embedded in the spoken text.
    """
    return [
        Segment(text=body, tone=tone, intensity=intensity)
        for _instruction, tone, intensity, body in _tagged_spans(text)
    ]


# Wording is load-bearing and was chosen by measurement, not taste. Against a pinned
# speaker over 4 reps, these gave happy-vs-sad dF0 +28.6Hz / pitch-spread +42.4.
# Appending explicit prosody ("bright high pitch, lively quick pace") DILUTED it to
# +4.4 / -2.2, and bare "(happy)" / "(sad)" tags were worse still at -15.0 / -6.1.
# Re-measure before rewording.
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
