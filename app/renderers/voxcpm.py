from typing import List, Optional
from app.models import Segment, Tone, RenderResponse
from app.renderers.base import BaseRenderer

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
    Example: '(Calm and soothing voice, speaking softly) หายใจเข้าลึกๆ ผ่อนคลาย'
    """

    def render(self, segments: List[Segment]) -> RenderResponse:
        if not segments:
            return RenderResponse(text="", prompt=None)

        out = []
        prev_tone: Optional[Tone] = None
        instructions_used = []

        for seg in segments:
            if seg.tone != prev_tone and seg.tone != Tone.NEUTRAL:
                instruction = format_voxcpm_instruction(seg.tone, seg.intensity)
                if instruction:
                    out.append(f"{instruction} ")
                    instructions_used.append(f"{seg.tone.value} (lvl {seg.intensity})")

            out.append(seg.text)
            prev_tone = seg.tone

        rendered_text = "".join(out).strip()
        prompt_summary = ", ".join(instructions_used) if instructions_used else "neutral"

        return RenderResponse(
            text=rendered_text,
            prompt=prompt_summary
        )
