from typing import List, Tuple, Any
from app.models import Segment, Tone
from app.merger import merge_segments


class ValidationError(Exception):
    """Exception raised when validation fails and escalation is required."""
    pass


def validate_and_build_segments(
    original_text: str,
    clauses: List[str],
    raw_labels: List[Any],
    max_segments: int = 20
) -> List[Segment]:
    """
    Validates LLM returned labels according to SPEC:
    1. index check (complete and exact 0..N-1) -> raises ValidationError to escalate if invalid
    2. tone in enum -> fallback invalid to Tone.NEUTRAL
    3. intensity in {1, 2, 3} -> fallback invalid to 2
    4. "".join(seg.text) == original_text -> raises ValidationError if mismatch
    5. remove empty text segments
    6. merge segments with identical adjacent tone, cap count <= max_segments
    """
    num_clauses = len(clauses)

    if not isinstance(raw_labels, list):
        raise ValidationError("LLM output is not a list")

    # Build index map
    label_map = {}
    for item in raw_labels:
        if isinstance(item, dict):
            i = item.get("i")
            tone_val = item.get("tone")
            intensity_val = item.get("intensity")
        elif hasattr(item, "i"):
            i = getattr(item, "i")
            tone_val = getattr(item, "tone")
            intensity_val = getattr(item, "intensity")
        else:
            raise ValidationError("Invalid label item format")

        if i is None or not isinstance(i, int) or i < 0 or i >= num_clauses:
            raise ValidationError(f"Label index {i} out of bounds (0..{num_clauses - 1})")

        if i in label_map:
            raise ValidationError(f"Duplicate label index: {i}")

        label_map[i] = (tone_val, intensity_val)

    # Check completeness (all indices 0..N-1 must be present)
    if len(label_map) != num_clauses or any(i not in label_map for i in range(num_clauses)):
        raise ValidationError(f"Missing indices from LLM. Expected 0..{num_clauses - 1}, got {list(label_map.keys())}")

    raw_segments: List[Segment] = []
    for i in range(num_clauses):
        tone_val, intensity_val = label_map[i]

        # 2. Tone validation (fallback invalid to NEUTRAL)
        try:
            if isinstance(tone_val, Tone):
                tone = tone_val
            elif isinstance(tone_val, str):
                tone = Tone(tone_val.lower().strip())
            else:
                tone = Tone.NEUTRAL
        except (ValueError, AttributeError):
            tone = Tone.NEUTRAL

        # 3. Intensity validation (fallback invalid to 2)
        try:
            intensity = int(intensity_val)
            if intensity not in (1, 2, 3):
                intensity = 2
        except (ValueError, TypeError):
            intensity = 2

        raw_segments.append(Segment(
            text=clauses[i],
            tone=tone,
            intensity=intensity
        ))

    # 4. Text reconstruction invariant check
    reconstructed = "".join(seg.text for seg in raw_segments)
    if reconstructed != original_text:
        raise ValidationError(f"Reconstructed text mismatch: '{reconstructed}' != '{original_text}'")

    # 5. Remove empty segments
    non_empty_segments = [seg for seg in raw_segments if seg.text]
    if not non_empty_segments and original_text:
        non_empty_segments = [Segment(text=original_text, tone=Tone.NEUTRAL, intensity=2)]

    # 6. Merge adjacent segments of same tone & enforce max_segments
    final_segments = merge_segments(non_empty_segments, max_segments=max_segments)

    return final_segments
