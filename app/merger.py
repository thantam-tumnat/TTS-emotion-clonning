from typing import List
from app.models import Segment, Tone


def merge_segments(segments: List[Segment], max_segments: int = 20) -> List[Segment]:
    """
    Merge consecutive segments that have the same tone.
    The resulting intensity is the maximum intensity of the group.
    
    If the number of merged segments exceeds max_segments,
    further merge adjacent segments until <= max_segments.
    """
    if not segments:
        return []

    # Step 1: Merge consecutive segments with identical tone
    merged: List[Segment] = []
    current_text = segments[0].text
    current_tone = segments[0].tone
    current_intensity = segments[0].intensity

    for seg in segments[1:]:
        if seg.tone == current_tone:
            current_text += seg.text
            current_intensity = max(current_intensity, seg.intensity)
        else:
            merged.append(Segment(
                text=current_text,
                tone=current_tone,
                intensity=current_intensity
            ))
            current_text = seg.text
            current_tone = seg.tone
            current_intensity = seg.intensity

    merged.append(Segment(
        text=current_text,
        tone=current_tone,
        intensity=current_intensity
    ))

    # Step 2: Enforce max_segments limit if needed
    if max_segments > 0 and len(merged) > max_segments:
        # Merge adjacent segments until len(merged) <= max_segments
        while len(merged) > max_segments:
            # Find the shortest segment pair to merge
            best_idx = 0
            min_combined_len = float("inf")
            for i in range(len(merged) - 1):
                combined_len = len(merged[i].text) + len(merged[i + 1].text)
                if combined_len < min_combined_len:
                    min_combined_len = combined_len
                    best_idx = i
            
            first = merged[best_idx]
            second = merged[best_idx + 1]
            
            # Combine them (keep the tone of the more intense or first, max intensity)
            chosen_tone = first.tone if first.intensity >= second.intensity else second.tone
            combined_segment = Segment(
                text=first.text + second.text,
                tone=chosen_tone,
                intensity=max(first.intensity, second.intensity)
            )
            
            # Replace the two with the merged one
            merged = merged[:best_idx] + [combined_segment] + merged[best_idx + 2:]
            
            # Re-merge identical consecutive tones if created
            cleaned: List[Segment] = []
            if merged:
                c_text = merged[0].text
                c_tone = merged[0].tone
                c_int = merged[0].intensity
                for seg in merged[1:]:
                    if seg.tone == c_tone:
                        c_text += seg.text
                        c_int = max(c_int, seg.intensity)
                    else:
                        cleaned.append(Segment(text=c_text, tone=c_tone, intensity=c_int))
                        c_text = seg.text
                        c_tone = seg.tone
                        c_int = seg.intensity
                cleaned.append(Segment(text=c_text, tone=c_tone, intensity=c_int))
                merged = cleaned

    return merged
