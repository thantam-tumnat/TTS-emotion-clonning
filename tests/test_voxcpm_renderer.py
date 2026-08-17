import pytest
from app.models import Segment, Tone
from app.renderers.voxcpm import VoxCPMRenderer, format_voxcpm_instruction, split_style_chunks
from app.renderers import get_renderer


def test_voxcpm_renderer_all_neutral():
    renderer = VoxCPMRenderer()
    segments = [
        Segment(text="สวัสดีครับ ", tone=Tone.NEUTRAL, intensity=2),
        Segment(text="ยินดีต้อนรับครับ", tone=Tone.NEUTRAL, intensity=2)
    ]
    res = renderer.render(segments)
    assert res.text == "สวัสดีครับ ยินดีต้อนรับครับ"
    assert "(" not in res.text
    assert ")" not in res.text


def test_voxcpm_renderer_calm_prompt():
    renderer = VoxCPMRenderer()
    segments = [
        Segment(text="หายใจเข้าลึกๆ ผ่อนคลาย แล้วค่อยๆ ปล่อยวางทุกอย่างลงนะ", tone=Tone.CALM, intensity=2)
    ]
    res = renderer.render(segments)
    # VoxCPM2's documented format puts no space between ')' and the content.
    assert res.text.startswith("(Calm and soothing voice, speaking softly)หายใจเข้าลึกๆ")


@pytest.mark.parametrize("tone,intensity,expected_substr", [
    (Tone.CALM, 1, "Slightly calm"),
    (Tone.CALM, 2, "Calm and soothing"),
    (Tone.CALM, 3, "Deeply calm"),
    (Tone.SAD, 2, "Sad and melancholic"),
    (Tone.ANGRY, 2, "Angry, firm and aggressive"),
    (Tone.HAPPY, 2, "Happy and cheerful"),
    (Tone.EXCITED, 2, "Excited and energetic"),
    (Tone.NERVOUS, 2, "Nervous and trembling"),
    (Tone.SARCASTIC, 2, "Sarcastic and mocking"),
])
def test_voxcpm_renderer_tones_and_intensities(tone, intensity, expected_substr):
    instr = format_voxcpm_instruction(tone, intensity)
    assert instr is not None
    assert expected_substr.lower() in instr.lower()


def test_voxcpm_renderer_factory():
    renderer = get_renderer("voxcpm")
    assert isinstance(renderer, VoxCPMRenderer)
    renderer_siangtts = get_renderer("siangtts")
    assert isinstance(renderer_siangtts, VoxCPMRenderer)


def test_voxcpm_renderer_multi_segment_transitions():
    """Each tone run becomes its own chunk with the instruction leading it.

    VoxCPM2 only honours a style parenthetical at position 0; one appearing mid-text
    gets spoken aloud, so res.text must carry only the opening instruction.
    """
    renderer = VoxCPMRenderer()
    segments = [
        Segment(text="ขอโทษนะ ฉันไม่ได้ตั้งใจ ", tone=Tone.SAD, intensity=2),
        Segment(text="แต่เธอก็ไม่ฟังฉันเลย", tone=Tone.ANGRY, intensity=2)
    ]
    res = renderer.render(segments)

    assert len(res.chunks) == 2
    assert res.chunks[0].instruction == "(Sad and melancholic voice, slight sighs)"
    assert res.chunks[1].instruction == "(Angry, firm and aggressive tone)"
    for chunk in res.chunks:
        assert chunk.text.startswith(chunk.instruction)
        assert "(" not in chunk.body

    # Single-shot rendering: leading instruction only, nothing mid-utterance.
    assert res.text.startswith("(Sad and melancholic voice, slight sighs)")
    assert "(Angry, firm and aggressive tone)" not in res.text
    assert res.text.count("(") == 1


def test_voxcpm_renderer_merges_consecutive_same_tone():
    renderer = VoxCPMRenderer()
    segments = [
        Segment(text="ดีใจมากเลย ", tone=Tone.HAPPY, intensity=2),
        Segment(text="ขอบคุณนะ", tone=Tone.HAPPY, intensity=2),
    ]
    res = renderer.render(segments)
    assert len(res.chunks) == 1
    assert res.chunks[0].body == "ดีใจมากเลย ขอบคุณนะ"


# ---------------------------------------------------------------------------
# split_style_chunks -- hand-written inline style tags
# ---------------------------------------------------------------------------

def test_split_style_chunks_splits_on_mid_text_tag():
    """A tag typed mid-text becomes its own chunk instead of being spoken aloud."""
    text = "(Excited and energetic tone)ของดีมากครับ\n(sad)แต่ของหมดแล้ว"
    chunks = split_style_chunks(text)
    assert len(chunks) == 2
    assert chunks[0] == "(Excited and energetic tone)ของดีมากครับ"
    assert chunks[1] == "(sad)แต่ของหมดแล้ว"


def test_split_style_chunks_returns_empty_without_tags():
    """No tag means the caller should fall back to LLM annotation."""
    assert split_style_chunks("สวัสดีครับ วันนี้อากาศดีมาก") == []


def test_split_style_chunks_ignores_thai_parentheses():
    """Thai inside brackets is spoken content, not a style instruction."""
    assert split_style_chunks("ราคา (พิเศษ) วันนี้เท่านั้น") == []


def test_split_style_chunks_keeps_untagged_lead_text():
    chunks = split_style_chunks("เริ่มก่อน(Angry)แล้วโกรธ")
    assert chunks == ["เริ่มก่อน", "(Angry)แล้วโกรธ"]


def test_split_style_chunks_single_leading_tag_is_one_chunk():
    text = "(Calm and soothing voice)หายใจเข้าลึกๆ"
    assert split_style_chunks(text) == [text]


def test_split_style_chunks_drops_tag_with_empty_body():
    assert split_style_chunks("(Angry)   ") == []
