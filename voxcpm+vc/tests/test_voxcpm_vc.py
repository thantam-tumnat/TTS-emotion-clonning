"""The donor -> VoxCPM2 -> SeedVC pipeline.

What matters here is not audio quality (untestable without the GPU and the worker)
but the two decisions that decide whether the take carries the right emotion in the
right voice: the donor clip is handed to VoxCPM2 *with its transcript*, and the
result goes through SeedVC to the target speaker.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import voxcpm_vc_service as vc

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Emotion vocabulary
# --------------------------------------------------------------------------- #

def test_every_annotator_tone_maps_to_a_donor():
    """Ten tones, five recordings — no tone may fall through the map.

    A tone with no mapping is a 422 on ordinary auto-annotated text, which is what
    makes this worth asserting rather than trusting the dict to stay complete.
    """
    from app.models import Tone

    for tone in Tone:
        emotion = vc.voxcpm_vc_service.validate_emotion(tone.value)
        assert emotion in vc.SUPPORTED_EMOTIONS


def test_unknown_tone_is_rejected_not_flattened():
    """Falling back to neutral would return a flat take that looks like a success."""
    with pytest.raises(ValueError) as exc:
        vc.voxcpm_vc_service.validate_emotion("bewildered")
    assert "bewildered" in str(exc.value)


def test_missing_tone_is_neutral():
    assert vc.voxcpm_vc_service.validate_emotion(None) == "neutral"
    assert vc.voxcpm_vc_service.validate_emotion("  ") == "neutral"


# --------------------------------------------------------------------------- #
# Donor sets
# --------------------------------------------------------------------------- #

def test_donor_sets_are_one_actor_each():
    """Every listed set must have an actor id and its five clips.

    The whole premise is that the five emotions come from one person; a set built
    from several actors would change speaker between emotions, and SeedVC converts
    timbre only — the second actor's pacing and accent would still come through.
    """
    sets = vc.voxcpm_vc_service.list_donor_sets()
    assert sets, "no donor sets on disk"
    for s in sets:
        assert s["complete"], f"{s['id']} is missing emotions"
        assert s["actor_id"], f"{s['id']} is not a single-actor set"
        assert set(s["emotions"]) == set(vc.SUPPORTED_EMOTIONS)


def test_donor_clip_carries_a_transcript():
    """No transcript means timbre-only cloning, which carries no emotion at all."""
    chosen = vc.voxcpm_vc_service.resolve_donor_set(None, gender="female")
    wav, transcript = vc.voxcpm_vc_service.donor_clip(chosen, "angry")
    assert wav.is_file()
    assert transcript.strip()


def test_resolve_prefers_requested_gender():
    assert vc.voxcpm_vc_service.resolve_donor_set(None, gender="male").startswith("male")
    assert vc.voxcpm_vc_service.resolve_donor_set(None, gender="female").startswith("female")


def test_unknown_donor_set_is_an_error_not_a_silent_substitution():
    with pytest.raises(FileNotFoundError):
        vc.voxcpm_vc_service.resolve_donor_set("nobody_042")


# --------------------------------------------------------------------------- #
# Text handling
# --------------------------------------------------------------------------- #

def test_style_parenthetical_is_stripped():
    """In continuation mode the parenthetical is spoken, not obeyed."""
    assert vc.strip_instruction("(โกรธมาก) สวัสดีครับ") == "สวัสดีครับ"
    assert vc.strip_instruction("no instruction here") == "no instruction here"


# --------------------------------------------------------------------------- #
# Generation wiring
# --------------------------------------------------------------------------- #

class _RecordingEngine:
    """Stands in for the queue synthesizer, remembering how it was called."""

    sample_rate = 24000

    def __init__(self):
        self.voices = []          # (path, transcript) per build_voice
        self.jobs = []            # (handle, [texts]) per render_batch

    def build_voice(self, ref_audio_path, prompt_text=None):
        self.voices.append((ref_audio_path, prompt_text))
        return f"handle_{len(self.voices)}"

    def render_batch(self, texts, *, prompt_cache=None, cfg_value=2.5,
                     inference_timesteps=10, lora_mode="on", **kwargs):
        import numpy as np

        self.jobs.append((prompt_cache, list(texts)))
        return [np.zeros(self.sample_rate, dtype="float32") for _ in texts], self.sample_rate


@pytest.fixture
def engine(monkeypatch):
    eng = _RecordingEngine()
    monkeypatch.setattr(vc.VoxCPMVCService, "_synth", lambda self: eng)
    vc.voxcpm_vc_service._donor_handles.clear()
    return eng


def _target_clip(tmp_path):
    import numpy as np
    import soundfile as sf

    path = tmp_path / "target.wav"
    sf.write(str(path), np.zeros(16000, dtype="float32"), 16000)
    return path


def test_donor_is_cloned_with_its_transcript(engine, tmp_path):
    """The transcript is what selects VoxCPM2's continuation mode.

    Without it the model clones timbre only and reads the line neutrally, which is
    the exact failure this pipeline exists to avoid.
    """
    chosen = vc.voxcpm_vc_service.resolve_donor_set(None, gender="female")
    donor_wav, donor_txt = vc.voxcpm_vc_service.donor_clip(chosen, "angry")

    vc.voxcpm_vc_service.render_chunks(
        ["(angry) ทดสอบเสียงโกรธ"],
        ref_audio_bytes=_target_clip(tmp_path).read_bytes(),
        ref_filename="target.wav",
        tones=["angry"],
        donor_set=chosen,
    )

    assert len(engine.voices) == 1
    path, transcript = engine.voices[0]
    assert donor_wav.name in path
    assert transcript == donor_txt


def test_one_job_per_emotion_not_per_chunk(engine, tmp_path):
    """Chunks sharing an emotion share that emotion's prompt cache, in one job."""
    chosen = vc.voxcpm_vc_service.resolve_donor_set(None, gender="female")

    vc.voxcpm_vc_service.render_chunks(
        ["(angry) หนึ่ง", "(neutral) สอง", "(angry) สาม"],
        ref_audio_bytes=_target_clip(tmp_path).read_bytes(),
        ref_filename="target.wav",
        tones=["angry", "neutral", "angry"],
        donor_set=chosen,
    )

    assert len(engine.jobs) == 2                    # angry, neutral
    handles = {handle for handle, _ in engine.jobs}
    assert len(handles) == 2                        # a different donor each
    by_size = sorted(len(texts) for _handle, texts in engine.jobs)
    assert by_size == [1, 2]


def test_generated_text_has_no_instruction_left(engine, tmp_path):
    chosen = vc.voxcpm_vc_service.resolve_donor_set(None, gender="female")

    vc.voxcpm_vc_service.render_chunks(
        ["(sad, level 3) ฉันเสียใจมาก"],
        ref_audio_bytes=_target_clip(tmp_path).read_bytes(),
        ref_filename="target.wav",
        tones=["sad"],
        donor_set=chosen,
    )

    _handle, texts = engine.jobs[0]
    assert texts == ["ฉันเสียใจมาก"]


def test_output_is_at_the_conversion_rate(engine, tmp_path):
    """SeedVC decides the rate of the finished take, not VoxCPM2."""
    chunks, rate = vc.voxcpm_vc_service.render_chunks(
        ["ทดสอบ"],
        ref_audio_bytes=_target_clip(tmp_path).read_bytes(),
        ref_filename="target.wav",
        tones=["neutral"],
    )
    assert chunks
    # The suite's stub converter passes audio through, so the rate it reports is the
    # engine's; the point asserted is that the rate comes back from the converted
    # file rather than being assumed.
    assert rate == engine.sample_rate


def test_seedvc_down_fails_loudly(monkeypatch, engine, tmp_path):
    """A take in the donor's voice is a wrong-speaker result that sounds correct."""
    monkeypatch.setattr(vc.VoxCPMVCService, "seedvc_health", lambda self: None)

    with pytest.raises(vc.VoxCPMVCUnavailable) as exc:
        vc.voxcpm_vc_service.render_chunks(
            ["ทดสอบ"],
            ref_audio_bytes=_target_clip(tmp_path).read_bytes(),
            ref_filename="target.wav",
            tones=["neutral"],
        )
    assert "8022" in str(exc.value)


# --------------------------------------------------------------------------- #
# API surface
# --------------------------------------------------------------------------- #

def test_donors_endpoint_lists_sets_and_the_tone_map():
    res = client.get("/api/donors")
    assert res.status_code == 200
    body = res.json()
    assert body["emotions"] == list(vc.SUPPORTED_EMOTIONS)
    assert body["tone_map"]["excited"] == "happy"
    assert body["sets"]


def test_donor_audio_endpoint_streams_a_clip():
    chosen = vc.voxcpm_vc_service.resolve_donor_set(None, gender="female")
    res = client.get(f"/api/donors/{chosen}/angry/audio")
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"


def test_donor_audio_endpoint_404s_on_a_missing_emotion():
    chosen = vc.voxcpm_vc_service.resolve_donor_set(None, gender="female")
    assert client.get(f"/api/donors/{chosen}/bewildered/audio").status_code == 404


# --------------------------------------------------------------------------- #
# Pipeline Explorer + benchmark donor/pre-VC/F0 (merged from the :8012 workflow)
# --------------------------------------------------------------------------- #

def test_donor_sets_ui_shape_is_frontend_ready():
    """The picker needs `emotions` as a list of {id, transcript}, plus a name."""
    sets = vc.voxcpm_vc_service.donor_sets_ui()
    assert sets
    s = sets[0]
    assert s["id"] and s["name"]
    assert isinstance(s["emotions"], list)
    assert {"id", "transcript"} <= set(s["emotions"][0])


def test_pipeline_donor_sets_and_defaults_endpoints():
    ds = client.get("/api/pipeline/donor-sets")
    assert ds.status_code == 200 and ds.json()["sets"]
    d = client.get("/api/pipeline/defaults")
    assert d.status_code == 200
    assert "cfg_value" in d.json()["values"]


def test_pipeline_trace_keeps_every_stage(engine, tmp_path, monkeypatch):
    """One utterance donor -> VoxCPM2 -> SeedVC leaves three playable stages."""
    chosen = vc.voxcpm_vc_service.resolve_donor_set(None, gender="female")
    # A house target must exist; point the default target at a real clip.
    target = _target_clip(tmp_path)
    monkeypatch.setattr(vc.VoxCPMVCService, "_default_target", lambda self: target)

    res = client.post("/api/pipeline/trace", json={
        "donor_set": chosen, "emotion": "angry", "text": "ทดสอบเสียงโกรธ",
    })
    assert res.status_code == 200, res.text
    body = res.json()
    keys = {s["key"] for s in body["stages"]}
    assert keys == {"donor", "voxcpm", "vc"}
    for st in body["stages"]:
        got = client.get(st["url"])
        assert got.status_code == 200


def test_benchmark_presets_expose_donor_sets():
    body = client.get("/api/benchmark/presets").json()
    assert body["donor_sets"]
    assert body["default_params"]["gender"] == "female"


def test_run_take_pins_the_requested_donor_and_returns_pre_vc(engine, tmp_path, monkeypatch):
    """A benchmark take clones the requested donor and hands back the pre-SeedVC clip."""
    target = _target_clip(tmp_path)
    monkeypatch.setattr(vc.VoxCPMVCService, "_default_target", lambda self: target)
    chosen = vc.voxcpm_vc_service.resolve_donor_set(None, gender="female")

    init = client.post("/api/benchmark/session/init", json={
        "text": "ทดสอบ", "emotions": ["angry"], "repeats": 1,
        "gender": "female", "donor_set": chosen,
    }).json()
    sid = init["session_id"]

    res = client.post("/api/benchmark/run-take", json={
        "session_id": sid, "emotion": "angry", "take_idx": 1, "text": "ทดสอบเสียงโกรธ",
        "donor_set": chosen, "gender": "female", "f0_compare": True,
    })
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["pre_vc_url"]
    assert body["model_input"] and body["model_input"]["chunks"]
    # The donor named in the debug trace is the one requested.
    assert body["model_input"]["chunks"][0]["donor_set"] == chosen
    # F0-compare trio present (baseline / A / B).
    assert {m["id"] for m in body["f0_variants"]} == {"baseline", "A", "B"}
    assert client.get(body["pre_vc_url"]).status_code == 200
