"""The webhook mirrors the caller's request onto the :8020 dashboard.

The pipeline substitutes a default for voice_id, sex and donor_set whenever the body
leaves them out, and the resulting take sounds exactly like one that was asked for.
These tests pin what the dashboard is told, since that row is the only place the
substitution is visible.
"""

import asyncio

import pytest

from app import webhook as wh


DONORS = [
    {"id": "f_one", "gender": "female", "complete": True},
    {"id": "m_one", "gender": "male", "complete": True},
]


class _FakeResponse:
    status_code = 202


class _FakeClient:
    """Captures what the webhook posts to the queue gateway."""

    posts: list = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        _FakeClient.posts.append((url, json))
        return _FakeResponse()

    async def patch(self, url, json=None):
        _FakeClient.posts.append((url, json))
        return _FakeResponse()


@pytest.fixture(autouse=True)
def stub_gateway_and_donors(monkeypatch):
    _FakeClient.posts = []
    monkeypatch.setattr(wh.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(wh.voxcpm_vc_service, "list_donor_sets", lambda: DONORS)
    monkeypatch.setattr(wh.settings, "webhook_default_voice", "", raising=False)
    monkeypatch.setattr(wh.settings, "default_gender", "female", raising=False)
    yield


def _submit(body_dict):
    """Run one accept and return (response, job_or_None, meta_payload_or_None)."""

    async def go():
        wh._state["jobs"] = {}
        wh._state["queue"] = asyncio.Queue()
        body = wh.WebhookBody(**body_dict)
        resp = await wh._accept(body, body_dict)
        jobs = list(wh._state["jobs"].values())
        meta = [p for (url, p) in _FakeClient.posts if url.endswith("/v2/jobs/external")]
        return resp, (jobs[0] if jobs else None), (meta[0] if meta else None)

    return asyncio.run(go())


def _accept(body_dict):
    """As above, for the cases that are expected to be accepted."""
    resp, job, meta = _submit(body_dict)
    assert resp.status_code == 200, resp.body
    return job, meta


def test_request_is_mirrored_verbatim_including_unknown_fields():
    body = {
        "prompt": "สวัสดีครับ",
        "voice_id": "abc-123",
        "sex": "male",
        "typo_field": "kept anyway",
    }
    job, meta = _accept(body)

    assert meta is not None, "the dashboard row must be registered at accept time"
    # Verbatim: a field the model drops is exactly what an operator needs to see.
    assert meta["request"]["received"] == body
    assert meta["request"]["received"]["typo_field"] == "kept anyway"
    assert job.request == body


def test_resolved_reports_the_caller_as_the_source_when_it_asked():
    job, meta = _accept({"prompt": "x", "voice_id": "abc-123", "sex": "male"})
    resolved = meta["request"]["resolved"]

    assert resolved["voice_id"] == "abc-123"
    assert resolved["voice_id_source"] == "request"
    assert resolved["sex"] == "male"
    assert resolved["sex_source"] == "request"
    # The row carries the target voice itself, so the card is not labelled with the
    # donor handle its per-emotion render jobs are conditioned on.
    assert meta["voice"] == {"speaker_id": "abc-123"}


def test_missing_voice_id_is_rejected_rather_than_defaulted():
    """SeedVC converts into a named target, so "no voice" is a broken request.

    It used to fall through to whatever clip sorted first in ref/ -- a take in a
    stranger's voice that is indistinguishable from a correct one.
    """
    resp, job, meta = _submit({"prompt": "x", "sex": "female"})

    assert resp.status_code == 400
    assert b"voice_id is required" in resp.body
    # Nothing queued, and no dashboard row for work that will never run.
    assert job is None
    assert meta is None


def test_blank_voice_id_is_rejected_too():
    resp, job, meta = _submit({"prompt": "x", "voice_id": "   "})

    assert resp.status_code == 400
    assert job is None


def test_configured_default_voice_accepts_the_request_and_is_named_as_such(monkeypatch):
    """WEBHOOK_DEFAULT_VOICE is the way to opt back into a house voice -- and the
    dashboard still has to say the caller never asked for it."""
    monkeypatch.setattr(wh.settings, "webhook_default_voice", "house_voice", raising=False)
    job, meta = _accept({"prompt": "x"})
    resolved = meta["request"]["resolved"]

    assert job.voice_id == "house_voice"
    assert meta["voice"] == {"speaker_id": "house_voice"}
    assert "WEBHOOK_DEFAULT_VOICE" in resolved["voice_id_source"]


def test_missing_sex_falls_back_to_the_configured_gender():
    job, meta = _accept({"prompt": "x", "voice_id": "abc"})
    resolved = meta["request"]["resolved"]

    assert resolved["sex"] == "female"
    assert "DEFAULT_GENDER" in resolved["sex_source"]


@pytest.mark.parametrize("raw", ["boy", "1", "MALE?", "unknown"])
def test_unrecognised_sex_is_flagged_not_silently_female(raw):
    job, meta = _accept({"prompt": "x", "voice_id": "abc", "sex": raw})
    resolved = meta["request"]["resolved"]

    assert resolved["sex_source"].startswith("unrecognised")
    assert resolved["sex"] in ("male", "female")


def test_recognised_sex_words_map_both_ways():
    assert wh._normalize_sex("m") == ("male", "request")
    assert wh._normalize_sex("Female") == ("female", "request")
    # Thai spellings arrive from the n8n flow as often as the English ones.
    assert wh._normalize_sex("ชาย") == ("male", "request")
    assert wh._normalize_sex("หญิง") == ("female", "request")


def test_pinned_donor_set_does_not_read_sex():
    job, meta = _accept({"prompt": "x", "voice_id": "abc", "sex": "male", "donor_set": "f_one"})
    resolved = meta["request"]["resolved"]

    assert resolved["donor_set"] == "f_one"
    assert resolved["donor_set_source"] == "request"
    assert resolved["sex"] is None
    assert "donor_set pinned" in resolved["sex_source"]


def test_random_donor_pick_is_named_as_random():
    job, meta = _accept({"prompt": "x", "voice_id": "abc", "sex": "male"})
    resolved = meta["request"]["resolved"]

    assert resolved["donor_set"] == "m_one"
    assert resolved["donor_set_source"].startswith("random male")


# --------------------------------------------------------------------------- #
# The receipt: what actually went into the model
# --------------------------------------------------------------------------- #

def test_engine_report_separates_the_target_from_the_donor():
    """`sex` never reaches the model -- it only picks the donor. The report has to
    show both halves, or a take in the wrong voice cannot be told apart from a take
    with the wrong donor."""
    job = wh.Job(job_id="j", queue_id="q", voice_id="voice-42", callback_url="")
    job.gender = "male"
    take = {
        "target_clip": "C:/temp/tts_jobs/voices/voice-42.mp3",
        "target_from": "voice-42",
        "donor_set": "male_003",
        "skip_neutral_vc": True,
        "seedvc": {"f0_mode": "B", "semi_tone_shift": -3},
    }
    groups = [
        {"emotion": "angry", "donor_set": "male_003", "donor_clip": "angry.wav",
         "donor_text": "โกรธ", "skip_vc": False, "pieces": ["a", "b"],
         "cfg_value": 2.5, "inference_timesteps": 10, "lora_mode": "on"},
        {"emotion": "neutral", "donor_set": "", "donor_clip": None, "donor_text": None,
         "skip_vc": True, "pieces": ["c"],
         "cfg_value": 2.5, "inference_timesteps": 10, "lora_mode": "on"},
    ]

    rep = wh._engine_report(job, take, groups)

    assert rep["voice_id"] == "voice-42"
    assert rep["target_clip"].endswith("voice-42.mp3")
    assert rep["sex"] == "male"
    assert rep["donor_set"] == "male_003"
    assert rep["seedvc"]["semi_tone_shift"] == -3
    # The emotional group is donor -> SeedVC -> target; the neutral one skipped the
    # conversion and was cloned from the target clip, so it has no donor to name.
    assert rep["groups"][0]["voice_converted"] is True
    assert rep["groups"][0]["donor_clip"] == "angry.wav"
    assert rep["groups"][0]["pieces"] == ["a", "b"]
    assert rep["groups"][1]["voice_converted"] is False
    assert rep["groups"][1]["donor_set"] is None


def test_engine_report_reaches_the_dashboard_after_the_take(monkeypatch, tmp_path):
    """A PATCH of its own, before upload and callback -- a run that dies at upload
    is exactly the one someone will want the receipt for."""
    monkeypatch.setattr(wh.settings, "webhook_use_llm", False, raising=False)
    # Keep the take's scratch dir out of the project tree.
    monkeypatch.setattr(wh, "WORK_ROOT", tmp_path / "work")

    def fake_synth(parts, **kw):
        kw["take_out"].update({"target_clip": "ref/house.wav", "target_from": "voice-9",
                               "donor_set": "female_002", "skip_neutral_vc": True,
                               "seedvc": {"f0_mode": "B"}})
        kw["debug_out"].append({"emotion": "neutral", "donor_set": "", "skip_vc": True,
                                "pieces": list(parts), "cfg_value": 2.5,
                                "inference_timesteps": 10, "lora_mode": "on"})
        return b"RIFFfake"

    monkeypatch.setattr(wh.voxcpm_vc_service, "synthesize_many", fake_synth)

    async def fake_upload(path):
        return "https://example.invalid/take.wav"

    async def fake_callback(url, job, *, error):
        return None

    monkeypatch.setattr(wh, "_upload", fake_upload)
    monkeypatch.setattr(wh, "_post_callback", fake_callback)

    async def go():
        wh._state["jobs"] = {}
        wh._state["queue"] = asyncio.Queue()
        body = {"prompt": "ทดสอบระบบ", "voice_id": "voice-9"}
        await wh._accept(wh.WebhookBody(**body), body)
        job = list(wh._state["jobs"].values())[0]
        await wh._run_job(job)
        return job

    job = asyncio.run(go())

    patches = [p for (url, p) in _FakeClient.posts if "engine" in (p or {})]
    assert patches, "the engine receipt was never sent to the gateway"
    assert patches[0]["engine"]["donor_set"] == "female_002"
    assert patches[0]["engine"]["voice_id"] == "voice-9"
    assert job.engine["groups"][0]["voice_converted"] is False
    assert job.status == "completed"
