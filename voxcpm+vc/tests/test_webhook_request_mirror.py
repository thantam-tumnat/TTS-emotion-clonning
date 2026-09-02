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


@pytest.fixture(autouse=True)
def stub_gateway_and_donors(monkeypatch):
    _FakeClient.posts = []
    monkeypatch.setattr(wh.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(wh.voxcpm_vc_service, "list_donor_sets", lambda: DONORS)
    monkeypatch.setattr(wh.settings, "webhook_default_voice", "", raising=False)
    monkeypatch.setattr(wh.settings, "default_gender", "female", raising=False)
    yield


def _accept(body_dict):
    """Run one accept and return (job, meta_payload)."""

    async def go():
        wh._state["jobs"] = {}
        wh._state["queue"] = asyncio.Queue()
        body = wh.WebhookBody(**body_dict)
        await wh._accept(body, body_dict)
        job = list(wh._state["jobs"].values())[0]
        meta = [p for (url, p) in _FakeClient.posts if url.endswith("/v2/jobs/external")]
        return job, (meta[0] if meta else None)

    return asyncio.run(go())


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


def test_missing_voice_id_is_reported_as_a_substitution():
    job, meta = _accept({"prompt": "x", "sex": "female"})
    resolved = meta["request"]["resolved"]

    assert resolved["voice_id"] is None
    assert resolved["voice_id_source"].startswith("not sent")
    assert meta["voice"] is None


def test_configured_default_voice_is_named_as_such(monkeypatch):
    monkeypatch.setattr(wh.settings, "webhook_default_voice", "house_voice", raising=False)
    job, meta = _accept({"prompt": "x"})
    resolved = meta["request"]["resolved"]

    assert job.voice_id == "house_voice"
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
