"""The remote GPU path must keep one speaker across takes too.

Production points the studio at the shared GPU service (``VOXCPM_SERVICE_URL``),
so the benchmark's takes go through ``QueueSynthesizer``, not the in-process
model. The decision of *which speaker each take gets* is made here, in
``_voice_spec``: it turns a resolved handle / speaker name / clip into the
``voice`` block of the render job. If that block differs between takes -- or is
``None`` when it should carry a handle -- every take comes back a different
person, which is the ear-obvious failure being chased.

These tests mock the HTTP layer so they assert the voice-selection logic alone,
with no GPU service running.
"""

import json

import pytest

from app.services.queue_client import QueueSynthesizer


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class _FakeClient:
    """Records every request and answers from a canned route table."""

    def __init__(self, routes, log):
        self._routes = routes
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, path, json=None, **kwargs):
        self._log.append(("POST", path, json))
        return self._routes.get(("POST", path), _FakeResponse())


def _wire(monkeypatch, routes):
    """Give every QueueSynthesizer a fake client over ``routes``; return the log."""
    log = []
    monkeypatch.setattr(
        QueueSynthesizer, "_client", lambda self, timeout=None: _FakeClient(routes, log)
    )
    return log


def test_seed_handle_is_reused_verbatim_for_every_take(monkeypatch):
    """Once the seed handle is in hand, each take must send that same handle.

    The service layer mints the seed once and passes it down as ``prompt_cache``;
    the remote client must not re-resolve or drop it.
    """
    _wire(monkeypatch, {})
    remote = QueueSynthesizer("http://gpu")

    specs = [remote._voice_spec("_auto_seed", None, None) for _ in range(3)]

    assert specs == [{"handle": "_auto_seed"}] * 3


def test_pinned_speaker_resolves_to_a_stable_handle(monkeypatch):
    routes = {
        ("POST", "/v2/voices/resolve"): _FakeResponse(
            200, {"voice_handle": "h-spk1", "speaker_id": "spk1"}
        )
    }
    log = _wire(monkeypatch, routes)
    remote = QueueSynthesizer("http://gpu")

    specs = [remote._voice_spec(None, "spk1", None) for _ in range(3)]

    assert specs == [{"handle": "h-spk1"}] * 3
    # Every take resolved to the same handle the service holds; nothing about the
    # request changed between takes.
    resolves = [body for method, path, body in log if path == "/v2/voices/resolve"]
    assert all(b["speaker_id"] == "spk1" for b in resolves)


def test_unpinned_seed_endpoint_returns_one_shared_handle(monkeypatch):
    """Calling the seed endpoint repeatedly hands back the same speaker.

    This is what lets an unpinned run stay one person even across separate
    processes: the handle is service-side and stable.
    """
    routes = {
        ("POST", "/v2/voices/seed"): _FakeResponse(200, {"voice_handle": "_auto_seed"})
    }
    _wire(monkeypatch, routes)
    remote = QueueSynthesizer("http://gpu")

    handles = [remote.seed_voice() for _ in range(3)]

    assert handles == ["_auto_seed"] * 3


def test_seed_unavailable_is_surfaced_not_silently_unconditioned(monkeypatch):
    """If the seed cannot be minted, seed_voice() returns None.

    That None is the trigger for the ear-obvious bug: with no handle, the take is
    generated unconditioned and every take is a fresh speaker. The contract this
    test pins is that the failure is *reported* as None (and logged) rather than
    masquerading as a valid voice -- so the layer above can decide, and an
    operator can see the 503 in the logs.
    """
    routes = {
        ("POST", "/v2/voices/seed"): _FakeResponse(503, {"error": "seed unavailable"})
    }
    _wire(monkeypatch, routes)
    remote = QueueSynthesizer("http://gpu")

    assert remote.seed_voice() is None
