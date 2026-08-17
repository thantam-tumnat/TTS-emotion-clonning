import pytest

from app.services import siangtts_service as svc


@pytest.fixture(autouse=True)
def mock_synthesizer(monkeypatch):
    """Force the sine-tone mock for tests.

    The service refuses to fall back to the mock by default -- a silent fallback is
    what made a failed model load sound like a broken model instead of an error. Tests
    opt in explicitly rather than depending on an 8GB model being loadable.
    """
    monkeypatch.setattr(svc.settings, "siangtts_allow_mock", True, raising=False)
    monkeypatch.setattr(svc.siangtts_service, "_synthesizer", svc._MockSynthesizer())
    monkeypatch.setattr(svc.siangtts_service, "_using_mock", True)
    yield
    svc.siangtts_service._synthesizer = None
    svc.siangtts_service._using_mock = False
    svc.siangtts_service._voices.clear()
