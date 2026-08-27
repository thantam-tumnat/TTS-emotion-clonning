import pytest

from app.services import siangtts_service as svc


@pytest.fixture(autouse=True)
def mock_synthesizer(monkeypatch, tmp_path):
    """Force the sine-tone mock for tests, against throwaway cache directories.

    The service refuses to fall back to the mock by default -- a silent fallback is
    what made a failed model load sound like a broken model instead of an error. Tests
    opt in explicitly rather than depending on an 8GB model being loadable.

    The cache directory is redirected because the service persists its auto seed
    voice there. Left pointing at the real voice_cache/, a test run dropped a
    mock-built seed into the project, and the next production run loaded it.
    """
    cache_dir = tmp_path / "voice_cache"
    cache_dir.mkdir()

    # No test may reach for the shared GPU service. Left at its default the suite
    # would depend on whether a machine happens to have one running on :8020, and
    # the tests that exercise the in-process load path would never reach it.
    monkeypatch.setattr(svc.settings, "voxcpm_service_url", "", raising=False)
    monkeypatch.setattr(svc.settings, "siangtts_allow_mock", True, raising=False)
    monkeypatch.setattr(svc.settings, "siangtts_cache_dir", str(cache_dir), raising=False)
    monkeypatch.setattr(svc.siangtts_service, "cache_dir", cache_dir, raising=False)
    monkeypatch.setattr(svc.siangtts_service, "_synthesizer", svc._MockSynthesizer())
    monkeypatch.setattr(svc.siangtts_service, "_using_mock", True)
    yield
    svc.siangtts_service._synthesizer = None
    svc.siangtts_service._using_mock = False
    svc.siangtts_service._voices.clear()
    # Built at most once per process, so it has to be cleared between tests or the
    # next test sees a seed it never generated.
    svc.siangtts_service._seed_voice = None
    svc.siangtts_service._seed_voice_failed = False


@pytest.fixture(autouse=True)
def stub_seedvc(monkeypatch, tmp_path):
    """Stand in for the SeedVC worker.

    The real one is a separate process holding ~2 GB of weights in its own venv, so
    the suite cannot start it. The stub copies the generated audio through unchanged,
    which is the right shape for every test here: they check routing, assembly and
    error handling, not conversion quality.
    """
    import soundfile as sf

    from app.services import voxcpm_vc_service as vc

    def fake_health(self):
        return {"status": "ok", "device": "stub"}

    def fake_convert(self, source, target, output, **kwargs):
        # Accept the F0-compare kwargs (auto_f0_adjust / semi_tone_shift) and ignore
        # them: the stub is a pass-through, so the three F0 modes differ only by name.
        audio, sr = sf.read(str(source), dtype="float32")
        sf.write(str(output), audio, sr, format="WAV", subtype="PCM_16")
        return output

    monkeypatch.setattr(vc.VoxCPMVCService, "seedvc_health", fake_health)
    monkeypatch.setattr(vc.VoxCPMVCService, "_convert", fake_convert)
    # A donor handle is cached per process; the mock engine hands back a different
    # kind of object than the real one, so a handle built under one must not leak
    # into a test running under the other.
    vc.voxcpm_vc_service._donor_handles.clear()
    yield
    vc.voxcpm_vc_service._donor_handles.clear()
