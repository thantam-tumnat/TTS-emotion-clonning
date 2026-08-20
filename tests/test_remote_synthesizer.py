import pytest
from app.services.siangtts_service import _RemoteSynthesizer, SiangTTSService


def test_remote_synthesizer_init():
    synth = _RemoteSynthesizer("http://127.0.0.1:8000")
    assert synth.base_url == "http://127.0.0.1:8000"
    assert synth.sample_rate == 48000


def test_remote_synthesizer_health_fail_graceful():
    synth = _RemoteSynthesizer("http://127.0.0.1:9999")
    assert synth.check_health() is False


def test_remote_voice_caching():
    synth = _RemoteSynthesizer("http://127.0.0.1:8000")
    v = synth.build_voice("ref/test.wav")
    assert "remote_latent" in v
