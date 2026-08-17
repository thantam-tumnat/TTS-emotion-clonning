import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services import siangtts_service as svc


@pytest.fixture
def client():
    return TestClient(app)


def test_model_load_failure_raises_instead_of_beeping():
    """A failed load must surface as an error, not a 440Hz tone.

    This was the original bug: get_synthesizer() swallowed every exception and
    substituted the mock, so a model that never loaded was indistinguishable from
    a model producing artifacts.
    """
    service = svc.SiangTTSService()
    service._synthesizer = None

    def boom(*args, **kwargs):
        raise RuntimeError("The paging file is too small for this operation to complete.")

    svc.settings.siangtts_allow_mock = False
    original = svc._RealSynthesizer.__init__
    svc._RealSynthesizer.__init__ = boom
    try:
        with pytest.raises(svc.SynthesizerUnavailable) as exc:
            service.get_synthesizer()
        assert "paging file" in str(exc.value)
        assert service.status["loaded"] is False
    finally:
        svc._RealSynthesizer.__init__ = original


def test_mock_fallback_only_when_explicitly_enabled():
    service = svc.SiangTTSService()
    service._synthesizer = None

    def boom(*args, **kwargs):
        raise RuntimeError("no GPU")

    svc.settings.siangtts_allow_mock = True
    original = svc._RealSynthesizer.__init__
    svc._RealSynthesizer.__init__ = boom
    try:
        synth = service.get_synthesizer()
        assert isinstance(synth, svc._MockSynthesizer)
        assert service.status["using_mock"] is True
        assert "no GPU" in service.status["load_error"]
    finally:
        svc._RealSynthesizer.__init__ = original


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "speakers_count" in data


def test_speakers_crud(client):
    # 1. List speakers initially
    res = client.get("/speakers")
    assert res.status_code == 200
    initial_count = len(res.json()["speakers"])

    # 2. Register a new speaker via upload
    dummy_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    files = {"file": ("test_speaker_alpha.wav", io.BytesIO(dummy_wav), "audio/wav")}
    data = {"speaker_id": "test_speaker_alpha"}
    
    upload_res = client.post("/speakers", files=files, data=data)
    assert upload_res.status_code == 200
    speaker_data = upload_res.json()
    assert speaker_data["id"] == "test_speaker_alpha"

    # 3. List speakers again
    list_res = client.get("/speakers")
    assert list_res.status_code == 200
    ids = [s["id"] for s in list_res.json()["speakers"]]
    assert "test_speaker_alpha" in ids

    # 4. Delete speaker
    del_res = client.delete("/speakers/test_speaker_alpha")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True


def test_synthesize_endpoint_json(client):
    res = client.post("/synthesize", json={
        "text": "หายใจเข้าลึกๆ ผ่อนคลาย แล้วค่อยๆ ปล่อยวางทุกอย่างลงนะ",
        "guidance": "สงบ นุ่มนวล",
        "engine": "voxcpm",
        "cfg_value": 2.5,
        "inference_timesteps": 10,
        "auto_annotate": True
    })
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert len(res.content) > 0


def test_synthesize_endpoint_with_upload(client):
    dummy_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    files = {"file": ("my_sample.wav", io.BytesIO(dummy_wav), "audio/wav")}
    data = {
        "text": "[calm] หายใจเข้าลึกๆ ผ่อนคลาย",
        "cfg_value": "2.5",
        "inference_timesteps": "10",
        "auto_annotate": "false"
    }
    res = client.post("/synthesize/upload", files=files, data=data)
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert len(res.content) > 0


def test_speak_endpoint_voxcpm(client):
    res = client.post("/speak", json={
        "text": "หายใจเข้าลึกๆ ผ่อนคลาย",
        "guidance": "สงบ นุ่มนวล",
        "engine": "voxcpm"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["engine"] == "voxcpm"
    assert len(data["segments"]) > 0
    assert "(Calm" in data["text"] or "หายใจเข้าลึกๆ" in data["text"]
