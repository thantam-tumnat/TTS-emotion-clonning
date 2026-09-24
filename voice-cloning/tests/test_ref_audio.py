"""Trimming a cut-off word from the end of a reference clip (src/ref_audio.py).

The clips are synthetic: tone bursts stand in for speech, a low hiss for the room.
What is on trial is only where a clip ends, which these control exactly.
"""
import numpy as np
import pytest

from src import ref_audio

SR = 16000


def _speech(seconds, level=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (level * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def _hiss(seconds, level=1e-4, seed=0):
    return (level * np.random.default_rng(seed).standard_normal(int(SR * seconds))).astype(np.float32)


def _clip(*parts):
    return np.concatenate(parts)


def _phrase():
    """~4 s of speech with ordinary pauses in it."""
    return _clip(_hiss(0.1), _speech(1.2), _hiss(0.3), _speech(1.5), _hiss(0.25), _speech(1.0))


def test_a_cut_off_onset_after_a_pause_is_trimmed():
    # What n8n's fixed 10 s window produced: the phrase, a pause, then 40 ms of the
    # next word before the cut.
    clip = _clip(_phrase(), _hiss(0.28), _speech(0.04))
    check = ref_audio.check_tail(clip, SR)
    assert check.status == "trimmed"
    assert check.fragment_s == pytest.approx(0.04, abs=0.015)
    phrase_end = len(_phrase()) / SR
    assert phrase_end <= check.cut_s <= phrase_end + ref_audio.HOLD_S + 0.015


def test_trimmed_clip_ends_on_silence_and_keeps_the_phrase():
    clip = _clip(_phrase(), _hiss(0.28), _speech(0.04))
    check = ref_audio.check_tail(clip, SR)
    out = ref_audio.apply_tail(clip, SR, check)

    assert len(out) == int(round(check.cut_s * SR)) + int(round(ref_audio.PAD_S * SR))
    assert np.all(out[-int(ref_audio.PAD_S * SR):] == 0)
    body = len(_phrase()) - int(0.05 * SR)
    np.testing.assert_array_equal(out[:body], clip[:body])


def test_a_clip_that_already_ends_on_silence_is_left_alone():
    clip = _clip(_phrase(), _hiss(0.3))
    check = ref_audio.check_tail(clip, SR)
    assert check.status == "clean"
    assert ref_audio.apply_tail(clip, SR, check) is clip


def test_a_long_tail_may_be_a_transcribed_word_and_is_not_touched():
    # 0.6 s after the last pause is a whole syllable or two: the transcript may well
    # include it, and cutting audio the transcript describes causes read-back.
    clip = _clip(_phrase(), _hiss(0.28), _speech(0.6))
    check = ref_audio.check_tail(clip, SR)
    assert check.status == "mid_speech"
    assert not check.trimmed
    assert ref_audio.apply_tail(clip, SR, check) is clip


def test_continuous_speech_to_the_end_is_not_touched():
    clip = _clip(_hiss(0.1), _speech(4.0))
    assert ref_audio.check_tail(clip, SR).status == "mid_speech"


def test_a_short_clip_is_never_trimmed():
    clip = _clip(_speech(1.5), _hiss(0.3), _speech(0.04))
    assert ref_audio.check_tail(clip, SR).status == "too_short"


def test_pauses_are_found_over_a_noisy_room():
    # A -40 dBFS floor under speech at about -13: a threshold fixed at 30 dB below
    # speech sits under the floor and finds no pause at all.
    room = 0.015
    clip = _clip(_hiss(0.1, room), _speech(1.2), _hiss(0.3, room), _speech(1.5),
                 _hiss(0.28, room), _speech(0.05))
    assert ref_audio.check_tail(clip, SR).status == "trimmed"


def test_stereo_is_trimmed_on_every_channel():
    mono = _clip(_phrase(), _hiss(0.28), _speech(0.04))
    stereo = np.stack([mono, mono * 0.5], axis=1)
    check = ref_audio.check_tail(stereo, SR)
    out = ref_audio.apply_tail(stereo, SR, check)
    assert check.trimmed
    assert out.shape[1] == 2
    assert np.all(out[-int(ref_audio.PAD_S * SR):] == 0)


def test_read_clip_round_trips_a_wav(tmp_path):
    import soundfile as sf

    path = tmp_path / "v.wav"
    sf.write(str(path), _phrase(), SR)
    samples, sr = ref_audio.read_clip(path)
    assert sr == SR
    assert len(samples) == len(_phrase())
