"""Every take of a benchmark run must be the same speaker.

The /test benchmark page renders one take at a time: each take is its own
``synthesize_variants`` call (one chunk of text), and the frontend sends the
same ``speaker_id`` for all of them. The existing chunk-split tests only prove
one voice *within* a single call. The failure the ear catches -- "each take is
a different person" -- lives *between* calls, so that is what these lock:

  * pinned speaker  -> every take conditions on the same cached latent
  * auto-seed (no speaker, the UI default "Auto-Seed Neutral Voice") -> the
    seed voice is minted once and every take conditions on that same seed
  * the seed is not regenerated per take (which was itself a fresh speaker each
    time, the original bug)

The spy records the prompt_cache handed to every generation, so "same speaker"
becomes "same object conditioned every take".
"""

import pytest

from app.services import siangtts_service as svc


def _spy(monkeypatch):
    """Record (text, prompt_cache) for every generation across all calls."""
    service = svc.siangtts_service
    synth = service.get_synthesizer()
    seen = []
    original = synth.synth

    def spy(text, **kwargs):
        seen.append((text, kwargs.get("prompt_cache")))
        return original(text, **kwargs)

    monkeypatch.setattr(synth, "synth", spy)
    return service, seen


def _run_takes(service, n, **kwargs):
    """Simulate the benchmark: n separate single-chunk renders, one per take."""
    for _ in range(n):
        service.synthesize_variants(
            ["ข้อความทดสอบเสียง"],
            variants=[{"post_process": False, "params": None}],
            tones=["neutral"],
            breaks=[False],
            **kwargs,
        )


def test_pinned_speaker_is_the_same_voice_every_take(monkeypatch):
    service, seen = _spy(monkeypatch)
    service._voices["spk1"] = "spk1_latent"

    try:
        _run_takes(service, 3, speaker_id="spk1")
    finally:
        service._voices.pop("spk1", None)

    caches = [c for _, c in seen]
    assert caches == ["spk1_latent"] * 3, (
        "each take resolved a different latent -> different person per take"
    )


def test_auto_seed_is_the_same_voice_every_take(monkeypatch):
    """No speaker picked (UI default). Every take must share one seed voice."""
    monkeypatch.setattr(
        svc.settings, "siangtts_auto_voice_consistency", True, raising=False
    )
    service, seen = _spy(monkeypatch)
    service._synthesizer.build_voice = lambda path, prompt_text=None: "seed_voice"

    _run_takes(service, 3, speaker_id=None)

    seed_text = svc.settings.siangtts_voice_seed_text
    spoken_caches = [(t, c) for t, c in seen if t != seed_text]

    # Three takes, each one spoken generation, all on the same seed.
    assert len(spoken_caches) == 3
    assert [c for _, c in spoken_caches] == ["seed_voice"] * 3, (
        "takes did not share one seed voice -> each take a different person"
    )


def test_seed_voice_is_minted_once_not_per_take(monkeypatch):
    """The seed line is generated exactly once across the whole run.

    Regenerating the seed per take was the original defect: an unseeded
    generation is a fresh random speaker, so every take became someone new.
    """
    monkeypatch.setattr(
        svc.settings, "siangtts_auto_voice_consistency", True, raising=False
    )
    service, seen = _spy(monkeypatch)
    service._synthesizer.build_voice = lambda path, prompt_text=None: "seed_voice"

    _run_takes(service, 4, speaker_id=None)

    seed_text = svc.settings.siangtts_voice_seed_text
    seed_generations = [t for t, _ in seen if t == seed_text]
    assert len(seed_generations) == 1, (
        f"seed voice generated {len(seed_generations)} times; it must be minted "
        f"once and reused, or every take is a different speaker"
    )


def test_anchor_reported_as_speaker_when_pinned(monkeypatch):
    service, _ = _spy(monkeypatch)
    service._voices["spk1"] = "spk1_latent"
    try:
        _run_takes(service, 1, speaker_id="spk1")
    finally:
        service._voices.pop("spk1", None)
    assert service.last_voice_anchor() == "speaker"


def test_anchor_reported_as_seed_for_auto_seed(monkeypatch):
    monkeypatch.setattr(
        svc.settings, "siangtts_auto_voice_consistency", True, raising=False
    )
    service, _ = _spy(monkeypatch)
    service._synthesizer.build_voice = lambda path, prompt_text=None: "seed_voice"
    _run_takes(service, 1, speaker_id=None)
    assert service.last_voice_anchor() == "seed"


def test_anchor_reported_as_none_when_unconditioned(monkeypatch):
    """The guard's whole purpose: the take with no voice anchor is *labelled*
    'none', so the API and UI can warn instead of shipping a silent mismatch."""
    monkeypatch.setattr(
        svc.settings, "siangtts_auto_voice_consistency", False, raising=False
    )
    service, _ = _spy(monkeypatch)
    _run_takes(service, 1, speaker_id=None)
    assert service.last_voice_anchor() == "none"


def test_auto_seed_off_leaves_takes_unconditioned(monkeypatch):
    """Documents the hazard: with consistency disabled and no speaker, nothing
    anchors the voice, so every take is unconditioned -- a different person.

    This is the ear-obvious failure. The guard against it is
    ``siangtts_auto_voice_consistency`` (default True); this test pins down what
    turning it off costs so the flag is never treated as cosmetic.
    """
    monkeypatch.setattr(
        svc.settings, "siangtts_auto_voice_consistency", False, raising=False
    )
    service, seen = _spy(monkeypatch)

    _run_takes(service, 2, speaker_id=None)

    caches = [c for _, c in seen]
    assert caches == [None, None], (
        "expected unconditioned takes when the consistency guard is off"
    )
