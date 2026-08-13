"""Tests for Synthesizer._to_device.

A prompt cache has to be single-device before generation concatenates its
tensors. What can actually break is the traversal — whether nested containers
are reached at all — so these run on CPU and check placement plus structure.
The Synthesizer is built without __init__ so no model is loaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.inference import Synthesizer


def _synth(device: str = "cpu") -> Synthesizer:
    s = object.__new__(Synthesizer)
    s.model = SimpleNamespace(tts_model=SimpleNamespace(device=torch.device(device)))
    return s


def _devices(value, path: str = ""):
    """Every tensor in the structure, as (path, device string)."""
    if torch.is_tensor(value):
        yield path, str(value.device)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _devices(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _devices(v, f"{path}[{i}]")


def test_moves_nested_tensors():
    cache = {
        "ref_audio_feat": torch.zeros(2),
        "nested": {"a": torch.ones(2)},
        "seq": [torch.ones(1), (torch.ones(1), torch.ones(1))],
    }
    out = _synth()._to_device(cache)
    found = dict(_devices(out))
    assert set(found) == {
        ".ref_audio_feat", ".nested.a", ".seq[0]", ".seq[1][0]", ".seq[1][1]",
    }
    assert all(d == "cpu" for d in found.values())


def test_preserves_non_tensors():
    cache = {"scalar": 7, "text": "hello", "none": None, "nested": {"b": "keep"}}
    out = _synth()._to_device(cache)
    assert out == cache


def test_preserves_container_types():
    out = _synth()._to_device({"t": (torch.ones(1), 3), "l": [torch.ones(1)]})
    assert isinstance(out["t"], tuple)
    assert isinstance(out["l"], list)


def test_empty_cache():
    assert _synth()._to_device({}) == {}


def test_does_not_mutate_input():
    inner = torch.ones(1)
    cache = {"a": inner}
    out = _synth()._to_device(cache)
    assert cache["a"] is inner
    assert torch.equal(out["a"], inner)
