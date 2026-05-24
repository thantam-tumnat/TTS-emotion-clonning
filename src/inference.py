"""Thin VoxCPM 2 inference wrapper for Thai prompts.

Used for:
- "before" baseline checks against the unmodified base model.
- Loading a trained LoRA adapter for qualitative inspection / eval audio generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from .thai_normalizer import normalize_thai_text


def synthesize(
    text: str,
    output_path: str | Path,
    *,
    base_model: str = "openbmb/VoxCPM2-2B",
    adapter_path: str | None = None,
    ref_audio: str | None = None,
    cfg_value: float = 2.5,
    inference_timesteps: int = 10,
    sample_rate: int = 48000,
) -> Path:
    """Generate a single utterance.

    NOTE: This is a stub — the exact `voxcpm` API surface for LoRA loading and
    reference-audio passing depends on the installed VoxCPM version. Adjust the
    `model.generate(...)` call to match your VoxCPM build before first run.
    """
    from voxcpm import VoxCPM  # lazy import — keeps `--help` snappy

    text = normalize_thai_text(text)

    model = VoxCPM.from_pretrained(base_model)
    if adapter_path is not None:
        # Replace with VoxCPM's own adapter-loading API once verified against the
        # installed version. Common shapes: model.load_lora(path) or PEFT-style.
        model.load_lora(adapter_path)  # type: ignore[attr-defined]

    wav = model.generate(
        text=text,
        prompt_wav=ref_audio,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wav, sample_rate)
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description="Synthesize Thai speech with VoxCPM 2 / VajaCPM.")
    p.add_argument("--text", required=True, help="Thai text to synthesize")
    p.add_argument("--out", default="out.wav", help="Output WAV path")
    p.add_argument("--base-model", default="openbmb/VoxCPM2-2B")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter (omit for base only)")
    p.add_argument("--base-only", action="store_true", help="Force --adapter=None")
    p.add_argument("--ref-audio", default=None, help="Reference WAV for voice cloning")
    args = p.parse_args()

    adapter = None if args.base_only else args.adapter
    out = synthesize(
        args.text,
        args.out,
        base_model=args.base_model,
        adapter_path=adapter,
        ref_audio=args.ref_audio,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
