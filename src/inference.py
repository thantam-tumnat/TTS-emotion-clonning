"""Thin VoxCPM 2 inference wrapper for Thai prompts.

Used for:
- "before" baseline checks against the unmodified base model.
- Loading a trained LoRA adapter for qualitative inspection / eval audio generation.

The LoRA checkpoint directory written by `train/train_lora.py`
(`lora_weights.safetensors` + `lora_config.json`) loads directly via
`lora_weights_path`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from .thai_normalizer import normalize_thai_text

DEFAULT_BASE_MODEL = "openbmb/VoxCPM2"


class Synthesizer:
    """Loads the model once; reuse across many prompts (eval loops)."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str | None = None,
        device: str | None = None,
        denoise_prompts: bool = False,
    ) -> None:
        from voxcpm import VoxCPM  # lazy import — keeps `--help` snappy

        lora_config = None
        if adapter_path is not None:
            cfg_file = Path(adapter_path) / "lora_config.json"
            if cfg_file.exists():
                from voxcpm.model.voxcpm2 import LoRAConfig

                with open(cfg_file, encoding="utf-8") as f:
                    lora_config = LoRAConfig(**json.load(f)["lora_config"])

        self.model = VoxCPM.from_pretrained(
            base_model,
            load_denoiser=denoise_prompts,
            lora_config=lora_config,
            lora_weights_path=adapter_path,
        )
        self.sample_rate = self.model.tts_model.sample_rate

    def synth(
        self,
        text: str,
        *,
        ref_audio: str | None = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
    ):
        text = normalize_thai_text(text)
        return self.model.generate(
            text=text,
            reference_wav_path=ref_audio,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )

    def synth_to_file(self, text: str, output_path: str | Path, **kwargs) -> Path:
        wav = self.synth(text, **kwargs)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), wav, self.sample_rate)
        return output_path


def synthesize(
    text: str,
    output_path: str | Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_path: str | None = None,
    ref_audio: str | None = None,
    cfg_value: float = 2.5,
    inference_timesteps: int = 10,
) -> Path:
    """One-shot convenience wrapper. For many prompts, use `Synthesizer`."""
    s = Synthesizer(base_model=base_model, adapter_path=adapter_path)
    return s.synth_to_file(
        text,
        output_path,
        ref_audio=ref_audio,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Synthesize Thai speech with VoxCPM 2 / SiangTTS.")
    p.add_argument("--text", required=True, help="Thai text to synthesize")
    p.add_argument("--out", default="out.wav", help="Output WAV path")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter (omit for base only)")
    p.add_argument("--base-only", action="store_true", help="Force --adapter=None")
    p.add_argument("--ref-audio", default=None, help="Reference WAV for voice cloning")
    p.add_argument("--cfg-value", type=float, default=2.5)
    p.add_argument("--timesteps", type=int, default=10)
    args = p.parse_args()

    adapter = None if args.base_only else args.adapter
    out = synthesize(
        args.text,
        args.out,
        base_model=args.base_model,
        adapter_path=adapter,
        ref_audio=args.ref_audio,
        cfg_value=args.cfg_value,
        inference_timesteps=args.timesteps,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
