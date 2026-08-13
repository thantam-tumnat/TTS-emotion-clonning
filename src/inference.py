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

    # ------------------------------------------------------------------
    # Reusable voice cache — encode a reference once, clone many times.
    # ------------------------------------------------------------------
    def _to_device(self, cache: dict) -> dict:
        """Put every tensor in a prompt cache on the model's device.

        `build_prompt_cache` hands back mixed placement — in ref_continuation
        mode the reference feature comes back on CPU while the rest is on the
        GPU — and generation concatenates the two, which torch.cat refuses.
        Loading a cache saved on another machine has the same problem.
        """
        import torch

        dev = self.model.tts_model.device

        def move(v):
            if torch.is_tensor(v):
                return v.to(dev)
            if isinstance(v, dict):
                return {k: move(x) for k, x in v.items()}
            if isinstance(v, list):
                return [move(x) for x in v]
            if isinstance(v, tuple):
                return tuple(move(x) for x in v)
            return v

        return {k: move(v) for k, v in cache.items()}

    def build_voice(self, ref_audio: str, prompt_text: str | None = None,
                    prompt_audio: str | None = None) -> dict:
        """Encode a reference clip into a reusable prompt cache (AudioVAE
        latents + ref tokens). Do this once per voice; pass the result to
        `synth_cached` for each generation to skip re-encoding the reference."""
        # voxcpm rejects a prompt_text with no prompt_wav_path ("must both be
        # provided or both be None"). A transcript on its own means "the
        # reference clip is the prompt" — ref_continuation mode — so pair it up
        # rather than making every caller remember to pass the path twice.
        if prompt_text and prompt_audio is None:
            prompt_audio = ref_audio
        return self._to_device(self.model.tts_model.build_prompt_cache(
            reference_wav_path=ref_audio,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_audio,
        ))

    def save_voice(self, prompt_cache: dict, path: str | Path) -> Path:
        """Persist a prompt cache to disk (tensors moved to CPU)."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cpu_cache = {
            k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in prompt_cache.items()
        }
        torch.save(cpu_cache, str(path))
        return path

    def load_voice(self, path: str | Path) -> dict:
        """Load a persisted prompt cache onto the model's device."""
        import torch

        dev = self.model.tts_model.device
        # map_location covers the top level; _to_device catches tensors nested
        # inside dicts/lists that save_voice wrote out unmoved.
        return self._to_device(torch.load(str(path), map_location=dev))

    def synth_cached(
        self,
        text: str,
        prompt_cache: dict,
        *,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
    ):
        """Synthesize using a prompt cache from `build_voice` (no ref re-encode).
        Mirrors the text cleanup the high-level `generate()` applies."""
        import re

        text = normalize_thai_text(text)
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        wav, _, _ = self.model.tts_model.generate_with_prompt_cache(
            target_text=text,
            prompt_cache=prompt_cache,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        return wav.squeeze(0).cpu().numpy()


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
