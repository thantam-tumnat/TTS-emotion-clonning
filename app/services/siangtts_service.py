from __future__ import annotations

import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

from app.config import settings
from app.services.thai_normalizer import normalize_thai_text

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


class SiangTTSService:
    def __init__(
        self,
        ref_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        base_model: str | None = None,
        adapter_path: str | None = None,
        device: str | None = None,
    ):
        self.ref_dir = Path(ref_dir or settings.siangtts_ref_dir)
        self.cache_dir = Path(cache_dir or settings.siangtts_cache_dir)
        self.base_model = base_model or settings.siangtts_base_model
        self.adapter_path = adapter_path or settings.siangtts_adapter
        self.device = device or settings.siangtts_device or None
        
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._voices: Dict[str, Any] = {}
        self._synthesizer: Any = None
        self._is_loaded: bool = False

    def get_synthesizer(self) -> Any:
        """Lazy loader for VoxCPM / SiangTTS Synthesizer."""
        if self._synthesizer is not None:
            return self._synthesizer

        try:
            import json
            from voxcpm import VoxCPM
            from voxcpm.model.voxcpm2 import LoRAConfig

            lora_config = None
            adapter = self.adapter_path if self.adapter_path and Path(self.adapter_path).exists() else None
            
            if adapter:
                cfg_file = Path(adapter) / "lora_config.json"
                if cfg_file.exists():
                    with open(cfg_file, encoding="utf-8") as f:
                        cfg_data = json.load(f).get("lora_config", {})
                        lora_config = LoRAConfig(**cfg_data)

            class _SynthesizerImpl:
                def __init__(self, base_model: str, adapter_path: str | None, lora_config: Any):
                    self.model = VoxCPM.from_pretrained(
                        base_model,
                        lora_config=lora_config,
                        lora_weights_path=adapter_path,
                    )
                    self.sample_rate = 48000

                def synth(self, text: str, *, ref_audio: str | None = None, prompt_cache: Any = None, cfg_value: float = 2.5, inference_timesteps: int = 10):
                    text = normalize_thai_text(text)
                    if prompt_cache is not None:
                        return self.model.generate_with_prompt_cache(
                            text=text,
                            prompt_cache=prompt_cache,
                            cfg_value=cfg_value,
                            inference_timesteps=inference_timesteps,
                        )
                    return self.model.generate(
                        text=text,
                        reference_wav_path=ref_audio,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                    )

                def build_voice(self, ref_audio_path: str) -> Any:
                    if hasattr(self.model, "build_prompt_cache"):
                        return self.model.build_prompt_cache(ref_audio_path)
                    return ref_audio_path

                def save_voice(self, cache: Any, dest_path: Path):
                    import torch
                    if hasattr(torch, "save") and not isinstance(cache, (str, Path)):
                        torch.save(cache, dest_path)

                def load_voice(self, src_path: Path) -> Any:
                    import torch
                    if hasattr(torch, "load"):
                        return torch.load(src_path, map_location="cpu")
                    return str(src_path)

            self._synthesizer = _SynthesizerImpl(self.base_model, adapter, lora_config)
            self._is_loaded = True
            return self._synthesizer
        except Exception as e:
            # Fallback mock synthesizer for CPU/testing environment
            print(f"[SiangTTS] voxcpm/torch not active or failed to load: {e}. Using fallback simulation mode.")
            return self._get_fallback_synthesizer()

    def _get_fallback_synthesizer(self) -> Any:
        class _MockSynthesizer:
            def __init__(self):
                self.sample_rate = 48000

            def synth(self, text: str, *, ref_audio: str | None = None, prompt_cache: Any = None, cfg_value: float = 2.5, inference_timesteps: int = 10):
                import numpy as np
                text = normalize_thai_text(text)
                duration_sec = max(1.0, min(10.0, len(text) * 0.12))
                num_samples = int(self.sample_rate * duration_sec)
                t = np.linspace(0, duration_sec, num_samples, endpoint=False)
                # Gentle pleasant tone burst for testing
                wave = (0.2 * np.sin(2 * np.pi * 440 * t) * np.exp(-t / 3.0)).astype(np.float32)
                return wave

            def build_voice(self, ref_audio_path: str) -> Any:
                return f"mock_latent_for_{ref_audio_path}"

            def save_voice(self, cache: Any, dest_path: Path):
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(str(cache))

            def load_voice(self, src_path: Path) -> Any:
                return f"loaded_mock_from_{src_path}"

        if self._synthesizer is None:
            self._synthesizer = _MockSynthesizer()
        return self._synthesizer

    def init_speakers(self) -> None:
        """Scan ref/ directory, precompute and cache prompt latents."""
        synth = self.get_synthesizer()
        for ref_file in sorted(self.ref_dir.iterdir()):
            if ref_file.suffix.lower() not in AUDIO_EXTS:
                continue
            sid = ref_file.stem
            cache_path = self.cache_dir / f"{sid}.pt"
            if cache_path.exists() and cache_path.stat().st_mtime >= ref_file.stat().st_mtime:
                try:
                    self._voices[sid] = synth.load_voice(cache_path)
                    continue
                except Exception:
                    pass
            try:
                cache = synth.build_voice(str(ref_file))
                synth.save_voice(cache, cache_path)
                self._voices[sid] = cache
            except Exception as ex:
                print(f"[SiangTTS] Warning: Could not cache voice '{sid}': {ex}")

    def list_speakers(self) -> List[Dict[str, Any]]:
        """List registered speaker profiles."""
        speakers = []
        for ref_file in sorted(self.ref_dir.iterdir()):
            if ref_file.suffix.lower() not in AUDIO_EXTS:
                continue
            sid = ref_file.stem
            cache_path = self.cache_dir / f"{sid}.pt"
            speakers.append({
                "id": sid,
                "name": sid.replace("_", " ").title(),
                "filename": ref_file.name,
                "cached": cache_path.exists(),
            })
        return speakers

    def register_speaker(self, speaker_id: str, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Save a new reference audio and precompute latent cache."""
        clean_id = "".join(c for c in speaker_id.strip().lower() if c.isalnum() or c in ("-", "_"))
        if not clean_id:
            clean_id = "custom_speaker"

        ext = Path(filename).suffix.lower()
        if ext not in AUDIO_EXTS:
            ext = ".wav"

        ref_path = self.ref_dir / f"{clean_id}{ext}"
        with open(ref_path, "wb") as f:
            f.write(audio_bytes)

        synth = self.get_synthesizer()
        cache_path = self.cache_dir / f"{clean_id}.pt"
        try:
            cache = synth.build_voice(str(ref_path))
            synth.save_voice(cache, cache_path)
            self._voices[clean_id] = cache
        except Exception as e:
            print(f"[SiangTTS] Failed to cache latent for {clean_id}: {e}")

        return {
            "id": clean_id,
            "name": clean_id.replace("_", " ").title(),
            "filename": ref_path.name,
            "cached": cache_path.exists(),
        }

    def delete_speaker(self, speaker_id: str) -> bool:
        """Remove a speaker profile from disk and memory."""
        found = False
        for f in self.ref_dir.glob(f"{speaker_id}.*"):
            if f.is_file():
                f.unlink(missing_ok=True)
                found = True

        cache_path = self.cache_dir / f"{speaker_id}.pt"
        if cache_path.exists():
            cache_path.unlink(missing_ok=True)

        if speaker_id in self._voices:
            del self._voices[speaker_id]
            found = True

        return found

    def synthesize(
        self,
        text: str,
        *,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
    ) -> bytes:
        """
        Synthesize Thai speech with voice cloning and emotion control instruction.
        Returns WAV binary audio bytes.
        """
        import soundfile as sf
        synth = self.get_synthesizer()
        sample_rate = getattr(synth, "sample_rate", 48000)

        # 1. Check if direct ref audio uploaded
        temp_ref_path = None
        prompt_cache = None
        ref_audio_path = None

        if ref_audio_bytes:
            ext = Path(ref_filename or "upload.wav").suffix.lower()
            if ext not in AUDIO_EXTS:
                ext = ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(ref_audio_bytes)
                temp_ref_path = tf.name
            ref_audio_path = temp_ref_path
        elif speaker_id:
            if speaker_id in self._voices:
                prompt_cache = self._voices[speaker_id]
            else:
                # Find file in ref/
                for cand in self.ref_dir.glob(f"{speaker_id}.*"):
                    if cand.suffix.lower() in AUDIO_EXTS:
                        ref_audio_path = str(cand)
                        break

        try:
            wav = synth.synth(
                text=text,
                ref_audio=ref_audio_path,
                prompt_cache=prompt_cache,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
            out_buf = io.BytesIO()
            sf.write(out_buf, wav, sample_rate, format="WAV")
            return out_buf.getvalue()
        finally:
            if temp_ref_path and os.path.exists(temp_ref_path):
                try:
                    os.remove(temp_ref_path)
                except Exception:
                    pass


siangtts_service = SiangTTSService()
