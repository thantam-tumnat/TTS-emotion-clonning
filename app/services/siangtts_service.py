from __future__ import annotations

import io
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Any, Sequence, Tuple

from app.config import settings
from app.services.pronunciation import apply_pronunciation
from app.services.thai_normalizer import normalize_thai_text

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

# Mock-only. The real synthesizer reports the model's own rate.
MOCK_SAMPLE_RATE = 48000


class SynthesizerUnavailable(RuntimeError):
    """Raised when the real VoxCPM2 model could not be loaded."""


_LEADING_STYLE_RE = re.compile(r"^\s*\([^)]*\)\s*")


def spoken_len(text: str) -> int:
    """Characters VoxCPM2 will actually voice.

    The leading style parenthetical is direction, not speech, so counting it would
    make a chunk with a long instruction look slower than it is -- and rate matching
    would then stretch it the wrong way.
    """
    return len(_LEADING_STYLE_RE.sub("", text or "").strip())


# VoxCPM2 voices a whole chunk in one autoregressive pass: voxcpm.core does no
# sentence splitting of its own, and it flattens newlines to spaces before
# generating. Past roughly this many spoken characters the speaker identity
# wanders mid-utterance -- five repetitions of one 98-character line reached the
# model as a single 494-character chunk and came back with the third sentence in
# a different voice. For scale, VoxCPM's own text normalizer splits paragraphs at
# 60-80 tokens. Splitting below the drift point and conditioning every piece on
# one prompt cache is what holds the timbre steady.
DEFAULT_MAX_CHUNK_CHARS = 140


class ChunkPiece(NamedTuple):
    """One synthesizable piece of a chunk that was too long to voice in one pass."""
    text: str              # instruction + body, ready for the engine
    paragraph_seam: bool   # this piece began after a line break in the source


def _atoms(text: str, level: int) -> Optional[List[str]]:
    """Break ``text`` at seam ``level``, or None when that seam is not present.

    Best seam first: a newline is the writer's own sentence mark, then terminal
    punctuation, then a space -- which in Thai separates clauses the way a comma
    does in English. Word tokens come last so even a forced break lands between
    words rather than inside a character cluster.
    """
    if level == 0:
        parts = re.split(r"(?<=\n)", text)
    elif level == 1:
        parts = re.split(r"(?<=[.!?…])\s*", text)
    elif level == 2:
        parts = re.split(r"(?<= )", text)
    elif level == 3:
        try:
            from pythainlp.tokenize import word_tokenize

            parts = word_tokenize(text, keep_whitespace=True)
        except Exception:
            return None
    else:
        return None

    parts = [p for p in parts if p]
    return parts if len(parts) > 1 else None


def _split_body(body: str, limit: int, level: int = 0) -> List[str]:
    """Greedily pack ``body`` into runs of at most ``limit`` spoken characters."""
    if len(body.strip()) <= limit:
        return [body] if body.strip() else []

    atoms = _atoms(body, level)
    if atoms is None:
        if level <= 3:
            return _split_body(body, limit, level + 1)
        # No seam left anywhere: cut on the character budget.
        return [body[i:i + limit] for i in range(0, len(body), limit)]

    packed: List[str] = []
    cur = ""
    for atom in atoms:
        if cur and len((cur + atom).strip()) > limit:
            packed.append(cur)
            cur = atom
        else:
            cur += atom
    if cur.strip():
        packed.append(cur)

    # A single atom can still be over budget on its own -- break that one finer.
    if any(len(p.strip()) > limit for p in packed):
        return [q for p in packed for q in _split_body(p, limit, level + 1)]
    return [p for p in packed if p.strip()]


def split_for_synthesis(text: str, limit: Optional[int] = None) -> List[ChunkPiece]:
    """Break one chunk into pieces VoxCPM2 can voice without the speaker drifting.

    The leading style parenthetical is re-attached to every piece: VoxCPM2 honours
    one only at position 0, so a piece that lost it would fall back to neutral
    partway through an emotion.
    """
    limit = limit or settings.siangtts_max_chunk_chars or DEFAULT_MAX_CHUNK_CHARS

    m = _LEADING_STYLE_RE.match(text or "")
    instruction = m.group(0).strip() if m else ""
    body = (text[m.end():] if m else (text or "")).strip()
    if not body:
        return []

    pieces = _split_body(body, max(1, limit))
    out: List[ChunkPiece] = []
    for i, piece in enumerate(pieces):
        clean = piece.strip()
        if not clean:
            continue
        # A piece that starts a new line earns the same long pause a hand-written
        # line break gets; see audio_post.GAP_PARAGRAPH_S.
        out.append(
            ChunkPiece(
                text=f"{instruction}{clean}",
                paragraph_seam=bool(i and pieces[i - 1].endswith("\n")),
            )
        )
    return out


def prepare_text(text: str) -> str:
    """Everything the text needs between the API and the model.

    Encoding hygiene first, then the user's pronunciation overrides -- but only over
    the spoken body. The leading style parenthetical is direction for the engine, not
    speech, so an override must never rewrite a word inside it.
    """
    text = normalize_thai_text(text)
    m = _LEADING_STYLE_RE.match(text)
    if m:
        return text[:m.end()] + apply_pronunciation(text[m.end():])
    return apply_pronunciation(text)


def _wav_to_numpy(wav: Any):
    """Normalize a VoxCPM waveform (tensor or ndarray) to a 1-D float32 array."""
    import numpy as np

    if hasattr(wav, "detach"):
        wav = wav.detach()
        if wav.dim() > 1:
            wav = wav.squeeze(0)
        # bfloat16 has no numpy equivalent; float() is a no-op when already float32.
        return wav.float().cpu().numpy()
    return np.asarray(wav, dtype="float32")


def set_lora_strength(tts_model: Any, lm: float, dit: float) -> Dict[str, float]:
    """Scale the loaded Thai LoRA independently on the LM and the DiT side.

    VoxCPM2 injects LoRA as ``LoRALinear`` layers that keep their strength in a
    ``scaling`` buffer rather than baked into the weights, so this is a live dial,
    not a reload. Nothing on a layer says which side it belongs to -- both sides
    were injected with the same rank and alpha -- so the split comes from where the
    layer lives: the LM is base_lm + residual_lm, the DiT is the feature decoder's
    estimator.

    Why this exists at all, and why the DiT side defaults to zero, is in
    app/config.py next to the settings. Returns what was actually applied so the
    caller can report it; a model without LoRA layers returns zero counts and is
    otherwise untouched.
    """
    try:
        from voxcpm.modules.layers.lora import LoRALinear
    except Exception as e:  # pragma: no cover - depends on the installed voxcpm
        print(f"[SiangTTS] Cannot scale LoRA ({e}); leaving it at its shipped strength.",
              file=sys.stderr)
        return {"lm": 0, "dit": 0}

    def apply(root: Any, value: float) -> int:
        n = 0
        for module in root.modules():
            if isinstance(module, LoRALinear):
                module.scaling.fill_(value)
                n += 1
        return n

    n_lm = apply(tts_model.base_lm, lm) + apply(tts_model.residual_lm, lm)
    n_dit = apply(tts_model.feat_decoder.estimator, dit)
    print(
        f"[SiangTTS] LoRA strength: lm={lm} ({n_lm} layers), dit={dit} ({n_dit} layers)",
        file=sys.stderr,
    )
    return {"lm": lm, "dit": dit, "lm_layers": n_lm, "dit_layers": n_dit}


class _RealSynthesizer:
    """Thin adapter over the VoxCPM wrapper.

    Note that prompt-cache construction and cached generation live on
    ``model.tts_model`` (VoxCPM2Model), not on the ``VoxCPM`` wrapper. Calling them
    on the wrapper silently does nothing useful.
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: Optional[str],
        lora_config: Any,
        device: Optional[str],
        load_denoiser: bool,
        optimize: bool,
    ):
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(
            base_model,
            load_denoiser=load_denoiser,
            optimize=optimize,
            device=device,
            lora_config=lora_config,
            lora_weights_path=adapter_path,
        )
        self.tts_model = self.model.tts_model
        self.sample_rate = self.tts_model.sample_rate
        self.lora_loaded = adapter_path is not None
        if self.lora_loaded:
            self.lora_scales = set_lora_strength(
                self.tts_model,
                settings.siangtts_lora_lm_scale,
                settings.siangtts_lora_dit_scale,
            )

    def synth(
        self,
        text: str,
        *,
        ref_audio: Optional[str] = None,
        prompt_cache: Any = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
    ):
        text = prepare_text(text)
        if prompt_cache is not None:
            wav, _, _ = self.tts_model.generate_with_prompt_cache(
                target_text=text,
                prompt_cache=prompt_cache,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                retry_badcase=True,
            )
            return _wav_to_numpy(wav)
        return _wav_to_numpy(
            self.model.generate(
                text=text,
                reference_wav_path=ref_audio,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
        )

    def build_voice(self, ref_audio_path: str, prompt_text: Optional[str] = None) -> Any:
        """Encode a reference clip into a reusable prompt cache.

        With a transcript, VoxCPM2's "ultimate cloning" mode (reference + continuation)
        gives noticeably higher timbre fidelity, so use it when one is available.
        """
        if prompt_text:
            return self.tts_model.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=ref_audio_path,
                reference_wav_path=ref_audio_path,
            )
        return self.tts_model.build_prompt_cache(reference_wav_path=ref_audio_path)

    def save_voice(self, cache: Any, dest_path: Path) -> None:
        import torch

        torch.save(cache, dest_path)

    def load_voice(self, src_path: Path) -> Any:
        import torch

        return torch.load(src_path, map_location="cpu", weights_only=False)


class _MockSynthesizer:
    """Emits a 440 Hz tone. Test scaffolding only -- never a production fallback."""

    def __init__(self):
        self.sample_rate = MOCK_SAMPLE_RATE
        self.lora_loaded = False

    def synth(
        self,
        text: str,
        *,
        ref_audio: Optional[str] = None,
        prompt_cache: Any = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
    ):
        import numpy as np

        text = prepare_text(text)
        duration_sec = max(1.0, min(10.0, len(text) * 0.12))
        num_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        return (0.2 * np.sin(2 * np.pi * 440 * t) * np.exp(-t / 3.0)).astype("float32")

    def build_voice(self, ref_audio_path: str, prompt_text: Optional[str] = None) -> Any:
        return f"mock_latent_for_{ref_audio_path}"

    def save_voice(self, cache: Any, dest_path: Path) -> None:
        dest_path.write_text(str(cache), encoding="utf-8")

    def load_voice(self, src_path: Path) -> Any:
        return f"loaded_mock_from_{src_path}"


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
        self._load_error: Optional[str] = None
        self._using_mock: bool = False
        # The neutral voice unpinned multi-chunk requests are conditioned on. Built
        # at most once; see _build_seed_voice.
        self._seed_voice: Any = None
        self._seed_voice_failed: bool = False
        # Ultimate-cloning-vs-style-tags is a property of the voice, not the
        # request, so say it once rather than on every chunk.
        self._hifi_warned: bool = False

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _resolve_adapter(self) -> Optional[str]:
        """Resolve the LoRA adapter to a local directory.

        The configured value is normally a Hugging Face repo id, not a path, so a
        bare ``Path.exists()`` check would always miss and silently drop the LoRA.
        """
        spec = (self.adapter_path or "").strip()
        if not spec:
            return None

        local = Path(spec)
        if local.exists():
            return str(local)

        try:
            from huggingface_hub import snapshot_download

            resolved = snapshot_download(repo_id=spec)
            print(f"[SiangTTS] LoRA adapter resolved: {spec} -> {resolved}", file=sys.stderr)
            return resolved
        except Exception as e:
            # Base VoxCPM2 supports Thai natively, so this degrades quality rather
            # than breaking synthesis. Warn loudly and carry on.
            print(
                f"[SiangTTS] WARNING: could not fetch LoRA adapter '{spec}': {e}\n"
                f"[SiangTTS] Continuing with base {self.base_model} (lower Thai quality).",
                file=sys.stderr,
            )
            return None

    def _load_lora_config(self, adapter_dir: Optional[str]) -> Any:
        if not adapter_dir:
            return None
        cfg_file = Path(adapter_dir) / "lora_config.json"
        if not cfg_file.exists():
            # VoxCPM builds a sensible default when weights are given without a config.
            return None
        try:
            import json

            from voxcpm.model.voxcpm2 import LoRAConfig

            with open(cfg_file, encoding="utf-8") as f:
                cfg_data = json.load(f).get("lora_config", {})
            return LoRAConfig(**cfg_data)
        except Exception as e:
            print(f"[SiangTTS] WARNING: bad lora_config.json ({e}); using defaults.", file=sys.stderr)
            return None

    def get_synthesizer(self) -> Any:
        """Lazy-load the VoxCPM2 synthesizer.

        A failure here used to be swallowed and replaced with a sine-tone mock, which
        is indistinguishable from a broken model at the speaker. It now raises unless
        mock mode is explicitly enabled.
        """
        if self._synthesizer is not None:
            return self._synthesizer

        try:
            try:
                import torch._dynamo

                torch._dynamo.config.suppress_errors = True
            except Exception:
                pass

            adapter = self._resolve_adapter()
            lora_config = self._load_lora_config(adapter)

            self._synthesizer = _RealSynthesizer(
                base_model=self.base_model,
                adapter_path=adapter,
                lora_config=lora_config,
                device=self.device,
                load_denoiser=settings.siangtts_load_denoiser,
                optimize=settings.siangtts_optimize,
            )
            self._is_loaded = True
            self._using_mock = False
            self._load_error = None
            print(
                f"[SiangTTS] Loaded {self.base_model} "
                f"(LoRA={'yes' if adapter else 'no'}, sample_rate={self._synthesizer.sample_rate})",
                file=sys.stderr,
            )
            return self._synthesizer
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            if not settings.siangtts_allow_mock:
                raise SynthesizerUnavailable(
                    f"VoxCPM2 failed to load: {self._load_error}. "
                    f"This is usually GPU memory (VoxCPM2 needs ~8GB VRAM) or host "
                    f"commit charge. Free the GPU, or set SIANGTTS_DEVICE=cpu. "
                    f"Set SIANGTTS_ALLOW_MOCK=true only if you want a test tone."
                ) from e

            print(
                f"[SiangTTS] WARNING: model load failed ({self._load_error}).\n"
                f"[SiangTTS] SIANGTTS_ALLOW_MOCK is on -- output will be a 440Hz TEST TONE, not speech.",
                file=sys.stderr,
            )
            self._synthesizer = _MockSynthesizer()
            self._using_mock = True
            self._is_loaded = False
            return self._synthesizer

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "loaded": self._is_loaded,
            "using_mock": self._using_mock,
            "load_error": self._load_error,
            "base_model": self.base_model,
            "lora_loaded": bool(getattr(self._synthesizer, "lora_loaded", False)),
            "lora_scales": getattr(self._synthesizer, "lora_scales", None),
            "sample_rate": getattr(self._synthesizer, "sample_rate", None),
        }

    # ------------------------------------------------------------------ #
    # Speaker registry
    # ------------------------------------------------------------------ #

    def _transcript_for(self, ref_file: Path) -> Optional[str]:
        """Sidecar transcript (``ref/<id>.txt``), only when Hi-Fi cloning is on.

        Handing VoxCPM2 a transcript alongside the clip switches it into ultimate
        cloning, which the model docs say ignores the control instruction entirely
        and reproduces the prompt clip's own emotion instead. Since every style tag
        here *is* a control instruction, honouring a stray sidecar would silently
        flatten every emotion in the studio -- and leave no error to explain it.
        Opt in with SIANGTTS_HIFI_CLONING=true, knowing tags stop working.
        """
        txt = ref_file.with_suffix(".txt")
        if not txt.exists():
            return None
        if not settings.siangtts_hifi_cloning:
            print(
                f"[SiangTTS] Ignoring transcript {txt.name}: Hi-Fi cloning would "
                f"disable style tags. Set SIANGTTS_HIFI_CLONING=true to use it.",
                file=sys.stderr,
            )
            return None
        content = txt.read_text(encoding="utf-8").strip()
        return content or None

    def _ref_files(self) -> List[Path]:
        return [p for p in sorted(self.ref_dir.iterdir()) if p.suffix.lower() in AUDIO_EXTS]

    def init_speakers(self) -> None:
        """Scan ref/, precomputing and caching prompt latents."""
        try:
            synth = self.get_synthesizer()
        except SynthesizerUnavailable as e:
            print(f"[SiangTTS] Skipping speaker init: {e}", file=sys.stderr)
            return

        for ref_file in self._ref_files():
            sid = ref_file.stem
            cache_path = self.cache_dir / f"{sid}.pt"
            if cache_path.exists() and cache_path.stat().st_mtime >= ref_file.stat().st_mtime:
                try:
                    self._voices[sid] = synth.load_voice(cache_path)
                    continue
                except Exception as e:
                    print(f"[SiangTTS] Stale cache for '{sid}' ({e}); rebuilding.", file=sys.stderr)
            try:
                cache = synth.build_voice(str(ref_file), self._transcript_for(ref_file))
                synth.save_voice(cache, cache_path)
                self._voices[sid] = cache
            except Exception as ex:
                print(f"[SiangTTS] WARNING: could not cache voice '{sid}': {ex}", file=sys.stderr)

    def list_speakers(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": f.stem,
                "name": f.stem.replace("_", " ").title(),
                "filename": f.name,
                "cached": (self.cache_dir / f"{f.stem}.pt").exists(),
            }
            for f in self._ref_files()
        ]

    def register_speaker(self, speaker_id: str, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        clean_id = "".join(c for c in speaker_id.strip().lower() if c.isalnum() or c in ("-", "_"))
        if not clean_id:
            clean_id = "custom_speaker"

        ext = Path(filename).suffix.lower()
        if ext not in AUDIO_EXTS:
            ext = ".wav"

        ref_path = self.ref_dir / f"{clean_id}{ext}"
        ref_path.write_bytes(audio_bytes)

        cache_path = self.cache_dir / f"{clean_id}.pt"
        try:
            synth = self.get_synthesizer()
            cache = synth.build_voice(str(ref_path), self._transcript_for(ref_path))
            synth.save_voice(cache, cache_path)
            self._voices[clean_id] = cache
        except Exception as e:
            print(f"[SiangTTS] WARNING: failed to cache latent for {clean_id}: {e}", file=sys.stderr)

        return {
            "id": clean_id,
            "name": clean_id.replace("_", " ").title(),
            "filename": ref_path.name,
            "cached": cache_path.exists(),
        }

    def delete_speaker(self, speaker_id: str) -> bool:
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

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    def _resolve_voice(
        self,
        speaker_id: Optional[str],
        ref_audio_bytes: Optional[bytes],
        ref_filename: Optional[str],
    ) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        """Return (prompt_cache, ref_audio_path, temp_path_to_clean_up)."""
        if ref_audio_bytes:
            ext = Path(ref_filename or "upload.wav").suffix.lower()
            if ext not in AUDIO_EXTS:
                ext = ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(ref_audio_bytes)
                temp_path = tf.name
            return None, temp_path, temp_path

        if speaker_id:
            if speaker_id in self._voices:
                return self._voices[speaker_id], None, None
            for cand in self.ref_dir.glob(f"{speaker_id}.*"):
                if cand.suffix.lower() in AUDIO_EXTS:
                    return None, str(cand), None

        return None, None, None

    def _clone_voice_from_audio(self, synth: Any, wav: Any, sample_rate: int) -> Any:
        """Encode already-generated audio into a timbre-only prompt cache.

        Lets a run of chunks keep one voice when the caller pinned no speaker.
        Returns None if cloning fails -- a drifting voice is better than a failed
        request, and the caller simply carries on unconditioned.

        Deliberately clones in *reference* mode (no transcript). Passing one would
        select VoxCPM2's continuation mode, which reproduces every vocal nuance of
        the source -- including its emotion. That made chunk 2 inherit chunk 1's
        mood, collapsing [sad] -> [happy] to a measured -4.9 Hz median-F0 change
        where a pinned speaker gets +59.5 Hz. Timbre is what we want to carry over;
        prosody must stay free for the next chunk's style tag to steer.
        """
        import soundfile as sf

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name
            sf.write(tmp_path, wav, sample_rate, format="WAV", subtype="PCM_16")
            return synth.build_voice(tmp_path)
        except Exception as e:
            print(
                f"[SiangTTS] Could not clone first chunk for voice consistency: {e}",
                file=sys.stderr,
            )
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _voice_from_path(self, synth: Any, path: str) -> Any:
        """Encode a reference clip once so every piece shares the same conditioning.

        Returns None on failure, leaving the caller to fall back to per-piece
        reference passing -- slower, but not a failed request.
        """
        try:
            return synth.build_voice(path)
        except Exception as e:
            print(f"[SiangTTS] Could not pre-encode reference '{path}': {e}", file=sys.stderr)
            return None

    # Filename for the auto seed voice. Leading underscore keeps it out of the
    # speaker listing, which enumerates ref/ rather than the cache.
    SEED_VOICE_FILE = "_auto_seed.pt"

    def _warn_if_instructions_are_dead(
        self, prompt_cache: Any, planned: Sequence[Tuple[int, str, bool]]
    ) -> None:
        """Say so when the voice in hand cannot honour the style tags being sent.

        A prompt cache carrying a transcript puts VoxCPM2 in ultimate-cloning mode,
        where the leading parenthetical is ignored and the prompt clip's own delivery
        wins. The audio still arrives, just flat, so nothing else in the pipeline
        would ever mention it.
        """
        if self._hifi_warned or not isinstance(prompt_cache, dict):
            return
        if prompt_cache.get("mode") not in ("continuation", "ref_continuation"):
            return
        if not any(_LEADING_STYLE_RE.match(text) for _, text, _ in planned):
            return
        self._hifi_warned = True
        print(
            "[SiangTTS] WARNING: this voice was built in ultimate-cloning mode, "
            "which ignores style instructions -- the emotion tags in this request "
            "will not be heard. Rebuild the voice without its transcript, or unset "
            "SIANGTTS_HIFI_CLONING.",
            file=sys.stderr,
        )

    def _build_seed_voice(
        self, synth: Any, sample_rate: int, cfg_value: float, inference_timesteps: int
    ) -> Any:
        """Return the neutral seed voice every unpinned chunk is conditioned on.

        Built once and then reused, from memory within a process and from
        ``voice_cache/_auto_seed.pt`` across restarts. Regenerating it per request
        made every request a slightly different speaker -- two /synthesize calls on
        the same text came back in different voices -- and spent an extra generation
        each time to do it.

        Generated at the service defaults rather than the caller's cfg/timesteps, so
        who the speaker is does not depend on per-request tuning knobs.

        Returns None on failure -- an inconsistent voice beats a failed request.
        """
        if self._seed_voice is not None:
            return self._seed_voice
        if self._seed_voice_failed:
            return None

        seed_text = (settings.siangtts_voice_seed_text or "").strip()
        if not seed_text:
            return None

        cache_path = self.cache_dir / self.SEED_VOICE_FILE
        if cache_path.exists():
            try:
                self._seed_voice = synth.load_voice(cache_path)
                return self._seed_voice
            except Exception as e:
                print(f"[SiangTTS] Stale seed voice ({e}); rebuilding.", file=sys.stderr)

        try:
            seed_wav = synth.synth(text=seed_text)
            voice = self._clone_voice_from_audio(synth, seed_wav, sample_rate)
        except Exception as e:
            print(f"[SiangTTS] Seed voice generation failed: {e}", file=sys.stderr)
            self._seed_voice_failed = True
            return None

        if voice is None:
            self._seed_voice_failed = True
            return None

        try:
            synth.save_voice(voice, cache_path)
        except Exception as e:
            # Worth keeping in memory even if it could not be persisted.
            print(f"[SiangTTS] Could not persist seed voice: {e}", file=sys.stderr)

        self._seed_voice = voice
        return voice

    def synthesize(
        self,
        text: str,
        *,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        lora_mode: Optional[str] = "on",
    ) -> bytes:
        """Synthesize a single utterance. Returns WAV bytes."""
        return self.synthesize_many(
            [text],
            speaker_id=speaker_id,
            ref_audio_bytes=ref_audio_bytes,
            ref_filename=ref_filename,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            lora_mode=lora_mode,
        )

    def render_chunks(
        self,
        texts: Sequence[str],
        *,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        tones: Optional[Sequence[Optional[str]]] = None,
        breaks: Optional[Sequence[bool]] = None,
        lora_mode: Optional[str] = "on",
    ) -> Tuple[List[Any], int]:
        """Synthesize each chunk against one voice; return raw audio and sample rate.

        Split out from ``synthesize_many`` so a single generation can be assembled
        more than one way. Sampling is not deterministic, so an A/B that generated
        twice would be comparing two different takes rather than two treatments of
        the same one.
        """
        import numpy as np

        from app.services.audio_post import Chunk

        def _tone_at(idx: int) -> Optional[str]:
            return tones[idx] if tones is not None and idx < len(tones) else None

        def _break_at(idx: int) -> bool:
            return bool(breaks[idx]) if breaks is not None and idx < len(breaks) else False

        # Expand in lockstep so tones/breaks stay aligned with the surviving text.
        # A chunk past the drift budget becomes several pieces that keep its tone
        # and its voice; only the first inherits the chunk's own leading pause.
        planned: List[Tuple[int, str, bool]] = []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                continue
            for j, piece in enumerate(split_for_synthesis(t)):
                planned.append((i, piece.text, piece.paragraph_seam if j else _break_at(i)))
        if not planned:
            raise ValueError("No text to synthesize")

        synth = self.get_synthesizer()
        sample_rate = getattr(synth, "sample_rate", MOCK_SAMPLE_RATE)

        # Apply LoRA strength based on lora_mode if loaded
        tts_model = getattr(synth, "tts_model", None)
        if tts_model is not None and getattr(synth, "lora_loaded", False):
            if lora_mode == "off":
                set_lora_strength(tts_model, 0.0, 0.0)
            elif lora_mode == "legacy":
                set_lora_strength(tts_model, 2.0, 2.0)
            else:
                set_lora_strength(
                    tts_model,
                    settings.siangtts_lora_lm_scale,
                    settings.siangtts_lora_dit_scale,
                )

        prompt_cache, ref_audio_path, temp_ref_path = self._resolve_voice(
            speaker_id, ref_audio_bytes, ref_filename
        )

        try:
            # Nothing pinned the voice, so each chunk would otherwise come back a
            # different speaker. Mint one neutral seed voice up front and condition
            # every chunk on it. Seeding from a *neutral* line rather than from
            # chunk 1 is what keeps the style tags independent: cloning chunk 1
            # carried its emotion into chunk 2 and flattened the contrast.
            if (
                settings.siangtts_auto_voice_consistency
                and len(planned) > 1
                and prompt_cache is None
                and ref_audio_path is None
            ):
                prompt_cache = self._build_seed_voice(
                    synth, sample_rate, cfg_value, inference_timesteps
                )

            # Encode the reference once and share it, rather than letting generate()
            # re-encode the same clip for every piece. Identical conditioning either
            # way -- both take VoxCPM2's reference (timbre-only) mode -- but shared
            # so no piece can be conditioned on a slightly different encode.
            if ref_audio_path is not None and len(planned) > 1:
                shared = self._voice_from_path(synth, ref_audio_path)
                if shared is not None:
                    prompt_cache, ref_audio_path = shared, None

            self._warn_if_instructions_are_dead(prompt_cache, planned)

            rendered: List[Chunk] = []
            for src_idx, chunk, break_before in planned:
                wav = np.asarray(
                    synth.synth(
                        text=chunk,
                        ref_audio=ref_audio_path,
                        prompt_cache=prompt_cache,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                    ),
                    dtype="float32",
                )
                rendered.append(
                    Chunk(
                        audio=wav,
                        tone=_tone_at(src_idx),
                        break_before=break_before,
                        text_len=spoken_len(chunk),
                    )
                )

            return rendered, sample_rate
        finally:
            if temp_ref_path and os.path.exists(temp_ref_path):
                try:
                    os.remove(temp_ref_path)
                except Exception:
                    pass

    def synthesize_many(
        self,
        texts: Sequence[str],
        *,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        tones: Optional[Sequence[Optional[str]]] = None,
        breaks: Optional[Sequence[bool]] = None,
        post_process: bool = True,
        lora_mode: Optional[str] = "on",
    ) -> bytes:
        """Synthesize several chunks against one voice and join them into one take.

        Each chunk carries its own leading style instruction, which is how per-segment
        emotion is expressed -- VoxCPM2 only honours a style parenthetical at the very
        start of the text it is given.

        ``tones`` and ``breaks`` run parallel to ``texts`` and tell the assembler how
        loud each chunk should sit and how much silence belongs in front of it. Without
        them the chunks are still trimmed, faded and levelled, just without the
        per-emotion offsets. ``post_process=False`` returns the bare concatenation,
        which is what tools/ab_gen.py renders as the "before" take.
        """
        import soundfile as sf

        from app.services.audio_post import assemble, butt_join

        rendered, sample_rate = self.render_chunks(
            texts,
            speaker_id=speaker_id,
            ref_audio_bytes=ref_audio_bytes,
            ref_filename=ref_filename,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            tones=tones,
            breaks=breaks,
            lora_mode=lora_mode,
        )

        audio = assemble(rendered, sample_rate) if post_process else butt_join(rendered, sample_rate)

        out_buf = io.BytesIO()
        sf.write(out_buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return out_buf.getvalue()


siangtts_service = SiangTTSService()
