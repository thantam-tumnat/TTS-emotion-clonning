"""VoxCPM2 (emotion cloned from a donor) + SeedVC (timbre swap) generation service.

The :8011 studio asks VoxCPM2 for emotion directly: it prefixes the text with a style
parenthetical and the LoRA's LM side reads it. That works, but the emotion it produces
is whatever the model infers from a word, and the same word does not land the same way
twice.

This pipeline takes the emotion from a recording instead:

    (angry) + script
        │  pick the donor clip for "angry" from ONE actor's set
        │      ref/emotions/<set>/angry_1.wav + angry_1.txt
        ▼
    VoxCPM2 (:8020 -> :8021) in continuation mode -- clip AND transcript, which is
    the mode that reproduces the prompt clip's own delivery and *ignores* control
    instructions. The script comes back spoken with the donor's anger, in the
    donor's voice.
        ▼
    SeedVC (:8022) with f0_condition -- swaps the timbre to the user's reference
    voice while keeping the pitch contour that carries the emotion.
        ▼
    audio_post.assemble -> one WAV

So VoxCPM2's own emotion feature is unused here: the style parenthetical is stripped
before generation, because in continuation mode it would be read aloud rather than
obeyed.

Everything else -- annotation, segmentation, chunk splitting, assembly, the speaker
registry -- is shared with the sibling studio and reused as-is. Generation still goes
through the queue gateway on :8020, so this service adds no second copy of VoxCPM2.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import settings
from app.services.siangtts_service import (
    AUDIO_EXTS,
    _LEADING_STYLE_RE,
    siangtts_service,
    spoken_len,
    split_for_synthesis,
)

# The five emotions the thai-ser donor sets provide. Emotion is a recording here, so
# this list is a fact about the donor library, not a vocabulary choice -- adding a
# sixth means recording a sixth clip for every actor.
SUPPORTED_EMOTIONS = ("neutral", "angry", "happy", "sad", "frustrated")

# The annotator's vocabulary is ten tones (app/models.py Tone), so most of it has to
# land on one of the five. The alternative -- rejecting the other five -- would make
# auto-annotate fail on ordinary Thai script, since "excited" and "calm" are among
# the tones it reaches for most.
#
# Each mapping is to the nearest donor by arousal *and* valence, which is what the
# donor actually carries: a high-arousal negative tone reads as the frustrated clip,
# a low-arousal one as sad. The mapping is deliberately visible -- /api/donors
# reports it, and every render logs which donor a chunk used -- because a silently
# substituted emotion is otherwise indistinguishable from one the model got right.
EMOTION_MAP: Dict[str, str] = {
    "neutral": "neutral",
    "calm": "neutral",          # low arousal, neutral valence
    "sad": "sad",
    "tired": "sad",             # low arousal, negative
    "happy": "happy",
    "excited": "happy",         # high arousal, positive
    "angry": "angry",
    "frustrated": "frustrated",
    "nervous": "frustrated",    # high arousal, negative, not aggressive
    "scared": "frustrated",     # ditto -- the tense donor, not the shouting one
    "sarcastic": "frustrated",  # the donor's edge is the closest thing available
}

# SeedVC's f0-conditioned models run at 44.1 kHz; everything after conversion is at
# that rate, whatever VoxCPM2 generated at.
SEEDVC_SAMPLE_RATE = 44100


class VoxCPMVCUnavailable(RuntimeError):
    """The GPU service or the SeedVC worker could not be reached."""


def strip_instruction(text: str) -> str:
    """Drop the leading style parenthetical.

    In continuation mode VoxCPM2 does not treat it as direction, so leaving it in
    means "(โกรธ)" is *spoken*. The emotion is the donor's job now.
    """
    return _LEADING_STYLE_RE.sub("", text or "").strip()


class VoxCPMVCService:
    def __init__(
        self,
        donor_dir: str | Path | None = None,
        seedvc_url: Optional[str] = None,
    ) -> None:
        self.donor_dir = Path(donor_dir or settings.emotion_donor_dir)
        self.seedvc_url = (seedvc_url or settings.seedvc_url).rstrip("/")
        # (set_id, emotion) -> voice handle held by the GPU service. A donor clip is
        # the same bytes on every request, so encoding it once per process is the
        # difference between one upload and one per chunk.
        self._donor_handles: Dict[Tuple[str, str], str] = {}
        self._manifest: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Donor sets
    # ------------------------------------------------------------------ #

    def manifest(self) -> dict:
        if self._manifest is None:
            path = self.donor_dir / "donors_manifest.json"
            try:
                self._manifest = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._manifest = {"sets": []}
        return self._manifest

    def list_donor_sets(self) -> List[Dict[str, Any]]:
        """Donor sets on disk, each with the emotions it actually has clips for.

        The manifest is metadata, not truth -- a set is only usable if the clip and
        its transcript are both present, so the directory is what gets listed.
        """
        by_id = {s.get("id"): s for s in self.manifest().get("sets", []) if s.get("id")}
        out: List[Dict[str, Any]] = []

        if not self.donor_dir.exists():
            return out

        for d in sorted(p for p in self.donor_dir.iterdir() if p.is_dir()):
            emotions = {}
            for emo in SUPPORTED_EMOTIONS:
                wav = d / f"{emo}_1.wav"
                txt = wav.with_suffix(".txt")
                if wav.is_file() and txt.is_file():
                    emotions[emo] = {
                        "file": wav.name,
                        "text": txt.read_text(encoding="utf-8").strip(),
                    }
            if not emotions:
                continue
            meta = by_id.get(d.name, {})
            out.append({
                "id": d.name,
                "gender": meta.get("gender") or ("male" if d.name.startswith("male") else "female"),
                "actor_id": meta.get("actor_id"),
                "mean_agreement": meta.get("mean_agreement"),
                "emotions": emotions,
                "complete": len(emotions) == len(SUPPORTED_EMOTIONS),
            })
        return out

    def resolve_donor_set(self, donor_set: Optional[str], gender: Optional[str] = None) -> str:
        """Pick the set to clone emotion from.

        Preference: what the request asked for, then the configured default, then the
        first complete set for the requested gender. One set for the whole take
        matters -- mixing actors between chunks changes the speaker mid-sentence in a
        way SeedVC cannot fully hide, because it converts timbre, not accent or pace.
        """
        sets = self.list_donor_sets()
        if not sets:
            raise FileNotFoundError(
                f"No donor sets found under {self.donor_dir}. Build them with "
                f"tools/build_donor_sets.py."
            )
        by_id = {s["id"]: s for s in sets}

        for candidate in (donor_set, settings.default_donor_set):
            if candidate and candidate in by_id:
                return candidate
            if candidate:
                raise FileNotFoundError(
                    f"Donor set '{candidate}' not found. Available: "
                    f"{', '.join(sorted(by_id))}"
                )

        g = (gender or settings.default_gender or "female").strip().lower()
        want = "male" if g.startswith("m") else "female"

        # Best first: a complete set whose five clips are one actor (that is what
        # being in the manifest means) and whose emotion labels the thai-ser raters
        # agreed on most. A set assembled from different actors would change the
        # speaker between emotions, and SeedVC converts timbre only -- the accent and
        # pacing of the second actor would still come through.
        def rank(s: Dict[str, Any]) -> tuple:
            return (
                0 if s["gender"] == want else 1,
                0 if s["complete"] else 1,
                0 if s.get("actor_id") else 1,
                -float(s.get("mean_agreement") or 0),
                s["id"],
            )

        return sorted(sets, key=rank)[0]["id"]

    def donor_clip(self, donor_set: str, emotion: str) -> Tuple[Path, str]:
        """The (clip, transcript) pair for one emotion of one set."""
        wav = self.donor_dir / donor_set / f"{emotion}_1.wav"
        txt = wav.with_suffix(".txt")
        if not wav.is_file() or not txt.is_file():
            raise FileNotFoundError(
                f"Donor clip for '{emotion}' missing from set '{donor_set}' "
                f"(expected {wav.name} + {txt.name} in {wav.parent})"
            )
        transcript = txt.read_text(encoding="utf-8").strip()
        if not transcript:
            raise ValueError(
                f"Donor transcript {txt} is empty. Without it VoxCPM2 falls back to "
                f"timbre-only cloning, which carries no emotion."
            )
        return wav, transcript

    @staticmethod
    def validate_emotion(raw_tone: Optional[str]) -> str:
        """Annotator tone -> donor emotion, via EMOTION_MAP.

        A tone outside the map is an error rather than a fallback to neutral: falling
        back would return a flat take that looks like a success.
        """
        if raw_tone is None or not str(raw_tone).strip():
            return "neutral"
        clean = str(raw_tone).strip().lower()
        if clean in EMOTION_MAP:
            return EMOTION_MAP[clean]
        raise ValueError(
            f"Unsupported emotion '{raw_tone}'. This pipeline clones emotion from donor "
            f"recordings; known tones are: {', '.join(sorted(EMOTION_MAP))}"
        )

    # ------------------------------------------------------------------ #
    # Engine + voice plumbing (shared with the sibling studio)
    # ------------------------------------------------------------------ #

    def _synth(self) -> Any:
        """The queue-gateway synthesizer, connected."""
        return siangtts_service.get_synthesizer()

    def _donor_handle(self, synth: Any, donor_set: str, emotion: str) -> Any:
        """Encode a donor clip *with* its transcript and cache the handle.

        The transcript is the whole point: passing one selects VoxCPM2's continuation
        mode, where the prompt clip's delivery is reproduced. Passing only the clip
        would give timbre-only cloning and a flat, neutral read.
        """
        key = (donor_set, emotion)
        cached = self._donor_handles.get(key)
        if cached:
            return cached

        wav, transcript = self.donor_clip(donor_set, emotion)
        handle = synth.build_voice(str(wav.resolve()), prompt_text=transcript)
        if isinstance(handle, str) and handle:
            self._donor_handles[key] = handle
        return handle

    def _target_voice_path(
        self,
        speaker_id: Optional[str],
        ref_audio_bytes: Optional[bytes],
        ref_filename: Optional[str],
    ) -> Tuple[Path, Optional[Path]]:
        """Resolve the SeedVC *target* — the voice the take should end up in.

        Returns (path, temp_path_to_delete). Unlike VoxCPM2, SeedVC needs a real file
        it can read, so a speaker that only exists on the GPU service is fetched back
        to a temp file rather than passed by name.
        """
        if ref_audio_bytes:
            ext = Path(ref_filename or "upload.wav").suffix.lower()
            if ext not in AUDIO_EXTS:
                ext = ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(ref_audio_bytes)
            tmp = Path(tf.name)
            return tmp, tmp

        if speaker_id:
            local = siangtts_service.get_speaker_audio_path(speaker_id)
            if local and local.is_file():
                return local, None

            remote = siangtts_service._remote()
            if remote is not None:
                fetched = remote.get_speaker_audio_bytes(speaker_id)
                if fetched:
                    data, _media, filename = fetched
                    ext = Path(filename or "ref.wav").suffix.lower()
                    if ext not in AUDIO_EXTS:
                        ext = ".wav"
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                        tf.write(data)
                    tmp = Path(tf.name)
                    return tmp, tmp

            raise FileNotFoundError(f"Speaker '{speaker_id}' has no reference audio to convert to")

        return self._default_target(), None

    def _default_target(self) -> Path:
        """A target voice for a request that pinned none.

        There is no "unpinned" option here the way there is on the :8011 pipeline:
        SeedVC converts *into* someone, so with nobody named the take would ship in
        the donor actor's voice -- a wrong-speaker result that sounds like a correct
        one. A stable house voice from ref/ is the safe answer, and unlike VoxCPM2's
        auto-seed it is the same voice on every request by construction.
        """
        ref_dir = Path(settings.siangtts_ref_dir)
        for name in ("default", "house", "female", "male"):
            for ext in AUDIO_EXTS:
                cand = ref_dir / f"{name}{ext}"
                if cand.is_file():
                    return cand

        if ref_dir.is_dir():
            clips = sorted(
                p for p in ref_dir.iterdir()
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS
            )
            if clips:
                return clips[0]

        raise FileNotFoundError(
            f"No target voice: nothing was pinned and {ref_dir} holds no reference "
            f"clips. Pass speaker_id, upload a clip, or put one in {ref_dir}."
        )

    # ------------------------------------------------------------------ #
    # SeedVC
    # ------------------------------------------------------------------ #

    def seedvc_health(self) -> Optional[dict]:
        import httpx

        try:
            res = httpx.get(f"{self.seedvc_url}/health", timeout=2.0)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass
        return None

    def _convert(self, source: Path, target: Path, output: Path) -> Path:
        import httpx

        payload = {
            "source": str(source.resolve()),
            "target": str(target.resolve()),
            "output": str(output.resolve()),
            "f0_condition": settings.seedvc_f0_condition,
            "auto_f0_adjust": settings.seedvc_auto_f0_adjust,
            "diffusion_steps": settings.seedvc_diffusion_steps,
            "semi_tone_shift": 0,
            "inference_cfg_rate": settings.seedvc_inference_cfg_rate,
        }
        try:
            res = httpx.post(
                f"{self.seedvc_url}/convert", json=payload, timeout=settings.seedvc_timeout
            )
        except Exception as e:                                            # noqa: BLE001
            raise VoxCPMVCUnavailable(
                f"SeedVC worker at {self.seedvc_url} is unreachable: {e}. Start it with "
                f"start_seedvc.bat (it runs in its own venv on :8022)."
            ) from e

        if res.status_code != 200:
            detail = res.text[:400]
            try:
                body = res.json()
                detail = body.get("error") or body.get("detail") or detail
            except Exception:
                pass
            raise VoxCPMVCUnavailable(f"SeedVC convert failed ({res.status_code}): {detail}")
        return output

    @staticmethod
    def _depeak(audio: Any) -> Any:
        """Pull an over-unity waveform back under the ceiling.

        VoxCPM2 can overshoot [-1, 1] on a loud emotion, and the WAV on the way to
        SeedVC is PCM_16 with no guard of its own, so the overshoot would arrive as
        hard clipping baked into the source of the conversion.
        """
        import numpy as np

        ceiling = float(settings.voxcpm_peak or 0)
        if ceiling <= 0:
            return audio
        arr = np.asarray(audio, dtype="float32")
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > ceiling:
            arr = arr * (ceiling / peak)
        return arr

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def render_chunks(
        self,
        texts: Sequence[str],
        *,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        tones: Optional[Sequence[Optional[str]]] = None,
        breaks: Optional[Sequence[bool]] = None,
        gender: Optional[str] = None,
        donor_set: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        lora_mode: Optional[str] = "on",
        pre_vc_out: Optional[List[Tuple[Any, int]]] = None,
        debug_out: Optional[List[dict]] = None,
    ) -> Tuple[List[Any], int]:
        """Generate every chunk and convert it to the target voice.

        Returns assembler-ready ``Chunk``s at SeedVC's rate.
        """
        import numpy as np
        import soundfile as sf

        from app.services.audio_post import Chunk

        def _tone_at(idx: int) -> Optional[str]:
            return tones[idx] if tones is not None and idx < len(tones) else None

        def _break_at(idx: int) -> bool:
            return bool(breaks[idx]) if breaks is not None and idx < len(breaks) else False

        # Expand long chunks in lockstep with their tone and leading pause, exactly as
        # the sibling studio does, then drop the style parenthetical from each piece:
        # the donor carries the emotion now, and a parenthetical left in continuation
        # mode is read out loud.
        planned: List[Tuple[int, str, bool, str]] = []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                continue
            emotion = self.validate_emotion(_tone_at(i))
            for j, piece in enumerate(split_for_synthesis(t)):
                body = strip_instruction(piece.text)
                if not body:
                    continue
                planned.append((i, body, piece.paragraph_seam if j else _break_at(i), emotion))
        if not planned:
            raise ValueError("No text to synthesize")

        chosen_set = self.resolve_donor_set(donor_set, gender=gender)
        target_wav, temp_target = self._target_voice_path(
            speaker_id, ref_audio_bytes, ref_filename
        )

        synth = self._synth()
        batch = getattr(synth, "render_batch", None)
        if batch is None:
            # An in-process engine (or the mock) generates one piece at a time. Wrap
            # it to the same shape so the rest of this method does not care which
            # engine it got.
            def batch(pieces, *, prompt_cache=None, cfg_value=2.5,
                      inference_timesteps=10, lora_mode="on"):
                rate = int(getattr(synth, "sample_rate", 48000) or 48000)
                return [
                    synth.synth(
                        text=piece,
                        prompt_cache=prompt_cache,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                        lora_mode=lora_mode,
                    )
                    for piece in pieces
                ], rate

        if settings.seedvc_required and self.seedvc_health() is None:
            raise VoxCPMVCUnavailable(
                f"SeedVC worker at {self.seedvc_url} is not responding. Without it the "
                f"take would come back in the donor's voice, not '{speaker_id or 'the uploaded clip'}'."
            )

        scratch = Path("scratch/voxcpm_vc")
        scratch.mkdir(parents=True, exist_ok=True)
        run_id = int(time.time() * 1000)

        # One job per emotion rather than per chunk: every piece of an emotion shares
        # that emotion's donor prompt cache, and the service can only guarantee that
        # if it sees them together. Grouping across the whole take (not just runs of
        # adjacent chunks) is safe here because SeedVC re-timbres everything to the
        # same target afterwards, so cross-group speaker drift cannot survive.
        groups: Dict[str, List[int]] = {}
        for idx, (_src, _body, _brk, emotion) in enumerate(planned):
            groups.setdefault(emotion, []).append(idx)

        generated: List[Optional[Any]] = [None] * len(planned)
        gen_rate = int(getattr(synth, "sample_rate", 48000) or 48000)

        try:
            for emotion, indices in groups.items():
                handle = self._donor_handle(synth, chosen_set, emotion)
                donor_wav, donor_txt = self.donor_clip(chosen_set, emotion)

                print(
                    f"[VoxCPM+VC] {emotion}: {len(indices)} piece(s) cloning "
                    f"{chosen_set}/{donor_wav.name}",
                    file=sys.stderr,
                )

                audios, gen_rate = batch(
                    [planned[i][1] for i in indices],
                    prompt_cache=handle,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    lora_mode=lora_mode,
                )
                if len(audios) != len(indices):
                    raise RuntimeError(
                        f"engine returned {len(audios)} pieces for {len(indices)} sent "
                        f"({emotion})"
                    )
                for slot, audio in zip(indices, audios):
                    generated[slot] = self._depeak(audio)

                if debug_out is not None:
                    debug_out.append({
                        "emotion": emotion,
                        "donor_set": chosen_set,
                        "donor_clip": donor_wav.name,
                        "donor_text": donor_txt,
                        "pieces": [planned[i][1] for i in indices],
                        "cfg_value": cfg_value,
                        "inference_timesteps": inference_timesteps,
                        "lora_mode": lora_mode,
                    })

            rendered: List[Chunk] = []
            for idx, (src_idx, body, break_before, emotion) in enumerate(planned):
                audio = generated[idx]
                if audio is None:
                    continue

                src_path = scratch / f"gen_{run_id}_{idx:03d}_{emotion}.wav"
                out_path = scratch / f"vc_{run_id}_{idx:03d}_{emotion}.wav"
                sf.write(str(src_path), np.asarray(audio, dtype="float32"), gen_rate,
                         format="WAV", subtype="PCM_16")

                if pre_vc_out is not None:
                    pre_vc_out.append((np.asarray(audio, dtype="float32"), gen_rate))

                try:
                    self._convert(src_path, target_wav, out_path)
                    converted, sr = sf.read(str(out_path), dtype="float32")
                finally:
                    src_path.unlink(missing_ok=True)
                    out_path.unlink(missing_ok=True)

                if converted.ndim > 1:
                    converted = converted.mean(axis=1)

                rendered.append(
                    Chunk(
                        audio=np.asarray(converted, dtype="float32"),
                        tone=_tone_at(src_idx),
                        break_before=break_before,
                        text_len=spoken_len(body),
                    )
                )

            if not rendered:
                raise ValueError("Nothing was rendered")
            return rendered, int(sr)

        finally:
            if temp_target is not None:
                try:
                    temp_target.unlink(missing_ok=True)
                except Exception:
                    pass

    def synthesize_many(
        self,
        texts: Sequence[str],
        *,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        tones: Optional[Sequence[Optional[str]]] = None,
        breaks: Optional[Sequence[bool]] = None,
        gender: Optional[str] = None,
        donor_set: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        post_process: bool = True,
        post_process_params: Optional[dict] = None,
        lora_mode: Optional[str] = "on",
        pre_vc_out: Optional[List[Tuple[Any, int]]] = None,
        debug_out: Optional[List[dict]] = None,
    ) -> bytes:
        """Render every chunk and join them into one take. Returns WAV bytes."""
        import soundfile as sf

        from app.services.audio_post import PostProcessConfig, assemble, butt_join

        rendered, sample_rate = self.render_chunks(
            texts,
            speaker_id=speaker_id,
            ref_audio_bytes=ref_audio_bytes,
            ref_filename=ref_filename,
            tones=tones,
            breaks=breaks,
            gender=gender,
            donor_set=donor_set,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            lora_mode=lora_mode,
            pre_vc_out=pre_vc_out,
            debug_out=debug_out,
        )

        if post_process:
            audio = assemble(rendered, sample_rate,
                             config=PostProcessConfig.from_dict(post_process_params))
        else:
            audio = butt_join(rendered, sample_rate)

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def synthesize(self, text: str, **kwargs: Any) -> bytes:
        return self.synthesize_many([text], **kwargs)

    def synthesize_variants(
        self,
        texts: Sequence[str],
        *,
        variants: Sequence[dict],
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        tones: Optional[Sequence[Optional[str]]] = None,
        breaks: Optional[Sequence[bool]] = None,
        gender: Optional[str] = None,
        donor_set: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
        lora_mode: Optional[str] = "on",
    ) -> Tuple[List[dict], int, List[Optional[str]]]:
        """One generation, assembled every way ``variants`` asks for.

        Generating once is what makes the comparison honest -- sampling is not
        deterministic, so two renders would differ by more than the assembly. It also
        matters more here than in the sibling studio, because a second render would
        pay for a second pass through SeedVC as well.
        """
        import numpy as np
        import soundfile as sf

        from app.services.audio_post import (
            PostProcessConfig,
            assemble_with_spans,
            butt_join_with_spans,
            voiced_rms,
        )

        rendered, sample_rate = self.render_chunks(
            texts,
            speaker_id=speaker_id,
            ref_audio_bytes=ref_audio_bytes,
            ref_filename=ref_filename,
            tones=tones,
            breaks=breaks,
            gender=gender,
            donor_set=donor_set,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            lora_mode=lora_mode,
        )

        usable = [c for c in rendered if c.audio is not None and np.asarray(c.audio).size]
        chunk_tones: List[Optional[str]] = [c.tone for c in usable]

        takes: List[dict] = []
        for spec in variants:
            if spec.get("post_process", True):
                config = PostProcessConfig.from_dict(spec.get("params"))
                audio, spans = assemble_with_spans(rendered, sample_rate, config=config)
            else:
                audio, spans = butt_join_with_spans(rendered, sample_rate)

            chunk_stats = []
            for i, (start, end) in enumerate(spans):
                seg = audio[int(start * sample_rate):int(end * sample_rate)]
                level = voiced_rms(seg, sample_rate)
                text_len = usable[i].text_len if i < len(usable) else 0
                dur = float(end - start)
                chunk_stats.append({
                    "tone": chunk_tones[i] if i < len(chunk_tones) else None,
                    "start_s": round(float(start), 3),
                    "end_s": round(float(end), 3),
                    "dur_s": round(dur, 3),
                    "text_len": int(text_len),
                    "pace_s_per_char": round(dur / text_len, 5) if text_len else None,
                    "level_db": round(float(20 * math.log10(level)), 2) if level > 1e-6 else None,
                })

            buf = io.BytesIO()
            sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
            takes.append({
                "id": spec.get("id"),
                "wav": buf.getvalue(),
                "dur_s": round(len(audio) / sample_rate, 3),
                "chunks": chunk_stats,
            })

        return takes, sample_rate, chunk_tones

    # ------------------------------------------------------------------ #

    @property
    def status(self) -> Dict[str, Any]:
        seedvc = self.seedvc_health()
        sets = self.list_donor_sets()
        return {
            "pipeline": "donor -> VoxCPM2 (continuation) -> SeedVC",
            "voxcpm": siangtts_service.status,
            "seedvc": {
                "url": self.seedvc_url,
                "reachable": seedvc is not None,
                "device": (seedvc or {}).get("device"),
                "f0_condition": settings.seedvc_f0_condition,
                "auto_f0_adjust": settings.seedvc_auto_f0_adjust,
                "diffusion_steps": settings.seedvc_diffusion_steps,
            },
            "donors": {
                "dir": str(self.donor_dir),
                "sets": len(sets),
                "complete_sets": sum(1 for s in sets if s["complete"]),
                "emotions": list(SUPPORTED_EMOTIONS),
                "tone_map": dict(EMOTION_MAP),
            },
            "sample_rate": SEEDVC_SAMPLE_RATE,
        }


voxcpm_vc_service = VoxCPMVCService()
