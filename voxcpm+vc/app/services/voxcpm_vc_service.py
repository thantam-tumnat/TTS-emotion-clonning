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
        # Resolved target-ref path -> voice handle, for the skip-VC neutral path
        # (zero-shot cloning of the target directly, no donor involved).
        self._target_handles: Dict[str, Any] = {}
        self._manifest: Optional[dict] = None
        # (set_id, emotion) -> the donor clip's own pace, seconds of voiced audio per
        # spoken character. Measured once per clip; it is a property of the recording.
        self._donor_pace: Dict[Tuple[str, str], Optional[float]] = {}

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

    def donor_sets_ui(self) -> List[Dict[str, Any]]:
        """``list_donor_sets`` reshaped for the studio/pipeline pickers.

        The canonical shape carries ``emotions`` as a dict keyed by emotion; the
        front-end wants a list of ``{id, transcript}`` plus a human name and a
        ``same_person`` flag, so this adapts without changing the canonical contract
        that ``/api/donors`` and ``status`` depend on.
        """
        out: List[Dict[str, Any]] = []
        for s in self.list_donor_sets():
            actor_id = s.get("actor_id")
            gender = s.get("gender")
            if actor_id:
                name = f"{gender.title()} · Actor {actor_id}" if gender else f"Actor {actor_id}"
            else:
                name = str(s["id"]).replace("_", " ").title()
            out.append({
                "id": s["id"],
                "name": name,
                "gender": gender,
                "actor_id": actor_id,
                "same_person": bool(actor_id),
                "mean_agreement": s.get("mean_agreement"),
                "complete": s.get("complete"),
                "emotions": [
                    {"id": emo, "transcript": (meta or {}).get("text", "")}
                    for emo, meta in (s.get("emotions") or {}).items()
                ],
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

    def _target_handle(self, synth: Any, target_wav: Path) -> Any:
        """Zero-shot voice handle for the target ref itself (no transcript).

        Used only by the skip-VC neutral path: VoxCPM2 clones the target's timbre
        directly with no donor and no continuation mode, so there is nothing for
        SeedVC to convert afterwards. Cached by resolved path like donor handles.
        """
        key = str(target_wav.resolve())
        cached = self._target_handles.get(key)
        if cached:
            return cached
        handle = synth.build_voice(key)
        if isinstance(handle, str) and handle:
            self._target_handles[key] = handle
        return handle

    @staticmethod
    def _resample_to(audio: Any, sr_from: int, sr_to: int) -> Any:
        """Resample ``audio`` from ``sr_from`` to ``sr_to``.

        Needed only for chunks that skip SeedVC: everything that goes through
        SeedVC comes back at SEEDVC_SAMPLE_RATE regardless of what VoxCPM2 rendered
        at, so a chunk that bypasses it has to be brought to that same rate by hand
        before assembly, or a mixed take (neutral + another emotion) would carry two
        different sample rates into one WAV.
        """
        import numpy as np

        arr = np.asarray(audio, dtype="float32")
        if sr_from == sr_to or arr.size == 0:
            return arr
        import torch
        import torchaudio

        t = torch.from_numpy(arr)
        out = torchaudio.functional.resample(t, sr_from, sr_to)
        return out.numpy().astype("float32")

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

    def _convert(
        self,
        source: Path,
        target: Path,
        output: Path,
        *,
        auto_f0_adjust: Optional[bool] = None,
        semi_tone_shift: int = 0,
    ) -> Path:
        """Convert ``source``'s timbre to ``target`` via the SeedVC worker.

        ``auto_f0_adjust`` / ``semi_tone_shift`` default to the server config; the
        F0-compare experiment overrides them per call so one VoxCPM2 generation can be
        converted several ways (see ``render_f0_compare``).
        """
        import httpx

        af0 = settings.seedvc_auto_f0_adjust if auto_f0_adjust is None else bool(auto_f0_adjust)
        payload = {
            "source": str(source.resolve()),
            "target": str(target.resolve()),
            "output": str(output.resolve()),
            "f0_condition": settings.seedvc_f0_condition,
            "auto_f0_adjust": af0,
            "diffusion_steps": settings.seedvc_diffusion_steps,
            "semi_tone_shift": int(semi_tone_shift),
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

    def _b_shift_to_target(self, chosen_set: str, target_wav: Path) -> int:
        """Semitone shift that moves the donor's NEUTRAL register onto the target's,
        so a converted take keeps the emotion's own pitch offset but sits in the
        target's register -- the F0-compare "B" treatment, computed for a real take.

        Anchors on each clip's *register* (a low percentile of voiced F0), not its
        median, so an expressive target/donor does not inflate the shift; clamped to
        F0_SHIFT_CLAMP_ST so it can never push a full octave. Returns 0 when either
        register cannot be measured (the take then behaves like mode A)."""
        target_reg = self._register_f0(Path(target_wav))
        donor_neutral_reg = None
        try:
            neutral_wav, _ = self.donor_clip(chosen_set, "neutral")
            donor_neutral_reg = self._register_f0(neutral_wav)
        except Exception:                                                # noqa: BLE001
            donor_neutral_reg = None
        if target_reg and donor_neutral_reg and donor_neutral_reg > 0:
            raw_shift = 12.0 * math.log2(target_reg / donor_neutral_reg)
            clamp = self.F0_SHIFT_CLAMP_ST
            return int(round(max(-clamp, min(clamp, raw_shift))))
        return 0

    def _f0_convert_kwargs(self, chosen_set: str, target_wav: Path) -> Dict[str, Any]:
        """SeedVC F0 handling for a real take, selected by settings.seedvc_f0_mode.

        Returns kwargs for ``_convert``. ``baseline`` returns nothing so ``_convert``
        keeps the server defaults; ``A``/``B`` turn auto-F0 off, and ``B`` adds the
        register-matching shift. Computed once per take (constant across its chunks)."""
        mode = (settings.seedvc_f0_mode or "baseline").strip().lower()
        if mode == "a":
            return {"auto_f0_adjust": False}
        if mode == "b":
            return {
                "auto_f0_adjust": False,
                "semi_tone_shift": self._b_shift_to_target(chosen_set, target_wav),
            }
        return {}

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
    # Donor-pace matching
    #
    # Continuation mode reproduces the donor's timbre and colour but renders new
    # text at VoxCPM2's own, more neutral speaking rate -- so an emotional donor
    # (a slow sad clip especially) comes back faster than the recording it cloned.
    # SeedVC is length-preserving, so the only place to fix the pace is here, on the
    # VoxCPM2 output before conversion. We stretch each piece (pitch untouched) to the
    # donor clip's measured pace: seconds of *voiced* audio per spoken character, so
    # pieces of different length stay comparable and leading/trailing padding does not
    # count. Micro-rhythm (where the pauses fall) is content-specific and not cloned;
    # this matches the overall rate, which is what "faster than the donor" is about.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _speech_pace(audio: Any, sr: int, char_len: int) -> Optional[float]:
        """Seconds of voiced audio per character, or None if unmeasurable."""
        import numpy as np

        from app.services.audio_post import trim_silence

        if not char_len or sr <= 0:
            return None
        arr = np.asarray(audio, dtype="float32")
        if arr.size == 0:
            return None
        voiced = trim_silence(arr, sr)
        secs = float(voiced.size) / sr
        if secs <= 0:
            return None
        return secs / char_len

    def _donor_pace_for(self, donor_set: str, emotion: str) -> Optional[float]:
        """The donor clip's own pace (sec/char), measured once and cached."""
        import soundfile as sf

        key = (donor_set, emotion)
        if key in self._donor_pace:
            return self._donor_pace[key]

        pace: Optional[float] = None
        try:
            wav, transcript = self.donor_clip(donor_set, emotion)
            y, sr = sf.read(str(wav), dtype="float32")
            if getattr(y, "ndim", 1) > 1:
                y = y.mean(axis=1)
            pace = self._speech_pace(y, int(sr), spoken_len(transcript))
        except Exception:
            pace = None
        self._donor_pace[key] = pace
        return pace

    def _match_to_donor_pace(
        self, audio: Any, gen_rate: int, body: str, donor_set: str, emotion: str
    ) -> Any:
        """Stretch one generated piece to the donor clip's pace (pitch untouched).

        A no-op when matching is disabled, the pace of either side cannot be measured,
        or the correction is negligible. The stretch is clamped so a big text/clip
        length mismatch cannot drag a piece into WSOLA artefacts.
        """
        import numpy as np

        from app.services.audio_post import time_stretch

        if not settings.voxcpm_vc_match_donor_pace:
            return audio

        char_len = spoken_len(body)
        gen_pace = self._speech_pace(audio, gen_rate, char_len)
        donor_pace = self._donor_pace_for(donor_set, emotion)
        if not gen_pace or not donor_pace or gen_pace <= 0:
            return audio

        ratio = donor_pace / gen_pace
        clamp = float(settings.voxcpm_vc_max_pace_stretch or 0)
        if clamp > 0:
            ratio = float(np.clip(ratio, 1.0 - clamp, 1.0 + clamp))
        if abs(ratio - 1.0) < 0.01:
            return audio
        return time_stretch(np.asarray(audio, dtype="float32"), gen_rate, ratio)

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
        raw_prompt: Optional[str] = None,
        client: Optional[str] = None,
        request_id: Optional[str] = None,
        pre_vc_out: Optional[List[Tuple[Any, int]]] = None,
        debug_out: Optional[List[dict]] = None,
        take_out: Optional[dict] = None,
    ) -> Tuple[List[Any], int]:
        """Generate every chunk and convert it to the target voice.

        Returns assembler-ready ``Chunk``s at SeedVC's rate. ``client`` labels the
        generation jobs on the queue gateway; callers that already show their own row
        on the dashboard (the webhook meta job) pass a ``*-internal`` name so the raw
        per-emotion jobs are collapsed out of the admin view.

        ``take_out`` is filled with the take-level facts this method resolves --
        which clip SeedVC converts *into*, which donor actor was drawn, how F0 was
        treated. ``debug_out`` covers the per-emotion half. Together they are what
        actually reached the model, as opposed to what the caller asked for, and the
        two are only the same when every field of the request was honoured.

        ``request_id`` ties every generation job this take submits to one row on
        that dashboard. One take becomes one job per emotion, so without it the
        operator sees three unrelated jobs and no way to tell which chunks belong
        together -- and the gateway cannot abandon the siblings when one of them
        hits CUDA OOM. A caller that already owns a dashboard row (the webhook)
        passes its own id so the generations attach to it; everyone else gets a
        generated one, which is still one card per request.
        """
        import uuid as _uuid

        request_id = request_id or f"req_{int(time.time() * 1000)}_{_uuid.uuid4().hex[:6]}"
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

        # SeedVC F0 treatment for the whole take (baseline / A / B), chosen in .env.
        # Constant across chunks -- the shift depends only on this donor + target -- so
        # compute it once here rather than per chunk in the convert loop below.
        f0_kw = self._f0_convert_kwargs(chosen_set, Path(target_wav))

        if take_out is not None:
            take_out.update({
                # The file, not the name: two reference directories can hold the same
                # voice_id, and "which clip did it actually read" is the question.
                "target_clip": str(Path(target_wav).resolve()),
                "target_from": (
                    "uploaded clip" if ref_audio_bytes
                    else speaker_id or "no speaker pinned — house voice from ref/"
                ),
                "donor_set": chosen_set,
                "gender_asked": gender,
                "skip_neutral_vc": bool(settings.voxcpm_vc_skip_neutral),
                "seedvc": {
                    "f0_mode": settings.seedvc_f0_mode,
                    "f0_condition": settings.seedvc_f0_condition,
                    "diffusion_steps": settings.seedvc_diffusion_steps,
                    "inference_cfg_rate": settings.seedvc_inference_cfg_rate,
                    "auto_f0_adjust": f0_kw.get("auto_f0_adjust", settings.seedvc_auto_f0_adjust),
                    "semi_tone_shift": f0_kw.get("semi_tone_shift", 0),
                },
            })

        synth = self._synth()
        batch = getattr(synth, "render_batch", None)
        if batch is None:
            # An in-process engine (or the mock) generates one piece at a time. Wrap
            # it to the same shape so the rest of this method does not care which
            # engine it got.
            def batch(pieces, *, prompt_cache=None, cfg_value=2.5,
                      inference_timesteps=10, lora_mode="on", raw_prompt=None,
                      client=None, parent_id=None):
                rate = int(getattr(synth, "sample_rate", 48000) or 48000)
                return [
                    synth.synth(
                        text=piece,
                        prompt_cache=prompt_cache,
                        cfg_value=cfg_value,
                        inference_timesteps=inference_timesteps,
                        lora_mode=lora_mode,
                        raw_prompt=raw_prompt,
                    )
                    for piece in pieces
                ], rate

        # One job per emotion rather than per chunk: every piece of an emotion shares
        # that emotion's donor prompt cache, and the service can only guarantee that
        # if it sees them together. Grouping across the whole take (not just runs of
        # adjacent chunks) is safe here because SeedVC re-timbres everything to the
        # same target afterwards, so cross-group speaker drift cannot survive.
        groups: Dict[str, List[int]] = {}
        for idx, (_src, _body, _brk, emotion) in enumerate(planned):
            groups.setdefault(emotion, []).append(idx)

        # SeedVC is only needed for groups that actually convert through it -- a take
        # that is all-neutral with the skip enabled never touches the worker, and
        # should not fail just because it happens to be down.
        needs_seedvc = any(
            not (emotion == "neutral" and settings.voxcpm_vc_skip_neutral)
            for emotion in groups
        )
        if needs_seedvc and settings.seedvc_required and self.seedvc_health() is None:
            raise VoxCPMVCUnavailable(
                f"SeedVC worker at {self.seedvc_url} is not responding. Without it the "
                f"take would come back in the donor's voice, not '{speaker_id or 'the uploaded clip'}'."
            )

        scratch = Path("scratch/voxcpm_vc")
        scratch.mkdir(parents=True, exist_ok=True)
        run_id = int(time.time() * 1000)

        generated: List[Optional[Any]] = [None] * len(planned)
        gen_rate = int(getattr(synth, "sample_rate", 48000) or 48000)

        try:
            for emotion, indices in groups.items():
                skip_vc = emotion == "neutral" and settings.voxcpm_vc_skip_neutral

                if skip_vc:
                    handle = self._target_handle(synth, Path(target_wav))
                    print(
                        f"[VoxCPM+VC] neutral: {len(indices)} piece(s) cloning target "
                        f"ref directly (VC skipped)",
                        file=sys.stderr,
                    )
                else:
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
                    raw_prompt=raw_prompt or (" ".join(texts) if texts else None),
                    client=client,
                    parent_id=request_id,
                )
                if len(audios) != len(indices):
                    raise RuntimeError(
                        f"engine returned {len(audios)} pieces for {len(indices)} sent "
                        f"({emotion})"
                    )
                for slot, audio in zip(indices, audios):
                    if skip_vc:
                        # No donor was cloned, so there is no donor pace to match --
                        # the target ref's own natural pace is what should come out.
                        generated[slot] = self._depeak(audio)
                    else:
                        generated[slot] = self._match_to_donor_pace(
                            self._depeak(audio), gen_rate, planned[slot][1], chosen_set, emotion
                        )

                if debug_out is not None:
                    debug_out.append({
                        "emotion": emotion,
                        # "" not None: main.py puts this straight into a response
                        # header, and a None header value is a 500 (no .encode()).
                        "donor_set": "" if skip_vc else chosen_set,
                        "donor_clip": None if skip_vc else donor_wav.name,
                        "donor_text": None if skip_vc else donor_txt,
                        "skip_vc": skip_vc,
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

                skip_vc = emotion == "neutral" and settings.voxcpm_vc_skip_neutral

                if pre_vc_out is not None:
                    pre_vc_out.append((np.asarray(audio, dtype="float32"), gen_rate))

                if skip_vc:
                    # Already the target's own timbre -- just bring it to the rate
                    # every SeedVC-converted chunk lands at, so a take mixing neutral
                    # with another emotion assembles at one consistent sample rate.
                    converted = self._resample_to(audio, gen_rate, SEEDVC_SAMPLE_RATE)
                    sr = SEEDVC_SAMPLE_RATE
                else:
                    src_path = scratch / f"gen_{run_id}_{idx:03d}_{emotion}.wav"
                    out_path = scratch / f"vc_{run_id}_{idx:03d}_{emotion}.wav"
                    sf.write(str(src_path), np.asarray(audio, dtype="float32"), gen_rate,
                             format="WAV", subtype="PCM_16")
                    try:
                        self._convert(src_path, target_wav, out_path, **f0_kw)
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

    def _post_params_for_assembly(self, params: Optional[dict]) -> Optional[dict]:
        """Post-process params with the generic rate pass turned off when we already
        paced each chunk to its donor.

        ``_match_rate`` re-times chunks toward the coarse ``TONE_DURATION_RATIO`` table
        (sad = 1.035x), anchored on the take's neutral chunks. Left on, a mixed take
        would speed the donor-paced sad chunks *back up* toward that generic ratio,
        partially undoing the donor match. A single-tone take is unaffected either way
        (its rate pass self-cancels), but turning it off keeps mixed takes honest. A
        caller that set ``match_rate`` explicitly wins.
        """
        if not settings.voxcpm_vc_match_donor_pace:
            return params
        if params and "match_rate" in params:
            return params
        out = dict(params or {})
        out["match_rate"] = False
        return out

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
        raw_prompt: Optional[str] = None,
        client: Optional[str] = None,
        request_id: Optional[str] = None,
        pre_vc_out: Optional[List[Tuple[Any, int]]] = None,
        debug_out: Optional[List[dict]] = None,
        take_out: Optional[dict] = None,
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
            raw_prompt=raw_prompt,
            client=client,
            request_id=request_id,
            pre_vc_out=pre_vc_out,
            debug_out=debug_out,
            take_out=take_out,
        )

        if post_process:
            audio = assemble(rendered, sample_rate,
                             config=PostProcessConfig.from_dict(
                                 self._post_params_for_assembly(post_process_params)))
        else:
            audio = butt_join(rendered, sample_rate)

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def synthesize(self, text: str, **kwargs: Any) -> bytes:
        raw = kwargs.pop("raw_prompt", text)
        return self.synthesize_many([text], raw_prompt=raw, **kwargs)

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
        raw_prompt: Optional[str] = None,
        pre_vc_sink: Optional[dict] = None,
        debug_sink: Optional[dict] = None,
    ) -> Tuple[List[dict], int, List[Optional[str]]]:
        """One generation, assembled every way ``variants`` asks for.

        Generating once is what makes the comparison honest -- sampling is not
        deterministic, so two renders would differ by more than the assembly. It also
        matters more here than in the sibling studio, because a second render would
        pay for a second pass through SeedVC as well.

        When a caller passes ``pre_vc_sink`` it gets back one joined clip of the
        VoxCPM2 output *before* SeedVC ({"wav": bytes, "sr": int}); ``debug_sink`` is
        filled with the raw model inputs ({"chunks": [...]}). Both describe the single
        shared generation, so they are produced once regardless of variant count.
        """
        import numpy as np
        import soundfile as sf

        from app.services.audio_post import (
            PostProcessConfig,
            assemble_with_spans,
            butt_join_with_spans,
            voiced_rms,
        )

        pre_vc_chunks: Optional[List[Tuple[Any, int]]] = [] if pre_vc_sink is not None else None
        debug_chunks: Optional[List[dict]] = [] if debug_sink is not None else None

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
            raw_prompt=raw_prompt,
            pre_vc_out=pre_vc_chunks,
            debug_out=debug_chunks,
        )

        if debug_sink is not None:
            debug_sink["chunks"] = debug_chunks or []

        if pre_vc_sink is not None and pre_vc_chunks:
            pre_sr = pre_vc_chunks[0][1]
            gap = np.zeros(int(0.06 * pre_sr), dtype=np.float32)
            pieces: List[Any] = []
            for idx, (arr, _asr) in enumerate(pre_vc_chunks):
                if idx:
                    pieces.append(gap)
                pieces.append(np.asarray(arr, dtype=np.float32))
            pre_audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
            pre_buf = io.BytesIO()
            sf.write(pre_buf, pre_audio, pre_sr, format="WAV", subtype="PCM_16")
            pre_vc_sink["wav"] = pre_buf.getvalue()
            pre_vc_sink["sr"] = int(pre_sr)
            pre_vc_sink["dur_s"] = round(len(pre_audio) / pre_sr, 3)

        usable = [c for c in rendered if c.audio is not None and np.asarray(c.audio).size]
        chunk_tones: List[Optional[str]] = [c.tone for c in usable]

        takes: List[dict] = []
        for spec in variants:
            if spec.get("post_process", True):
                config = PostProcessConfig.from_dict(
                    self._post_params_for_assembly(spec.get("params")))
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
    # Pipeline explorer + F0 compare (single-utterance, stage-by-stage)
    # ------------------------------------------------------------------ #

    def get_donor_clip_path(self, donor_set: str, emotion: str) -> Optional[Path]:
        wav = self.donor_dir / donor_set / f"{emotion}_1.wav"
        return wav if wav.is_file() else None

    def tuning_defaults(self) -> Dict[str, Any]:
        """Server default value + bounds for each VoxCPM2 knob (for the pipeline UI).

        VoxCPM2 exposes only cfg_value and inference_timesteps -- the F5 studio's
        sway/target_rms/silence knobs have no equivalent here, so they are absent.
        """
        return {
            "values": {
                "cfg_value": settings.voxcpm_cfg_value if hasattr(settings, "voxcpm_cfg_value") else 2.5,
                "inference_timesteps": settings.voxcpm_inference_timesteps
                if hasattr(settings, "voxcpm_inference_timesteps") else 10,
            },
            "specs": {
                "cfg_value": {"min": 1.0, "max": 10.0, "step": 0.1},
                "inference_timesteps": {"min": 4, "max": 50, "step": 1},
            },
        }

    def _generate_one(
        self,
        synth: Any,
        donor_set: str,
        emotion: str,
        body: str,
        *,
        cfg_value: float,
        inference_timesteps: int,
        lora_mode: Optional[str],
        raw_prompt: Optional[str] = None,
    ) -> Tuple[Any, int]:
        """One VoxCPM2 continuation-mode generation for a single piece of text.

        Returns (audio, rate). Shares the batch/single-engine shim render_chunks uses,
        so it works against the queue gateway and the in-process mock alike.
        """
        import numpy as np

        handle = self._donor_handle(synth, donor_set, emotion)
        batch = getattr(synth, "render_batch", None)
        if batch is None:
            rate = int(getattr(synth, "sample_rate", 48000) or 48000)
            audio = synth.synth(
                text=body,
                prompt_cache=handle,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                lora_mode=lora_mode,
                raw_prompt=raw_prompt or body,
            )
            paced = self._match_to_donor_pace(
                self._depeak(audio), rate, body, donor_set, emotion
            )
            return paced, rate

        audios, rate = batch(
            [body],
            prompt_cache=handle,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            lora_mode=lora_mode,
            raw_prompt=raw_prompt or body,
        )
        if not audios:
            raise RuntimeError("engine returned no audio")
        paced = self._match_to_donor_pace(
            self._depeak(audios[0]), int(rate), body, donor_set, emotion
        )
        return paced, int(rate)

    # B mode's shift may move F0 by at most this many semitones. A full octave
    # (the old +/-12) is almost never a natural register move and, with SeedVC's
    # formants held at the target, reads as chipmunk/metallic -- so B is clamped
    # tighter than baseline would ever need.
    F0_SHIFT_CLAMP_ST = 6

    # The register anchor is a LOW percentile of voiced F0, not the median. A clip the
    # user handed as the target may itself be expressive (an excited or shouted line),
    # which inflates its median and makes B over-shift. The p20 tracks the speaker's
    # baseline register instead of their peaks, so an expressive target no longer
    # drags the whole emotion up.
    F0_REGISTER_PERCENTILE = 20

    @staticmethod
    def _voiced_f0_candidates(wav_path: Path) -> Optional["Any"]:
        """Per-frame voiced F0 estimates (Hz) via autocorrelation, transcript-free.

        Returns a numpy array of the voiced frames' F0, or None when nothing voiced is
        measurable. Median and register anchors both derive from this one pass.
        """
        import numpy as np
        import soundfile as sf

        try:
            y, sr = sf.read(str(wav_path), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if y.size == 0 or sr <= 0:
                return None
            frame = int(sr * 0.030)
            hop = int(sr * 0.010)
            if len(y) < frame:
                return None
            rms = float(np.sqrt(np.mean(y.astype("float64") ** 2)))
            min_lag = max(1, int(sr / 450))
            max_lag = int(sr / 60)
            cands: List[float] = []
            for i in range(0, len(y) - frame, hop):
                fr = y[i:i + frame]
                if np.sqrt(np.mean(fr ** 2)) < max(rms * 0.2, 1e-4):
                    continue
                w = fr * np.hanning(len(fr))
                corr = np.correlate(w, w, mode="full")[len(w) - 1:]
                if len(corr) <= max_lag or corr[0] <= 0:
                    continue
                region = corr[min_lag:max_lag]
                if region.size == 0:
                    continue
                peak = int(np.argmax(region)) + min_lag
                if corr[peak] > 0.3 * corr[0]:
                    cands.append(float(sr / peak))
            if len(cands) < 3:
                return None
            return np.asarray(cands, dtype="float64")
        except Exception:
            return None

    @classmethod
    def _median_f0(cls, wav_path: Path) -> Optional[float]:
        """Median voiced F0 (Hz). Kept for diagnostics/reporting."""
        import numpy as np

        cands = cls._voiced_f0_candidates(wav_path)
        if cands is None:
            return None
        return float(np.median(cands))

    @classmethod
    def _register_f0(cls, wav_path: Path) -> Optional[float]:
        """The speaker's baseline register: a low percentile of voiced F0.

        This is what B anchors on, so an expressive target/donor clip (peaks high in
        pitch) does not inflate the constant shift the way the median would.
        """
        import numpy as np

        cands = cls._voiced_f0_candidates(wav_path)
        if cands is None:
            return None
        return float(np.percentile(cands, cls.F0_REGISTER_PERCENTILE))

    def render_trace(
        self,
        *,
        donor_set: str,
        emotion: str,
        run_dir: Path,
        text: Optional[str] = None,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        tuning: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run one utterance donor -> VoxCPM2 -> SeedVC, keeping every stage on disk.

        Returns the donor clip, the VoxCPM2 output (emotional speech still in the
        donor's timbre) and the SeedVC output (timbre swapped to the target), so the
        caller can offer playback of each step.
        """
        import numpy as np
        import shutil
        import soundfile as sf

        emotion = self.validate_emotion(emotion)
        donor_wav = self.get_donor_clip_path(donor_set, emotion)
        if donor_wav is None:
            raise FileNotFoundError(f"No donor clip for set '{donor_set}', emotion '{emotion}'")
        donor_txt_p = donor_wav.with_suffix(".txt")
        donor_txt = donor_txt_p.read_text(encoding="utf-8").strip() if donor_txt_p.exists() else ""

        target_wav, temp_target = self._target_voice_path(speaker_id, ref_audio_bytes, ref_filename)

        # Text to synthesize: the user's text, else the donor transcript as a default.
        gen_text_raw = (text or "").strip() or donor_txt
        body = strip_instruction(gen_text_raw)
        if not body:
            raise ValueError("No text to synthesize")

        tuning = tuning or {}
        cfg_value = float(tuning.get("cfg_value") if tuning.get("cfg_value") is not None else 2.5)
        inference_timesteps = int(
            tuning.get("inference_timesteps") if tuning.get("inference_timesteps") is not None else 10
        )

        if settings.seedvc_required and self.seedvc_health() is None:
            raise VoxCPMVCUnavailable(
                f"SeedVC worker at {self.seedvc_url} is not responding."
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        donor_out = run_dir / f"donor_{emotion}{donor_wav.suffix.lower()}"
        stage_a = run_dir / f"A_voxcpm_{emotion}.wav"
        stage_b = run_dir / f"B_vc_{emotion}.wav"

        try:
            shutil.copy2(donor_wav, donor_out)

            synth = self._synth()
            t0 = time.time()
            audio, gen_rate = self._generate_one(
                synth, donor_set, emotion, body,
                cfg_value=cfg_value, inference_timesteps=inference_timesteps, lora_mode="on",
                raw_prompt=gen_text_raw,
            )
            sf.write(str(stage_a), np.asarray(audio, dtype="float32"), gen_rate,
                     format="WAV", subtype="PCM_16")
            gen_secs = time.time() - t0

            t0 = time.time()
            self._convert(stage_a, Path(target_wav), stage_b)
            vc_secs = time.time() - t0
        finally:
            if temp_target is not None:
                temp_target.unlink(missing_ok=True)

        return {
            "emotion": emotion,
            "donor_set": donor_set,
            "gen_text": gen_text_raw,
            "donor_transcript": donor_txt,
            "target": speaker_id or (ref_filename or "upload" if ref_audio_bytes else None) or Path(target_wav).stem,
            "tuning": {"cfg_value": cfg_value, "inference_timesteps": inference_timesteps},
            "gen_secs": round(gen_secs, 1),
            "vc_secs": round(vc_secs, 1),
            "files": {
                "donor": donor_out.name,
                "voxcpm": stage_a.name,
                "vc": stage_b.name,
            },
        }

    # F0-compare modes. Each converts the SAME VoxCPM2 output a different way so the
    # emotion-vs-register trade-off can be judged by ear:
    #   baseline = current behaviour (SeedVC re-centres every emotion's pitch to the
    #              target -> flat register, emotion pitch lost).
    #   A        = keep VoxCPM2's absolute pitch (emotion survives) but in the donor's
    #              register, not the target's.
    #   B        = keep the pitch contour, shifted by a constant so the donor's neutral
    #              lands on the target's register -> emotion AND register.
    F0_COMPARE_MODES: List[Dict[str, Any]] = [
        {"id": "baseline", "label": "ปัจจุบัน (auto-f0)", "auto_f0_adjust": True, "shift": "none"},
        {"id": "A", "label": "A · คงอารมณ์", "auto_f0_adjust": False, "shift": "none"},
        {"id": "B", "label": "B · คงอารมณ์+ตรง target", "auto_f0_adjust": False, "shift": "to_target"},
    ]

    def render_f0_compare(
        self,
        text: str,
        *,
        emotion: str,
        speaker_id: Optional[str] = None,
        ref_audio_bytes: Optional[bytes] = None,
        ref_filename: Optional[str] = None,
        gender: Optional[str] = None,
        donor_set: Optional[str] = None,
        cfg_value: float = 2.5,
        inference_timesteps: int = 10,
    ) -> Dict[str, Any]:
        """Generate one utterance once with VoxCPM2, then voice-convert it three ways.

        Returns ``{"modes": [{id,label,wav,sr,auto_f0_adjust,semi_tone_shift}],
        "pre_vc": {wav,sr}, "diag": {...}}``. The heavy generation runs a single time;
        the three modes differ only in the SeedVC F0 handling. Single-utterance scope.
        """
        import numpy as np
        import soundfile as sf

        emotion = self.validate_emotion(emotion)
        body = strip_instruction(text)
        if not body:
            raise ValueError("No text to synthesize")

        chosen_set = self.resolve_donor_set(donor_set, gender=gender)
        target_wav, temp_target = self._target_voice_path(speaker_id, ref_audio_bytes, ref_filename)

        if settings.seedvc_required and self.seedvc_health() is None:
            raise VoxCPMVCUnavailable(
                f"SeedVC worker at {self.seedvc_url} is not responding."
            )

        # B's constant shift: move the donor's NEUTRAL register onto the target's.
        # Using neutral (not this emotion) as the anchor keeps the emotion's own pitch
        # offset intact after the shift. The anchor is each clip's *register* (a low
        # percentile), not its median, so an expressive target/donor does not inflate
        # the shift; and the result is clamped tight so B can never push F0 a full
        # octave up into chipmunk territory.
        semi_shift_b = 0
        target_reg = self._register_f0(Path(target_wav))
        donor_neutral_reg = None
        neutral_wav = None
        try:
            neutral_wav, _ = self.donor_clip(chosen_set, "neutral")
            donor_neutral_reg = self._register_f0(neutral_wav)
        except Exception:
            donor_neutral_reg = None
        if target_reg and donor_neutral_reg and donor_neutral_reg > 0:
            raw_shift = 12.0 * math.log2(target_reg / donor_neutral_reg)
            clamp = self.F0_SHIFT_CLAMP_ST
            semi_shift_b = int(round(max(-clamp, min(clamp, raw_shift))))

        # Medians are kept for the diagnostics line only (register drives the shift).
        target_med = self._median_f0(Path(target_wav))
        donor_neutral_med = self._median_f0(neutral_wav) if neutral_wav is not None else None

        scratch = Path("scratch/voxcpm_vc")
        scratch.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        src_wav = scratch / f"f0cmp_{emotion}_{ts}.wav"

        modes_out: List[Dict[str, Any]] = []
        pre_vc: Dict[str, Any] = {}
        try:
            synth = self._synth()
            audio, gen_rate = self._generate_one(
                synth, chosen_set, emotion, body,
                cfg_value=cfg_value, inference_timesteps=inference_timesteps, lora_mode="on",
            )
            arr = np.asarray(audio, dtype="float32")
            sf.write(str(src_wav), arr, gen_rate, format="WAV", subtype="PCM_16")

            pbuf = io.BytesIO()
            sf.write(pbuf, arr, gen_rate, format="WAV", subtype="PCM_16")
            pre_vc = {"wav": pbuf.getvalue(), "sr": int(gen_rate)}

            for spec in self.F0_COMPARE_MODES:
                shift = semi_shift_b if spec["shift"] == "to_target" else 0
                vc_out = scratch / f"f0cmp_{emotion}_{spec['id']}_{ts}.wav"
                self._convert(
                    src_wav, Path(target_wav), vc_out,
                    auto_f0_adjust=spec["auto_f0_adjust"], semi_tone_shift=shift,
                )
                marr, sr = sf.read(str(vc_out), dtype="float32")
                if marr.ndim > 1:
                    marr = marr.mean(axis=1)
                obuf = io.BytesIO()
                sf.write(obuf, marr, sr, format="WAV", subtype="PCM_16")
                modes_out.append({
                    "id": spec["id"],
                    "label": spec["label"],
                    "wav": obuf.getvalue(),
                    "sr": int(sr),
                    "auto_f0_adjust": spec["auto_f0_adjust"],
                    "semi_tone_shift": int(shift),
                })
                vc_out.unlink(missing_ok=True)
        finally:
            src_wav.unlink(missing_ok=True)
            if temp_target is not None:
                temp_target.unlink(missing_ok=True)

        return {
            "modes": modes_out,
            "pre_vc": pre_vc,
            "diag": {
                "emotion": emotion,
                "donor": f"{chosen_set}/{emotion}_1.wav",
                "target_med_hz": round(target_med, 1) if target_med else None,
                "donor_neutral_med_hz": round(donor_neutral_med, 1) if donor_neutral_med else None,
                # What B actually anchored on (register = p20), plus the clamp, so a
                # shift that hit the clamp is visible rather than looking arbitrary.
                "target_reg_hz": round(target_reg, 1) if target_reg else None,
                "donor_neutral_reg_hz": round(donor_neutral_reg, 1) if donor_neutral_reg else None,
                "b_shift_clamp_st": self.F0_SHIFT_CLAMP_ST,
                "b_semi_tone_shift": semi_shift_b,
            },
        }

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
