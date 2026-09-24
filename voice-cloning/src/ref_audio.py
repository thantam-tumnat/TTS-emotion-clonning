"""Clean up the tail of a reference clip before it becomes a prompt cache.

Reference clips arrive as a fixed-length window cut from a longer recording (n8n's
`/audio-clip` takes exactly 10 s), with no regard for where words end. So a clip can
finish on the first few dozen milliseconds of the *next* word -- sound that is in
no transcript, sitting right where VoxCPM2 picks up.

That matters in continuation mode (a transcript was given): the model generates as
if carrying on from the end of the clip. A stray onset there gets carried on as an
extra syllable at the start of the take ("มี" in front of every line), and it blurs
where the transcript ends, so the model sometimes re-reads the back half of the
transcript before the new text, or drifts off into something that is not Thai.
Cutting the fragment off and ending the clip on silence stopped all three in A/B
renders of the voice that surfaced this.

Only a *short* fragment after a real pause is removed. Anything longer may be a word
the transcript does contain, and dropping audio the transcript still describes
causes the very read-back this is meant to prevent -- so those clips are left alone
and reported instead.
"""

from __future__ import annotations

import io
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

FRAME_S = 0.01          # analysis hop
MIN_GAP_S = 0.15        # a pause, not a stop consonant's closure
MAX_FRAGMENT_S = 0.35   # longest tail still treated as a cut-off onset
MIN_KEEP_S = 2.0        # never trim a clip down to less than this
HOLD_S = 0.05           # keep the start of the pause: the last word's decay
FADE_S = 0.03
PAD_S = 0.4             # silence the clip ends on after trimming
# Silence is judged against the clip's own levels, since clips arrive at every
# recording level and noise floor: a few dB above its quietest frames, but always at
# least MIN_DEPTH_DB below its speech and never more than MAX_DEPTH_DB below it.
NOISE_MARGIN_DB = 8.0
MIN_DEPTH_DB = 20.0
MAX_DEPTH_DB = 30.0
FLOOR_DB = -60.0


@dataclass(frozen=True)
class TailCheck:
    """What the tail of a clip looks like, and what (if anything) to cut."""

    status: str                     # "trimmed" | "clean" | "mid_speech" | "too_short"
    duration_s: float
    cut_s: Optional[float] = None   # where the kept audio ends (before padding)
    fragment_s: float = 0.0         # length of the sound that was cut off

    @property
    def trimmed(self) -> bool:
        return self.status == "trimmed"


def read_clip(path: str | Path) -> tuple[np.ndarray, int]:
    """Samples (float32, channels kept) and rate.

    libsndfile reads most of what arrives, but some uploads named .mp3 are not ones
    it will open; ffmpeg reads those, so it is the fallback rather than a dependency.
    """
    import soundfile as sf

    try:
        samples, sr = sf.read(str(path), dtype="float32")
        return samples, int(sr)
    except Exception:
        ffmpeg = os.environ.get("SIANGTTS_FFMPEG", "ffmpeg")
        proc = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "wav", "-"],
                              capture_output=True, check=True, timeout=60)
        samples, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
        return samples, int(sr)


def _mono(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=1) if samples.ndim > 1 else samples


def _frame_db(mono: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, int(round(sr * FRAME_S)))
    n = len(mono) // hop
    if n == 0:
        return np.zeros(0)
    frames = mono[: n * hop].reshape(n, hop).astype(np.float64)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return 20.0 * np.log10(rms + 1e-10)


def silence_threshold_db(db: np.ndarray) -> float:
    """Frame level below which a frame counts as silence, for this clip."""
    noise, speech = (float(v) for v in np.percentile(db, [10, 90]))
    near_noise = min(noise + NOISE_MARGIN_DB, speech - MIN_DEPTH_DB)
    return max(near_noise, speech - MAX_DEPTH_DB, FLOOR_DB)


def check_tail(samples: np.ndarray, sr: int) -> TailCheck:
    """Classify how the clip ends, and where to cut if it ends on a fragment."""
    mono = _mono(np.asarray(samples, dtype=np.float32))
    duration = len(mono) / sr
    if duration < MIN_KEEP_S + MAX_FRAGMENT_S:
        return TailCheck("too_short", duration)

    db = _frame_db(mono, sr)
    silent = db < silence_threshold_db(db)
    min_gap = int(round(MIN_GAP_S / FRAME_S))

    # Runs of silent frames, as [start, end) frame indices.
    runs: list[tuple[int, int]] = []
    start = None
    for i, s in enumerate(silent):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(silent)))

    if runs and runs[-1][1] == len(silent) and runs[-1][1] - runs[-1][0] >= min_gap:
        return TailCheck("clean", duration)

    gaps = [r for r in runs if r[1] - r[0] >= min_gap and r[1] < len(silent)]
    if not gaps:
        return TailCheck("mid_speech", duration)

    g0, g1 = gaps[-1]
    fragment = duration - g1 * FRAME_S
    gap_start = g0 * FRAME_S
    if fragment > MAX_FRAGMENT_S or gap_start < MIN_KEEP_S:
        return TailCheck("mid_speech", duration, fragment_s=fragment)

    cut = gap_start + min(HOLD_S, (g1 - g0) * FRAME_S)
    return TailCheck("trimmed", duration, cut_s=cut, fragment_s=fragment)


def apply_tail(samples: np.ndarray, sr: int, check: TailCheck) -> np.ndarray:
    """The clip cut at `check.cut_s`, faded out, and ended on `PAD_S` of silence."""
    if not check.trimmed or check.cut_s is None:
        return samples
    out = np.array(samples[: int(round(check.cut_s * sr))], dtype=np.float32, copy=True)
    fade = min(len(out), int(round(FADE_S * sr)))
    if fade:
        ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        out[-fade:] *= ramp[:, None] if out.ndim > 1 else ramp
    pad_shape = (int(round(PAD_S * sr)),) + out.shape[1:]
    return np.concatenate([out, np.zeros(pad_shape, dtype=np.float32)])


__all__ = ["TailCheck", "apply_tail", "check_tail", "read_clip", "silence_threshold_db"]
