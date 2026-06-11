"""Prepare dubbing-ai/vaja-thai → JSONL manifest @ 16 kHz for VajaCPM.

Adapted from dubbing-ai/indextts2-thai's prepare_vaja_thai.py, with three changes:

1. TARGET_SR 22050 → 16000. VoxCPM2's AudioVAE *encodes* at 16 kHz (it decodes
   at 48 kHz — that's the output side, irrelevant for training data). Storing at
   the encoder rate avoids a per-epoch resample in the DataLoader.
2. Output JSONL (audio / text / ref_audio / duration / dataset_id / no_digit_aug)
   instead of CSV.
3. Per-speaker `ref_audio` pairing on 30–50% of rows (RESEARCH.md §8.3).

Default tier oversampling weights (from RESEARCH.md §8.3):
    Tier 1 (studio/best)  → 2x
    Tier 2 (clean)        → 1x
    Tier 3 (acceptable)   → 1x
    Tier 4 (marginal)     → skip

Usage:
    uv run python train/prepare_vaja_thai.py --output-dir data/vaja
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.thai_normalizer import normalize_thai_text  # noqa: E402
from train.audio_prep import resample_trim_save  # noqa: E402

TARGET_SR = 16000  # AudioVAE encoder input rate (VoxCPM2 decodes at 48 kHz)
DEFAULT_TIER_WEIGHTS = {1: 2, 2: 1, 3: 1, 4: 0}
REF_AUDIO_PROBABILITY = 0.4   # 30–50% per VoxCPM docs
DATASET_ID = "vaja_thai"


def prepare(
    output_dir: Path,
    config: str = "all",
    max_samples: int = 0,
    min_quality_tier: int = 1,
    tier_weights: dict[int, int] | None = None,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> None:
    from datasets import interleave_datasets, load_dataset

    if tier_weights is None:
        tier_weights = DEFAULT_TIER_WEIGHTS.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    if config == "all":
        configs = ["tsync2", "gigaspeech2", "commonvoice", "porjai_central"]
        streams = [
            load_dataset("dubbing-ai/vaja-thai", c, split="train", streaming=True)
            for c in configs
        ]
        ds = interleave_datasets(streams, stopping_strategy="all_exhausted")
    else:
        ds = load_dataset("dubbing-ai/vaja-thai", config, split="train", streaming=True)

    rng = random.Random(seed)
    rows: list[dict] = []
    speaker_to_clips: dict[str, list[str]] = defaultdict(list)
    seen = 0

    for sample in tqdm(ds, desc="streaming vaja-thai"):
        if max_samples and seen >= max_samples:
            break
        tier = int(sample.get("quality_tier", 4))
        if tier < min_quality_tier:
            continue
        weight = tier_weights.get(tier, 0)
        if weight == 0:
            continue

        text = normalize_thai_text(sample["text"])
        if not text:
            continue

        audio = sample["audio"]
        wav_id = f"vaja_{seen:08d}"
        wav_rel = f"wavs/{wav_id}.wav"
        duration = resample_trim_save(
            audio["array"], audio["sampling_rate"],
            wav_dir / f"{wav_id}.wav", target_sr=TARGET_SR,
        )
        if duration is None:
            continue   # outside [1, 30] s window after silence trim

        speaker = str(sample.get("speaker_id") or sample.get("source") or "unknown")
        speaker_to_clips[speaker].append(wav_rel)

        for _ in range(weight):
            rows.append(
                {
                    "audio": wav_rel,
                    "text": text,
                    "duration": round(duration, 3),
                    "speaker": speaker,
                    "dataset_id": DATASET_ID,
                    "tier": tier,
                }
            )
        seen += 1

    # Attach ref_audio for ~REF_AUDIO_PROBABILITY of rows, using a different clip
    # from the same speaker. Rows whose speaker has only one clip get no ref.
    for r in rows:
        if rng.random() >= REF_AUDIO_PROBABILITY:
            continue
        candidates = [c for c in speaker_to_clips[r["speaker"]] if c != r["audio"]]
        if candidates:
            r["ref_audio"] = rng.choice(candidates)

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    # Validation rows: pin no_digit_aug=True so eval is stable
    for r in val_rows:
        r["no_digit_aug"] = True

    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "val.jsonl", val_rows)
    print(f"wrote {len(train_rows)} train / {len(val_rows)} val rows → {output_dir}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("data/vaja"))
    p.add_argument("--config", default="all", choices=["all", "tsync2", "porjai_central", "commonvoice", "gigaspeech2"])
    p.add_argument("--max-samples", type=int, default=0, help="0 = no limit")
    p.add_argument("--min-tier", type=int, default=1)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prepare(
        output_dir=args.output_dir,
        config=args.config,
        max_samples=args.max_samples,
        min_quality_tier=args.min_tier,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
