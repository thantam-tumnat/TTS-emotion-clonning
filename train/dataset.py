"""JSONL → VoxCPM-compatible dataset, with DataLoader-time augmentation hooks.

Bridges our manifest format and whatever dataset class the installed VoxCPM build
expects. Returns plain dicts; `train_lora.py` adapts them into VoxCPM's collate.

Augmentations from `src.augment` run inside `__getitem__` so each epoch produces
fresh text spellings of the same audio (RESEARCH.md §8.6.3).

Manifest row schema (Vaja-Thai):
    {"audio": "wavs/x.wav", "text": "...", "duration": 4.31,
     "speaker": "...", "dataset_id": "vaja_thai", "tier": 1,
     "ref_audio": "wavs/y.wav"?, "no_digit_aug": false?}

Manifest row schema (LibriTTS-R adds):
    "text_original": "...", "text_normalized": "..."
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.augment import (
    case_jitter,
    maybe_digitize_thai,
    pick_libritts_text,
    whitespace_jitter,
)

log = logging.getLogger(__name__)


class VajaCPMDataset(Dataset):
    """Multi-source JSONL dataset with per-source weights for the sampler.

    Construct via either:
        - `VajaCPMDataset(manifest_paths=[...])` — equal weighting (back-compat).
        - `VajaCPMDataset.from_sources([{"path": "...", "weight": 1.0}, ...])` —
          per-source weights honored by `make_weighted_sampler()`.

    The weights are *normalized*: `weight: 0.25` for one source out of two with
    `weight: 1.0` means the smaller source contributes 0.25 / 1.25 ≈ 20% of the
    effective batch.
    """

    def __init__(
        self,
        manifest_paths: list[str | Path] | None = None,
        *,
        sources: list[dict[str, Any]] | None = None,
        root_dirs: list[str | Path] | None = None,
        is_train: bool = True,
        augment_cfg: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        if (manifest_paths is None) == (sources is None):
            raise ValueError("Pass exactly one of `manifest_paths` or `sources`.")

        self.is_train = is_train
        self.augment_cfg = augment_cfg or {}
        self._base_seed = seed
        self.rows: list[dict[str, Any]] = []
        self.row_roots: list[Path] = []
        self.row_source_idx: list[int] = []
        self.source_weights: list[float] = []
        self._source_row_counts: list[int] = []

        if sources is not None:
            paths = [s["path"] for s in sources]
            self.source_weights = [float(s.get("weight", 1.0)) for s in sources]
        else:
            paths = list(manifest_paths)  # type: ignore[arg-type]
            self.source_weights = [1.0] * len(paths)

        roots = [Path(r) for r in (root_dirs or [Path(p).parent for p in paths])]
        if len(roots) != len(paths):
            raise ValueError("root_dirs length must match number of manifests")

        for src_idx, (path, root) in enumerate(zip(paths, roots)):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Manifest not found: {path} — did you run the prepare scripts?"
                )
            count = 0
            with open(path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{path}:{line_no} bad JSONL: {e}") from e
                    self.rows.append(row)
                    self.row_roots.append(root)
                    self.row_source_idx.append(src_idx)
                    count += 1
            self._source_row_counts.append(count)
            log.info("loaded %d rows from %s (weight=%.3f)", count, path, self.source_weights[src_idx])

        if not self.rows:
            raise ValueError("No rows loaded from any manifest.")

    @classmethod
    def from_sources(
        cls,
        sources: list[dict[str, Any]],
        **kwargs: Any,
    ) -> "VajaCPMDataset":
        return cls(sources=sources, **kwargs)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def _rng_for(self, idx: int) -> random.Random:
        # Per-item RNG keeps augmentation deterministic given (seed, idx).
        # `set_epoch` rotates the seed so successive epochs see different spellings.
        return random.Random(self._base_seed * 10_000_019 + idx)

    def _augment_text(self, row: dict[str, Any], rng: random.Random) -> str:
        cfg = self.augment_cfg
        ds_id = row.get("dataset_id", "")
        no_aug = bool(row.get("no_digit_aug", False))

        if ds_id == "libritts_r":
            text = pick_libritts_text(
                row.get("text_original") or row.get("text", ""),
                row.get("text_normalized") or row.get("text", ""),
                p_normalized=cfg.get("en_text_original_vs_normalized", {}).get("p_normalized", 0.3),
                rng=rng,
            )
            text = case_jitter(text, p=cfg.get("case_jitter", {}).get("p", 0.1), rng=rng)
        else:
            text = row.get("text", "")
            if not no_aug:
                text = maybe_digitize_thai(
                    text,
                    p_full=cfg.get("thai_digit", {}).get("p_full", 0.4),
                    p_partial=cfg.get("thai_digit", {}).get("p_partial", 0.1),
                    rng=rng,
                )

        text = whitespace_jitter(
            text,
            p=cfg.get("whitespace_jitter", {}).get("p", 0.15),
            rng=rng,
        )
        return text

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        root = self.row_roots[idx]

        if self.is_train:
            rng = self._rng_for(idx)
            text = self._augment_text(row, rng)
        else:
            text = row.get("text") or row.get("text_normalized") or ""

        item: dict[str, Any] = {
            "audio_path": str(root / row["audio"]),
            "text": text,
            "duration": row.get("duration"),
            "dataset_id": row.get("dataset_id"),
        }
        if row.get("ref_audio"):
            item["ref_audio_path"] = str(root / row["ref_audio"])
        return item

    def set_epoch(self, epoch: int) -> None:
        """Trainer should call this at start of each epoch to rotate augmentation."""
        self._base_seed = epoch

    # ------------------------------------------------------------------
    def per_row_weights(self) -> list[float]:
        """Per-row sampling weights so each *source* contributes its `weight`
        fraction of the effective batch, regardless of source row count.

        For source `s` with `n_s` rows and configured weight `w_s`, every row in
        that source gets weight `w_s / n_s`. Normalization is handled by
        `WeightedRandomSampler` (it accepts unnormalized weights).
        """
        per_source = [
            (w / n if n else 0.0)
            for w, n in zip(self.source_weights, self._source_row_counts)
        ]
        return [per_source[s] for s in self.row_source_idx]

    def make_weighted_sampler(
        self,
        num_samples: int | None = None,
        replacement: bool = True,
        generator=None,
    ):
        """Build a `torch.utils.data.WeightedRandomSampler` honoring source weights."""
        from torch.utils.data import WeightedRandomSampler

        weights = self.per_row_weights()
        if num_samples is None:
            num_samples = len(self.rows)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )


__all__ = ["VajaCPMDataset"]
