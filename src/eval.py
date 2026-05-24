"""Evaluation: Thai CER (Typhoon-Whisper-Large-v3) + speaker SIM (WavLM) + digit-eval.

Mirrors JaiTTS's `cal_wer.sh` / `cal_sim.sh` pipeline (RESEARCH.md §8.7 step 8).
This is a stub — wire up the actual ASR + SV pipelines once the trained adapter is
ready. Kept thin so the project layout is complete, not so the eval is feature-full.

Prompt TSV format (eval/prompts_*.tsv):
    id<TAB>text<TAB>ref_audio_path
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .inference import synthesize


def load_prompts(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows


def synthesize_all(prompts: list[dict[str, str]], out_dir: Path, **synth_kwargs) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in prompts:
        synthesize(
            text=row["text"],
            output_path=out_dir / f"{row['id']}.wav",
            ref_audio=row.get("ref_audio") or None,
            **synth_kwargs,
        )


def compute_cer(audio_dir: Path, prompts: list[dict[str, str]]) -> float:
    """Stub: transcribe with Typhoon-Whisper-Large-v3 and compute CER vs prompts.

    Implementation outline:
      from transformers import pipeline
      asr = pipeline("automatic-speech-recognition", "scb10x/typhoon-asr-realtime")
      hyp = [asr(str(audio_dir / f"{r['id']}.wav"))["text"] for r in prompts]
      ref = [r["text"] for r in prompts]
      from jiwer import cer
      return cer(ref, hyp)
    """
    raise NotImplementedError("CER pipeline not yet wired — see docstring.")


def compute_sim(audio_dir: Path, prompts: list[dict[str, str]]) -> float:
    """Stub: WavLM speaker-verification cosine similarity vs ref_audio."""
    raise NotImplementedError("SIM pipeline not yet wired — see docstring.")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate VajaCPM checkpoints.")
    p.add_argument("--prompts", required=True, help="TSV file with id/text/ref_audio columns")
    p.add_argument("--out-dir", default="eval/out", help="Where to write synthesized audio")
    p.add_argument("--adapter", default=None, help="LoRA adapter path; omit for base-only")
    p.add_argument("--base-model", default="openbmb/VoxCPM2-2B")
    p.add_argument("--cer", action="store_true", help="Compute Thai CER after synthesis")
    p.add_argument("--sim", action="store_true", help="Compute speaker SIM after synthesis")
    args = p.parse_args()

    prompts = load_prompts(args.prompts)
    out_dir = Path(args.out_dir)
    synthesize_all(
        prompts,
        out_dir,
        base_model=args.base_model,
        adapter_path=args.adapter,
    )

    if args.cer:
        print(f"CER: {compute_cer(out_dir, prompts):.4f}")
    if args.sim:
        print(f"SIM: {compute_sim(out_dir, prompts):.4f}")


if __name__ == "__main__":
    main()
