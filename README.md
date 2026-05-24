# VajaCPM

Thai voice-cloning TTS built by LoRA fine-tuning **VoxCPM 2 (2B)** on the
[`dubbing-ai/vaja-thai`](https://huggingface.co/datasets/dubbing-ai/vaja-thai) corpus,
plus a small **LibriTTS-R** slice with raw `text_original` to retain English digit
reading and code-switching.

This repo is the practical implementation of the plan in [`RESEARCH.md`](RESEARCH.md).
The reference architecture / training recipe is adapted from the
[JaiTTS paper](https://arxiv.org/abs/2604.27607), but VajaCPM trains within a single
**RTX 3090 24 GB** budget using LoRA rather than full SFT.

## Status

Scaffold + monitoring + tests are production-ready and validated end-to-end via
`--dry-run`. The actual VoxCPM trainer hookup in `train/train_lora.py` is the only
remaining stub — it raises `NotImplementedError` with a clear integration sketch in
its docstring; everything else (data prep, augmentation, dataset weighting,
TensorBoard, in-training audio eval, timing tracker) runs on a CPU-only machine.

## Layout

```
VajaCPM/
├── conf/
│   └── voxcpm_lora.yaml       # LoRA training recipe (§8.2 of RESEARCH.md)
├── src/
│   ├── thai_normalizer.py     # encoding hygiene only (no number-to-word, no segmentation)
│   ├── augment.py             # DataLoader-time text augmentations
│   ├── inference.py           # thin wrapper around voxcpm.VoxCPM
│   └── eval.py                # CER (Typhoon-Whisper) + SIM (WavLM) + digit-eval
├── train/
│   ├── prepare_vaja_thai.py   # vaja-thai → JSONL @ 48 kHz, ref_audio pairing
│   ├── prepare_libritts.py    # LibriTTS-R text_original → JSONL @ 48 kHz
│   ├── dataset.py             # JSONL → VoxCPM dataset; hooks src/augment.py
│   ├── train_lora.py          # invokes VoxCPM trainer with conf/voxcpm_lora.yaml
│   └── publish_to_hf.py       # push LoRA adapter + config + samples to HF
├── eval/
│   ├── prompts_short.tsv      # Thai short-form (1–15 s)
│   ├── prompts_long.tsv       # Thai long-form (16–30 s)
│   ├── prompts_digits.tsv     # digit-eval (years / prices / phones)
│   └── prompts_listen.tsv     # in-training audio snapshots (TB Audio tab)
├── tests/                     # pytest unit tests for normalizer + augment + dataset + audio prep
├── data/                      # gitignored
├── checkpoints/               # gitignored
├── pyproject.toml             # uv-managed
├── conftest.py                # makes `pytest` resolve src/ + train/ from any CWD
├── RESEARCH.md
└── README.md
```

## Setup

```bash
# Install uv (one-time): https://docs.astral.sh/uv/
uv sync
```

## Workflow

```bash
# 1. Sanity-check the base model on Thai
uv run python -m src.inference --base-only --text "สวัสดีครับ"

# 2. Prepare manifests
uv run python train/prepare_vaja_thai.py --output-dir data/vaja --max-samples 0
uv run python train/prepare_libritts.py  --output-dir data/libritts --subset train-clean-100

# 3. Train LoRA (the dry-run validates the dataset + monitor pipeline without GPU)
uv run python train/train_lora.py --config conf/voxcpm_lora.yaml --dry-run
uv run python train/train_lora.py --config conf/voxcpm_lora.yaml

# 4. Evaluate
uv run python -m src.eval --adapter checkpoints/last --prompts eval/prompts_short.tsv
```

## Tests

```bash
uv run pytest tests/ -q
```

Covers the Thai text normalizer (Unicode hygiene, NFC, tone-mark preservation), the
augmentation safety gates from §8.6.2 (round-trip / ordinal / fraction / year /
digit-recital filters, Thai-cluster-safe whitespace jitter), the manifest weighting
(`WeightedRandomSampler` ratios), and the audio-prep silence trim + duration filter.

## Monitoring

Three things run alongside training, configured under `monitor:` in
`conf/voxcpm_lora.yaml`:

- **TensorBoard** — scalars (loss, throughput, val metrics) flushed every 100 steps.
  Launch with:

    ```bash
    uv run tensorboard --logdir runs/
    ```

- **In-training audio sampler** — every `every_steps` (default 1000) the model
  synthesizes the prompts in `eval/prompts_listen.tsv` (Thai short / long / digits /
  code-switch + EN sanity checks). Output goes to:
  - the **TensorBoard Audio tab** (listen in browser), and
  - `runs/<run>/audio_snapshots/step_<N>/` (raw WAVs, easy to A/B locally).

- **Timing tracker** — wall-clock, step throughput, GPU name + peak VRAM, and val
  metrics at every checkpoint, written incrementally to:

    ```
    runs/<run>/training_summary.json
    ```

  This file is the source of truth for any publication / model-card timing claims.

See `RESEARCH.md` §8 for the full execution plan.

## License

Code: Apache-2.0. Trained checkpoints inherit the most restrictive license of their
training data — with the default Vaja-Thai mix (including `tsync2`), that is
**CC-BY-NC-SA** (non-commercial). Drop `tsync2` + `gigaspeech2` slices to release under
CC-BY-SA-4.0.
