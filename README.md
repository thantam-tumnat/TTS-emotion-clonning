# SiangTTS

Thai voice-cloning TTS built by LoRA fine-tuning **VoxCPM 2 (2B)** on the
[`dubbing-ai/vaja-thai`](https://huggingface.co/datasets/dubbing-ai/vaja-thai) corpus,
plus a small **LibriTTS-R** slice with raw `text_original` to retain English digit
reading and code-switching.

This repo is the practical implementation of the plan in [`RESEARCH.md`](RESEARCH.md).
**[`PLAN.md`](PLAN.md) is the live execution roadmap** — next steps and the
decision tree for each evaluation outcome.
The reference architecture / training recipe is adapted from the
[JaiTTS paper](https://arxiv.org/abs/2604.27607), but SiangTTS trains within a single
**RTX 3090 24 GB** budget using LoRA rather than full SFT.

## Status

Fully implemented against **voxcpm 2.0.3**. `train/train_lora.py` mirrors the
mechanics of VoxCPM's official `scripts/train_voxcpm_finetune.py` (packer, bf16
autocast, grad accumulation, LoRA-only checkpoints, resume, save-on-signal) and
adds per-source weighted sampling, DataLoader-time text augmentation with
tokenize-after-augment, and the MonitorBundle (TensorBoard + audio snapshots +
timing JSON). Checkpoints use VoxCPM's loadable LoRA layout
(`lora_weights.safetensors` + `lora_config.json`), so they work directly with
`voxcpm clone --lora-path ...` and `src/inference.py --adapter ...`.

Key facts learned from the installed API (these differ from early RESEARCH.md
assumptions): VoxCPM2's AudioVAE **encodes at 16 kHz** and decodes at 48 kHz, so
manifests are stored at 16 kHz; the LoRA config keys are
`enable_lm/enable_dit/enable_proj/r/alpha/dropout`; the base HF id is
`openbmb/VoxCPM2`. Torch is pinned to cu128 wheels (driver 575.x = CUDA 12.9
cannot run cu130).

Everything except the actual GPU run is validated: 45 unit tests pass, and
`--dry-run` exercises datasets + monitoring on a CPU-only machine.

## Layout

```
SiangTTS/
├── conf/
│   ├── voxcpm_lora.yaml       # LoRA training recipe (§8.2 of RESEARCH.md) — 3090-sized
│   └── voxcpm_sft.yaml        # full-SFT recipe — A100-80G sized (escalation path)
├── src/
│   ├── thai_normalizer.py     # encoding hygiene only (no number-to-word, no segmentation)
│   ├── augment.py             # DataLoader-time text augmentations
│   ├── inference.py           # thin wrapper around voxcpm.VoxCPM
│   └── eval.py                # CER (Typhoon-Whisper) + SIM (WavLM) + digit-eval
├── train/
│   ├── prepare_vaja_thai.py   # vaja-thai → JSONL @ 16 kHz, ref_audio pairing
│   ├── prepare_libritts.py    # LibriTTS-R text_original → JSONL @ 16 kHz
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

# 3b. Full SFT (rented A100-80G — does NOT fit the 3090; same script, no `lora:`)
uv run python train/train_lora.py --config conf/voxcpm_sft.yaml

# 4. Evaluate
uv run python -m src.eval --adapter checkpoints/siangtts-lora-v0/latest --prompts eval/prompts_short.tsv
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
