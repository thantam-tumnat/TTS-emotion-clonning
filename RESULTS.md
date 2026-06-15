# SiangTTS — Results Log

Append-only record of eval numbers. CER = Typhoon-Whisper-Large-v3 character
error rate (whitespace/punct stripped); SIM = WavLM x-vector cosine (gen vs ref).
See [`PLAN.md`](PLAN.md) for what each outcome means and what to do next.

| Date | Checkpoint | Eval set | CER | SIM | Notes |
|---|---|---|---|---|---|
| 2026-06-15 | vanilla VoxCPM2 (base, no LoRA) | prompts_short (5) | **5.70%** | — | Phase-0 baseline. No ref audio. Small sample (5 prompts) — high variance, treat as ballpark. Already < OmniVoice 7.71%; confirms VoxCPM2 has usable Thai priors. |
| 2026-06-15 | smoke LoRA, step 500 (tier-1 porjai only, ~2.7 ep) | prompts_short (5) | **3.80%** | — | Validates adapter-eval path (384 LoRA params loaded). Beats baseline (5.70%→3.80%) from a tiny single-speaker smoke. Directional only. |

## Smoke run findings (2026-06-15)

Tier-1-only LoRA (porjai, ~2,940 clips), `conf/voxcpm_smoke.yaml`. Purpose was
mechanics, not quality. What we learned:

- **VRAM:** batch_size 2 OOMs on the 24 GB 3090 (peak 22.9 GB). batch_size 1
  peaks at **20.4 GB** — adopted for both smoke and `conf/voxcpm_lora.yaml`.
  Launch training with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Trainable params:** 36.2M / 2326.2M (1.56%) at r=64 on LM+DiT.
- **Generation-length collapse from over-training:** at step 500 the in-training
  snapshots have healthy durations (3.4–5.1 s); by step 1000 every prompt
  collapsed to exactly 1.0 s (stop-token overfit on the tiny single-speaker
  set). → For the real run: fewer epochs (now 2), more/diverse data, and treat
  snapshot duration as a health signal. Track it in Phase 1.
- **Adapter checkpoints load cleanly** into `voxcpm.VoxCPM(lora_weights_path=...)`.

## Reference points (other systems, not directly comparable — different eval sets/ASR)

- JaiTTS-v1.0 (closed): 1.94% short / 2.55% long — beats human GT (1.98% / 2.47%)
- OmniVoice base: 7.71% (FLEURS) — worse than its GT 6.98%

## Caveats on the baseline

- Only 5 short prompts — expand `eval/prompts_*.tsv` before treating any CER as
  precise. The number is a directional "before", good for measuring LoRA deltas.
- Base model run with `optimize=True` (torch.compile) and default cfg=2.5,
  timesteps=10.
