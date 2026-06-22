"""SiangTTS comparison demo (Gradio).

Two parts:

1. `prep` — for a handful of Common Voice val prompts (each with a same-speaker
   reference clip), assemble a 4-way comparison set:
       ref         — the reference voice (a *different* utterance, same speaker)
       ground_truth— the real recording of the prompt text
       base        — VoxCPM2 (no adapter), cloning from ref
       lora        — SiangTTS (VoxCPM2 + Thai LoRA), cloning from ref
   Writes wavs + manifest.json under demo/samples/.

2. `app` — a Gradio page with a fixed comparison table (plays all four per row)
   plus an interactive tab (type Thai text + optional reference → base vs LoRA).

Usage:
    uv run python -m src.demo prep --n 8        # GPU: generate comparison set
    uv run python -m src.demo app               # launch Gradio (loads model for live tab)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEMO_DIR = Path("demo/samples")
MANIFEST = DEMO_DIR / "manifest.json"
DEFAULT_BASE = "openbmb/VoxCPM2"
DEFAULT_ADAPTER = "checkpoints/siangtts-lora-v0/latest"


# ---------------------------------------------------------------------------
# Prep — generate the comparison set (GPU)
# ---------------------------------------------------------------------------

def prep(
    n: int = 8,
    val_manifest: str = "data/vaja-cv/val.jsonl",
    base_model: str = DEFAULT_BASE,
    adapter: str = DEFAULT_ADAPTER,
    seed: int = 0,
) -> None:
    import random

    from .inference import Synthesizer

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(val_manifest).parent

    rows = [json.loads(line) for line in open(val_manifest, encoding="utf-8")]
    rows = [r for r in rows if r.get("ref_audio")]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    if not rows:
        raise SystemExit(f"No ref_audio rows in {val_manifest}")

    # Copy ref + ground-truth recordings into the demo dir.
    entries = []
    for i, r in enumerate(rows):
        eid = f"ex_{i:02d}"
        gt_src = root / r["audio"]
        ref_src = root / r["ref_audio"]
        shutil.copy2(gt_src, DEMO_DIR / f"{eid}_ground_truth.wav")
        shutil.copy2(ref_src, DEMO_DIR / f"{eid}_ref.wav")
        entries.append({"id": eid, "text": r["text"],
                        "ref": str(ref_src), "ref_rel": f"{eid}_ref.wav",
                        "ground_truth_rel": f"{eid}_ground_truth.wav"})

    # Base outputs (one model load), then LoRA outputs (second load).
    for tag, adapter_path in (("base", None), ("lora", adapter)):
        print(f"[demo] synthesizing {tag} ...")
        synth = Synthesizer(base_model=base_model, adapter_path=adapter_path)
        for e in entries:
            out = DEMO_DIR / f"{e['id']}_{tag}.wav"
            synth.synth_to_file(e["text"], out, ref_audio=e["ref"])
            e[f"{tag}_rel"] = out.name
        del synth  # free GPU between loads

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[demo] wrote {len(entries)} examples → {MANIFEST}")


# ---------------------------------------------------------------------------
# App — Gradio comparison + interactive tabs
# ---------------------------------------------------------------------------

def build_app(base_model: str = DEFAULT_BASE, adapter: str = DEFAULT_ADAPTER):
    import gradio as gr

    entries = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else []

    # Lazy single Synthesizer for the interactive tab (loaded on first use).
    live: dict = {}

    def _ensure_live():
        if "lora" not in live:
            from .inference import Synthesizer

            live["lora"] = Synthesizer(base_model=base_model, adapter_path=adapter)
            live["base"] = Synthesizer(base_model=base_model, adapter_path=None)
        return live

    def generate(text: str, ref):
        if not text.strip():
            raise gr.Error("Enter some Thai text.")
        s = _ensure_live()
        ref_path = ref if ref else None
        base_wav = s["base"].synth(text, ref_audio=ref_path)
        lora_wav = s["lora"].synth(text, ref_audio=ref_path)
        sr = s["lora"].sample_rate
        return (sr, base_wav), (sr, lora_wav)

    with gr.Blocks(title="SiangTTS — Thai TTS + Voice Cloning") as demo:
        gr.Markdown(
            "# SiangTTS — Thai Voice-Cloning TTS\n"
            "LoRA fine-tune of **VoxCPM2** on Thai. Compare the reference voice, "
            "the real recording, the **base** model, and the **SiangTTS LoRA**."
        )

        with gr.Tab("Comparison"):
            if not entries:
                gr.Markdown("_No samples yet — run `uv run python -m src.demo prep` first._")
            for e in entries:
                gr.Markdown(f"**{e['id']}** — {e['text']}")
                with gr.Row():
                    gr.Audio(str(DEMO_DIR / e["ref_rel"]), label="Reference voice")
                    gr.Audio(str(DEMO_DIR / e["ground_truth_rel"]), label="Ground truth")
                    gr.Audio(str(DEMO_DIR / e["base_rel"]), label="Base VoxCPM2")
                    gr.Audio(str(DEMO_DIR / e["lora_rel"]), label="SiangTTS (LoRA)")

        with gr.Tab("Try it"):
            txt = gr.Textbox(label="Thai text", value="สวัสดีครับ ยินดีที่ได้รู้จัก")
            ref_in = gr.Audio(label="Reference voice (optional, 3–10s)", type="filepath")
            btn = gr.Button("Generate", variant="primary")
            with gr.Row():
                out_base = gr.Audio(label="Base VoxCPM2")
                out_lora = gr.Audio(label="SiangTTS (LoRA)")
            btn.click(generate, [txt, ref_in], [out_base, out_lora])

    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="SiangTTS comparison demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prep", help="Generate the 4-way comparison set (GPU)")
    pp.add_argument("--n", type=int, default=8)
    pp.add_argument("--val-manifest", default="data/vaja-cv/val.jsonl")
    pp.add_argument("--base-model", default=DEFAULT_BASE)
    pp.add_argument("--adapter", default=DEFAULT_ADAPTER)

    pa = sub.add_parser("app", help="Launch the Gradio demo")
    pa.add_argument("--base-model", default=DEFAULT_BASE)
    pa.add_argument("--adapter", default=DEFAULT_ADAPTER)
    pa.add_argument("--share", action="store_true", help="Create a public gradio link")
    pa.add_argument("--port", type=int, default=7860)

    args = p.parse_args()
    if args.cmd == "prep":
        prep(n=args.n, val_manifest=args.val_manifest,
             base_model=args.base_model, adapter=args.adapter)
    else:
        app = build_app(base_model=args.base_model, adapter=args.adapter)
        app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
