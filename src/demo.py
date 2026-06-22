"""SiangTTS **static demo page** — curated samples + GitHub Pages HTML.

This module owns the offline comparison artifact. The interactive Gradio UI lives
in `src/app.py`; the inference API lives in `src/serve.py`. All three share the
samples generated here (demo/samples/manifest.json).

1. `prep` — curate a *diverse* set of Common Voice val prompts (each with a
   same-speaker reference clip) spanning gender (estimated from pitch),
   short/long text, and numeric content; then assemble a 4-way comparison:
       ref          — reference voice (a different utterance, same speaker)
       ground_truth — the real recording of the prompt text
       base         — VoxCPM2 (no adapter), cloning from ref
       lora         — SiangTTS (VoxCPM2 + Thai LoRA), cloning from ref
   Plus a digit-reading showcase (base vs LoRA on Arabic-numeral prompts; no
   ground truth — the dataset has no Arabic digits). Writes wavs + manifest.json.

2. `html` — render a self-contained static page (docs/index.html + docs/samples/)
   for GitHub Pages. No server needed; plays in any browser / on GitHub.

Usage:
    uv run python -m src.demo prep        # GPU: curate + generate all samples
    uv run python -m src.demo html        # build docs/ static page (GitHub Pages)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEMO_DIR = Path("demo/samples")
MANIFEST = DEMO_DIR / "manifest.json"
DOCS_DIR = Path("docs")
DEFAULT_BASE = "openbmb/VoxCPM2"
DEFAULT_ADAPTER = "checkpoints/siangtts-lora-v0/latest"

_THAI_NUMWORDS = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด",
                  "แปด", "เก้า", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน", "ล้าน"]
F0_GENDER_HZ = 165.0   # median-F0 split for male/female estimate

# Digit-reading showcase prompts (Arabic numerals → spoken Thai). No ground
# truth exists for these (dataset transcripts have no Arabic digits), so they
# run base-vs-LoRA only.
DIGIT_PROMPTS = [
    ("digit_year", "เขาเกิดเมื่อปี 1990 ที่จังหวัดเชียงใหม่"),
    ("digit_price", "สินค้าชิ้นนี้ราคา 1,250 บาท"),
    ("digit_phone", "ติดต่อได้ที่เบอร์ 081-234-5678"),
    ("digit_mix", "ร้านเปิด 9 โมงเช้า ถึง 5 ทุ่ม ทุกวัน"),
]


# ---------------------------------------------------------------------------
# Curation — pick a diverse comparison set
# ---------------------------------------------------------------------------

def _median_f0(path: str) -> float:
    import librosa
    import numpy as np

    y, sr = librosa.load(path, sr=16000)
    f, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr)
    f = f[~np.isnan(f)]
    return float(np.median(f)) if len(f) else 0.0


def curate(val_manifest: str, n_per_bucket: int = 1) -> list[dict]:
    """Select a diverse set: {male,female} × {short,long} + numeric examples."""
    root = Path(val_manifest).parent
    rows = [json.loads(line) for line in open(val_manifest, encoding="utf-8")]
    cand = [r for r in rows if r.get("ref_audio")]

    feats = []
    for r in cand:
        hz = _median_f0(str(root / r["audio"]))
        if hz <= 0:
            continue  # no voiced frames — skip
        feats.append({
            **r,
            "pitch_hz": round(hz, 1),
            "gender_est": "male" if hz < F0_GENDER_HZ else "female",
            "length": ("short" if r["duration"] <= 4.5
                       else "long" if r["duration"] >= 8.0 else "mid"),
            "has_numeric": any(w in r["text"] for w in _THAI_NUMWORDS),
        })

    picked: list[dict] = []
    seen_audio: set[str] = set()

    def take(pred, k):
        for f in feats:
            if k <= 0:
                break
            if f["audio"] in seen_audio or not pred(f):
                continue
            picked.append(f)
            seen_audio.add(f["audio"])
            k -= 1

    # gender × length grid, then ensure numeric coverage
    for g in ("male", "female"):
        for ln in ("short", "long"):
            take(lambda f, g=g, ln=ln: f["gender_est"] == g and f["length"] == ln, n_per_bucket)
    take(lambda f: f["has_numeric"] and f["gender_est"] == "male", 1)
    take(lambda f: f["has_numeric"] and f["gender_est"] == "female", 1)
    return picked


# ---------------------------------------------------------------------------
# Prep — generate the comparison set + digit showcase (GPU)
# ---------------------------------------------------------------------------

def prep(
    val_manifest: str = "data/vaja-cv/val.jsonl",
    base_model: str = DEFAULT_BASE,
    adapter: str = DEFAULT_ADAPTER,
) -> None:
    from .inference import Synthesizer

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(val_manifest).parent

    print("[demo] curating diverse examples (pitch/length/numeric) ...")
    picks = curate(val_manifest)
    entries = []
    for i, r in enumerate(picks):
        eid = f"ex_{i:02d}"
        shutil.copy2(root / r["audio"], DEMO_DIR / f"{eid}_ground_truth.wav")
        shutil.copy2(root / r["ref_audio"], DEMO_DIR / f"{eid}_ref.wav")
        entries.append({
            "id": eid, "kind": "compare", "text": r["text"],
            "ref": str(root / r["ref_audio"]),
            "ref_rel": f"{eid}_ref.wav", "ground_truth_rel": f"{eid}_ground_truth.wav",
            "gender_est": r["gender_est"], "pitch_hz": r["pitch_hz"],
            "length": r["length"], "duration": round(r["duration"], 1),
            "has_numeric": r["has_numeric"],
        })
    print(f"[demo] curated {len(entries)} comparison examples: "
          f"{[(e['gender_est'], e['length'], 'num' if e['has_numeric'] else '') for e in entries]}")

    digit_entries = [{"id": did, "kind": "digit", "text": txt} for did, txt in DIGIT_PROMPTS]

    # Base outputs (one load), then LoRA outputs (second load).
    for tag, adapter_path in (("base", None), ("lora", adapter)):
        print(f"[demo] synthesizing {tag} ...")
        synth = Synthesizer(base_model=base_model, adapter_path=adapter_path)
        for e in entries:                       # cloning from ref
            out = DEMO_DIR / f"{e['id']}_{tag}.wav"
            synth.synth_to_file(e["text"], out, ref_audio=e["ref"])
            e[f"{tag}_rel"] = out.name
        for e in digit_entries:                 # default voice, no ref
            out = DEMO_DIR / f"{e['id']}_{tag}.wav"
            synth.synth_to_file(e["text"], out)
            e[f"{tag}_rel"] = out.name
        del synth

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(entries + digit_entries, f, ensure_ascii=False, indent=2)
    print(f"[demo] wrote {len(entries) + len(digit_entries)} entries → {MANIFEST}")


# ---------------------------------------------------------------------------
# Static HTML — GitHub Pages
# ---------------------------------------------------------------------------

_PAGE_CSS = """
body{font-family:system-ui,'Segoe UI',sans-serif;max-width:1100px;margin:2rem auto;
  padding:0 1rem;color:#1b1f24;line-height:1.5}
h1{margin-bottom:.2rem} .sub{color:#57606a;margin-top:0}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #d0d7de;padding:.5rem;text-align:left;vertical-align:top}
th{background:#f6f8fa} audio{width:200px}
.badge{display:inline-block;font-size:.75rem;padding:.1rem .45rem;border-radius:1rem;
  margin-right:.3rem;background:#eaeef2}
.male{background:#ddf4ff} .female{background:#ffeff7} .num{background:#fff8c5}
.txt{font-size:.95rem} .results td,.results th{text-align:center}
.note{color:#57606a;font-size:.85rem}
"""

_RESULTS_TABLE = """
<table class="results">
<tr><th>Metric</th><th>Base VoxCPM2</th><th>SiangTTS (LoRA)</th></tr>
<tr><td>Short-form Thai CER</td><td>5.70%</td><td><b>~3.8%</b></td></tr>
<tr><td>Long-form Thai CER</td><td>runaway</td><td><b>1.6%</b></td></tr>
<tr><td>Voice cloning SIM / CER</td><td>—</td><td><b>0.882 / 2.5%</b></td></tr>
<tr><td>Numerals (year/price/phone)</td><td>weak</td><td><b>reads correctly</b></td></tr>
</table>
"""


def build_html() -> None:
    if not MANIFEST.exists():
        raise SystemExit("Run `python -m src.demo prep` first.")
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    DOCS_DIR.mkdir(exist_ok=True)
    out_samples = DOCS_DIR / "samples"
    out_samples.mkdir(exist_ok=True)
    for wav in DEMO_DIR.glob("*.wav"):
        shutil.copy2(wav, out_samples / wav.name)

    def aud(rel: str) -> str:
        return f'<audio controls preload="none" src="samples/{rel}"></audio>'

    compare = [e for e in entries if e.get("kind") == "compare"]
    digits = [e for e in entries if e.get("kind") == "digit"]

    rows = []
    for e in compare:
        g = e["gender_est"]
        badges = (f'<span class="badge {g}">{"♂" if g=="male" else "♀"} {g} (est.)</span>'
                  f'<span class="badge">{e["length"]} · {e["duration"]}s</span>')
        if e.get("has_numeric"):
            badges += '<span class="badge num">🔢 numeric</span>'
        rows.append(
            f"<tr><td class='txt'>{e['text']}<br>{badges}</td>"
            f"<td>{aud(e['ref_rel'])}</td><td>{aud(e['ground_truth_rel'])}</td>"
            f"<td>{aud(e['base_rel'])}</td><td>{aud(e['lora_rel'])}</td></tr>"
        )
    compare_table = (
        "<table><tr><th>Text</th><th>Reference voice</th><th>Ground truth</th>"
        "<th>Base VoxCPM2</th><th>SiangTTS (LoRA)</th></tr>" + "".join(rows) + "</table>"
    )

    drows = [
        f"<tr><td class='txt'>{e['text']}</td><td>{aud(e['base_rel'])}</td>"
        f"<td>{aud(e['lora_rel'])}</td></tr>"
        for e in digits
    ]
    digit_table = (
        "<table><tr><th>Text (Arabic numerals)</th><th>Base VoxCPM2</th>"
        "<th>SiangTTS (LoRA)</th></tr>" + "".join(drows) + "</table>"
    )

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SiangTTS — Thai Voice-Cloning TTS Demo</title><style>{_PAGE_CSS}</style></head><body>
<h1>SiangTTS — Thai Voice-Cloning TTS</h1>
<p class="sub">LoRA fine-tune of VoxCPM2 on Thai · trained on a single RTX 3090 ·
<a href="https://huggingface.co/dubbing-ai/SiangTTS-VoxCPM2-Thai-LoRA">model on HuggingFace</a></p>
{_RESULTS_TABLE}
<h2>Voice-cloning comparison</h2>
<p class="note">Each row: the <b>reference voice</b> (a different utterance from the
same speaker), the real <b>ground-truth</b> recording, the <b>base</b> VoxCPM2, and
<b>SiangTTS</b>. Examples span gender, text length, and numeric content. Gender is
<i>estimated from pitch</i>.</p>
{compare_table}
<h2>Number &amp; digit reading</h2>
<p class="note">Arabic numerals spoken as Thai (no ground truth — the training
transcripts contain no Arabic digits, so this shows the augmentation effect).</p>
{digit_table}
<p class="note">CER is measured with Typhoon-Whisper-Large-v3 and is an upper bound
on error — the ASR judge itself mis-recognises some rare Thai words the model
pronounces correctly. License: CC-BY-SA-4.0.</p>
</body></html>"""
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"[demo] wrote {DOCS_DIR/'index.html'} ({len(compare)} comparison + {len(digits)} digit) "
          f"+ {len(list(out_samples.glob('*.wav')))} wavs → enable GitHub Pages on /docs")


def main() -> None:
    p = argparse.ArgumentParser(description="SiangTTS static demo page")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prep", help="Curate + generate all comparison samples (GPU)")
    pp.add_argument("--val-manifest", default="data/vaja-cv/val.jsonl")
    pp.add_argument("--base-model", default=DEFAULT_BASE)
    pp.add_argument("--adapter", default=DEFAULT_ADAPTER)

    sub.add_parser("html", help="Build docs/index.html static page (GitHub Pages)")

    args = p.parse_args()
    if args.cmd == "prep":
        prep(val_manifest=args.val_manifest, base_model=args.base_model, adapter=args.adapter)
    elif args.cmd == "html":
        build_html()


if __name__ == "__main__":
    main()
