"""Entry point: LoRA fine-tune VoxCPM 2 on the VajaCPM mix.

The actual VoxCPM trainer invocation is a stub — depends on the installed VoxCPM
build's API surface. Until that's verified, this file:

1. Loads the YAML config.
2. Builds train / val datasets with per-source weighted sampling.
3. Constructs the `MonitorBundle` (TensorBoard + audio sampler + timing tracker).
4. Provides `--dry-run` that exercises every callback without GPU work, so the
   monitoring pipeline is fully validated before the heavy training run.

Integration sketch when wiring to VoxCPM's trainer:

    monitor = MonitorBundle.from_config(cfg)
    monitor.timing.on_train_start(config_snapshot=cfg)
    sampler = train_ds.make_weighted_sampler(num_samples=len(train_ds))
    loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler, ...)

    for epoch in range(cfg["train"]["num_epochs"]):
        train_ds.set_epoch(epoch)            # rotates augmentation seeds
        for step, batch in enumerate(loader, start=global_step):
            loss = trainer.step(batch)
            scalars = {"train/loss": loss}
            scalars.update(monitor.timing.on_step_end(step) or {})
            monitor.tb.add_scalars(scalars, step)

            if step % cfg["train"]["eval_every_steps"] == 0:
                val = trainer.evaluate()
                monitor.tb.add_scalars({f"val/{k}": v for k, v in val.items()}, step)
                monitor.timing.on_eval()

            if monitor.audio.should_run(step):
                monitor.audio.run(step, synthesize_fn=trainer.synthesize, tb=monitor.tb)

            if step % cfg["train"]["save_every_steps"] == 0:
                ckpt = trainer.save(step)
                monitor.timing.on_checkpoint(step, ckpt, val_metrics=val)
                # Reset CUDA peak counter so the *next* checkpoint reflects only
                # the work done since this one.
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

        monitor.timing.on_epoch_end()

    monitor.timing.on_train_end(final_metrics=val)
    monitor.tb.close()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make `python train/train_lora.py` work without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.callbacks import MonitorBundle  # noqa: E402
from train.dataset import VajaCPMDataset  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_datasets(cfg: dict) -> tuple[VajaCPMDataset, VajaCPMDataset]:
    """Train uses per-source weights from YAML. Val ignores weights (uniform)."""
    train_sources = cfg["manifests"]["train"]
    val_sources = [{"path": m["path"], "weight": 1.0} for m in cfg["manifests"]["val"]]
    train_ds = VajaCPMDataset.from_sources(
        train_sources, is_train=True, augment_cfg=cfg.get("augment")
    )
    val_ds = VajaCPMDataset.from_sources(
        val_sources, is_train=False, augment_cfg=cfg.get("augment")
    )
    return train_ds, val_ds


def _stub_synth(text: str, ref_audio: str | None = None):
    """Sine-wave placeholder synth so the audio sampler's wiring can be validated
    without VoxCPM. Generates ~1s of a 440 Hz tone at 48 kHz."""
    import numpy as np
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    wav = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return wav


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("conf/voxcpm_lora.yaml"))
    p.add_argument("--dry-run", action="store_true",
                   help="Exercise dataset + monitor callbacks without invoking the GPU trainer.")
    args = p.parse_args()

    cfg = load_config(args.config)
    train_ds, val_ds = build_datasets(cfg)
    print(f"train: {len(train_ds)} items   val: {len(val_ds)} items")
    if len(train_ds):
        print("first train sample:", train_ds[0])

    monitor = MonitorBundle.from_config(cfg)
    monitor.timing.on_train_start(config_snapshot=cfg)

    if args.dry_run:
        print("[dry-run] exercising 3 fake steps + 1 eval + 1 audio snapshot + 1 ckpt")
        for step in (1, 2, 1000):
            scalars = {"train/loss": 5.0 - step * 0.001}
            scalars.update(monitor.timing.on_step_end(step) or {})
            monitor.tb.add_scalars(scalars, step)
            if monitor.audio.should_run(step):
                monitor.audio.run(step, synthesize_fn=_stub_synth, tb=monitor.tb)
        monitor.timing.on_eval()
        monitor.tb.add_scalars({"val/loss": 4.5}, 1000)
        monitor.timing.on_checkpoint(1000, "checkpoints/dryrun", val_metrics={"val/loss": 4.5})
        monitor.timing.on_epoch_end()
        monitor.timing.on_train_end(final_metrics={"val/loss": 4.5})
        monitor.tb.close()
        print("[dry-run] OK — see TB log dir + training_summary.json under runs/")
        return

    monitor.tb.close()
    raise NotImplementedError(
        "Hook to VoxCPM's trainer once the installed API surface is verified. "
        "See module docstring for the integration shape using MonitorBundle hooks."
    )


if __name__ == "__main__":
    main()
