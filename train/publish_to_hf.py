"""Push a trained LoRA adapter (weights + config + sample audio) to the HF Hub.

Stub. Adapt to the user's HF account / repo naming convention before first run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def push(
    repo_id: str,
    adapter_dir: Path,
    samples_dir: Path | None = None,
    config_path: Path | None = None,
    private: bool = True,
) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(repo_id=repo_id, folder_path=str(adapter_dir), path_in_repo=".")
    if samples_dir is not None and samples_dir.exists():
        api.upload_folder(repo_id=repo_id, folder_path=str(samples_dir), path_in_repo="samples")
    if config_path is not None and config_path.exists():
        api.upload_file(repo_id=repo_id, path_or_fileobj=str(config_path), path_in_repo=config_path.name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="e.g. username/vajacpm-lora-v0")
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--samples-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=Path("conf/voxcpm_lora.yaml"))
    p.add_argument("--public", action="store_true", help="Push as public; default is private")
    args = p.parse_args()

    push(
        repo_id=args.repo_id,
        adapter_dir=args.adapter_dir,
        samples_dir=args.samples_dir,
        config_path=args.config,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
