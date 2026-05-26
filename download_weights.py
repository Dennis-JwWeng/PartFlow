#!/usr/bin/env python3
"""Download PartFlow stage-model weights from Hugging Face.

Fetches the two trained stage models from `ART-3D/PartFlow_models` into ./weights:

    weights/
        stage1_ss/model.pt
        stage2_slat/model.pt

The frozen DINOv2 encoder and TRELLIS SS/SLat decoders are NOT downloaded here
— `inference.py` fetches them on first run from their official repos.
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

HF_REPO = "ART-3D/PartFlow_models"
_HERE = os.path.abspath(os.path.dirname(__file__))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=HF_REPO, help="Hugging Face model repo id.")
    p.add_argument("--out", default=os.path.join(_HERE, "weights"),
                   help="Destination directory.")
    p.add_argument("--token", default=None, help="Hugging Face token (if private).")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading {args.repo} -> {args.out}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="model",
        local_dir=args.out,
        token=args.token,
    )
    print("Done. Weights are under:", args.out)


if __name__ == "__main__":
    main()
