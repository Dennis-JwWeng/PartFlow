# PartFlow

### Feedforward 3D Editing Learns from Semantic-Part Transformation

[Project Page](https://dennis-jwweng.github.io/pxform/) · [Paper (coming soon)](#) · [arXiv (coming soon)](#) · [🤗 Dataset (Pxform)](https://huggingface.co/datasets/ART-3D/Pxform_v1) · [🤗 Weights](https://huggingface.co/ART-3D/PartFlow_models)

This repository contains the **inference code** for **PartFlow**, a feedforward
3D editing network that edits an existing 3D asset to match a target edit image.

## Overview

Scalable feedforward 3D editing should be learned from **semantic-part
transformations**. We propose **Pxform**, a high-quality 3D editing dataset with
over 100K consistent before/after editing pairs across seven edit types,
grounding edits directly in semantic 3D parts. Built upon Pxform, **PartFlow**
is a feedforward 3D editing network that injects source-aware latent control
into pretrained 3D generative priors, and requires **no 3D edit mask during
inference**.

## How it works

PartFlow edits in two stages, conditioning a pretrained 3D generative prior
([TRELLIS](https://github.com/microsoft/TRELLIS)) on the **source asset's
latent** and a **target edit image**:

```
edit image ──DINOv2──► image condition
source sparse-structure latent + condition ──Stage 1──► edited sparse structure ──► edited voxels
source structured latent (SLAT) + condition ──Stage 2──► edited SLAT ──► edit.glb
```

The DINOv2 image encoder and the TRELLIS sparse-structure / SLAT decoders are
frozen pretrained models, fetched automatically from their official Hugging
Face repos on first run. Only the two PartFlow stage models are released here.

## Installation

PartFlow needs the same CUDA extensions as TRELLIS (`spconv`, `flash-attn`,
`kaolin`, `diff_gaussian_rasterization`, `nvdiffrast`, `diffoctreerast`).
Tested with Python 3.10, PyTorch 2.5.0 + CUDA 12.4.

```bash
# 1. compiled CUDA extensions (adapted from TRELLIS)
. ./setup.sh --new-env --basic --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# 2. pure-pip dependencies
pip install -r requirements.txt
```

## Weights

```bash
python download_weights.py          # -> ./weights/{stage1_ss,stage2_slat}/
```

This pulls the two trained stage models from
[`ART-3D/PartFlow_models`](https://huggingface.co/ART-3D/PartFlow_models).

## Data layout (Pxform format)

Inference reads pre-encoded inputs. Each *case* is a directory:

```
<case_dir>/
    ori_ss_latents.npz   # key `mean`: float32 [8, 16, 16, 16]   — source sparse-structure latent
    ori_latents.npz      # `coords` [N,3] int, `feats` [N,8] f32 — source structured latent (SLAT)
    edit_img.png         # the target edit image (RGB or RGBA)
    case_meta.json       # optional metadata (prompt, edit type, ...)
```

`ori_ss_latents.npz` / `ori_latents.npz` are the TRELLIS latents of the
**source** asset; produce them with the standard TRELLIS image-to-3D encoder.
Ground-truth `edit_*` files, if present, are ignored by inference.

`dataset.py:PxformDataset` accepts either a single case directory or a parent
directory holding many cases, so it plugs straight into a `DataLoader`.

## Run inference

```bash
# single case
python inference.py --input examples/mod_glass_disc_table --output_dir outputs

# a whole directory of cases
python inference.py --input /path/to/pxform/cases --output_dir outputs

# useful flags
#   --steps 50           flow-sampling steps
#   --cfg_strength 0.0   classifier-free guidance (0 = condition only)
#   --manifest ids.json  restrict to a list of case ids
#   --skip_existing      resume a partial run
```

Each case writes `outputs/<edit_id>/edit.glb` and `pred_slat.npz`; a
`run_summary.json` is written at the end.

### Notes

- **Raw SLAT space** — PartFlow operates on un-normalised SLAT features; no
  feature normalisation is applied at inference.
- **Image preprocessing** — edit images are segmented with `rembg`(u2net) and
  tight-cropped to match training. Set `PARTFLOW_SKIP_REMBG=1` if your inputs
  are already clean cut-outs.
- **Offline DINOv2** — set `DINOV2_LOCAL_PATH` to a local `facebookresearch/dinov2`
  checkout to avoid the torch.hub download.

## Repository layout

```
PartFlow/
├── inference.py        two-stage inference pipeline + CLI
├── dataset.py          PxformDataset (Pxform per-case loader)
├── download_weights.py fetch weights from Hugging Face
├── configs/            Stage 1 / Stage 2 model configs
├── examples/           one ready-to-run example case
├── trellis/            TRELLIS backbone + PartFlow stage models
├── setup.sh            CUDA-extension installer
└── requirements.txt    pure-pip dependencies
```

## Citation

```bibtex
@article{pxform2026,
  title   = {Pxform: Feedforward 3D Editing Learns from Semantic-Part Transformation},
  author  = {Pxform Team},
  journal = {Preprint},
  year    = {2026}
}
```

## Acknowledgements

Built on [TRELLIS](https://github.com/microsoft/TRELLIS) and
[DINOv2](https://github.com/facebookresearch/dinov2).
