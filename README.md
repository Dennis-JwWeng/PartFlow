# PartFlow

**PartFlow** is a two-stage, image-conditioned 3D editing model. Given an
existing 3D asset (encoded as TRELLIS latents) and a target *edit image*, it
produces an edited 3D asset as a textured `.glb`.

It is built on [TRELLIS](https://github.com/microsoft/TRELLIS): a ControlNet
branch is added to each of the two TRELLIS flow-matching DiTs and trained on
paired edit data, while the TRELLIS encoders/decoders stay frozen.

```
edit image ──DINOv2──► condition tokens
ori SS latent   + cond ──Stage 1: SS ControlNet────► edited SS latent ──ss_dec──► edited voxels
ori SLat        + cond ──Stage 2: SLat ControlNet──► edited SLat ──TRELLIS decoders──► edit.glb
```

Only the two ControlNet denoisers are PartFlow weights. The DINOv2 image
encoder and the TRELLIS SS/SLat decoders are frozen pretrained models, fetched
automatically from their official Hugging Face repos on first run.

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

This pulls the two trained denoisers from
[`ART-3D/PartFlow_models`](https://huggingface.co/ART-3D/PartFlow_models).

## Data layout (Pxform format)

Inference reads pre-encoded inputs. Each *case* is a directory:

```
<case_dir>/
    ori_ss_latents.npz   # key `mean`: float32 [8, 16, 16, 16]   — original sparse-structure latent
    ori_latents.npz      # `coords` [N,3] int, `feats` [N,8] f32 — original structured latent (SLat)
    edit_img.png         # the target edit image (RGB or RGBA)
    case_meta.json       # optional metadata (prompt, edit type, ...)
```

`ori_ss_latents.npz` / `ori_latents.npz` are the TRELLIS latents of the
**original** asset; produce them with the standard TRELLIS image-to-3D encoder.
Ground-truth `edit_*` files, if present, are ignored by inference.

`dataset.py:PxformDataset` accepts either a single case directory or a parent
directory holding many cases, so it plugs straight into a `DataLoader`.

## Run inference

```bash
# single case
python inference.py --input examples/add_ring_band --output_dir outputs

# a whole directory of cases
python inference.py --input /path/to/pxform/h3d_edit --output_dir outputs

# useful flags
#   --steps 50           flow-Euler sampling steps
#   --cfg_strength 0.0   classifier-free guidance (0 = condition only)
#   --manifest ids.json  restrict to a list of case ids
#   --skip_existing      resume a partial run
```

Each case writes `outputs/<edit_id>/edit.glb` and `pred_slat.npz`; a
`run_summary.json` is written at the end.

### Notes

- **SLat space is raw** — PartFlow is trained on un-normalised SLat features,
  so inference uses raw space (no SLat normalisation).
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
├── trellis/            TRELLIS backbone + PartFlow ControlNet models
├── setup.sh            CUDA-extension installer
└── requirements.txt    pure-pip dependencies
```

## Acknowledgements

Built on [TRELLIS](https://github.com/microsoft/TRELLIS) and
[DINOv2](https://github.com/facebookresearch/dinov2).
