# PartFlow training

The SS and SLAT models train separately on paired Pxform latents. SLAT training
uses ground-truth edited coordinates; inference uses the SS model's generated
coordinates. Only ControlNet parameters are optimized. The pretrained trunk,
DINO image encoder, SS decoder and Gaussian decoder stay frozen.

## Download and organize data

The public [Pxform_v1 dataset](https://huggingface.co/datasets/ART-3D/Pxform_v1/tree/main/data)
ships matching training and mask archives for shards `00`–`09`, plus per-shard
manifests. Everything is prepared under this repository by default:

```text
PartFlow/data/Pxform_v1/
├── downloads/data/                 downloaded HF files; retained for reuse
│   ├── train_shards/               paired training tar.zst archives
│   ├── train_mask_sidecars/        semantic part-mask tar.zst archives
│   └── train_manifests/            public JSONL manifests
├── training/
│   ├── release.json                source commit and published split metadata
│   ├── manifests/all.jsonl         combined locally prepared shard manifests
│   └── <edit_type>/<shard>/<obj_id>/<edit_id>/
│       ├── before.npz
│       ├── after.npz
│       ├── before.png
│       ├── after.png
│       └── meta.json               instruction and camera metadata
└── mask_sidecars/<edit_type>/<shard>/<obj_id>/<edit_id>.npz
```

From the repository root, start with one shard (~3.94 GB of compressed pairs
and ~13.6 MB of compressed masks for `00`; allow additional space for extraction):

```bash
pip install -r requirements-data.txt
python download_data.py --shards 00
```

Download all shards for full training, or add selected shards later:

```bash
python download_data.py --shards all
python download_data.py --shards 01 02
python download_data.py --verify_only
```

The script downloads both pairs and masks, extracts `.tar.zst`, combines only
prepared shard manifests, and checks every record's files, camera metadata and
local-edit mask presence. Downloads use Hugging Face caching/resume; successfully
extracted archives have completion markers. Interrupted extraction is repeated.
A branch is resolved to a commit and recorded in `training/release.json`; pass
`--revision <commit>` to reproduce that release. Use the same revision when
adding shards. Full NPZ shapes and mask token lengths are checked by the loader.
If the publisher changes an existing shard, use a fresh `--data_dir` to avoid
mixing old extracted records with a new release.

A custom destination is supported with `download_data.py --data_dir /path/to/Pxform_v1`;
then train with `--data_dir /path/to/Pxform_v1/training`. The mask directory is
found automatically beside `training`. No sibling source repository is needed.
Downloaded data is ignored by Git.

The public release contains the **training split only**. Its `release.json`
allows `split: train` without a separate object-ID split file. It does not
provide validation data: setting `split: val` requires an actual
`training/data/splits/val.obj_ids.txt` and matching records. Existing expanded
local datasets with `data/splits/{train,val}.obj_ids.txt` retain object-based
filtering. Do not generate a validation set by randomly splitting edits of the
same object.

Each NPZ contains raw encoder outputs: `ss [8,16,16,16]`,
`slat_coords [N,3 or 4]`, and `slat_feats [N,8]`. `after.png` supplies the DINO
condition and RGB render target. Published `meta.json.camera` supplies Blender
camera-to-world `transform_matrix` and horizontal `camera_angle_x`; older
expanded datasets using `view.meta.json.frame` are also supported.

Normalization is representation-specific:

| Input | Processing |
|---|---|
| SS latent | Keep raw SS encoder output; no additional dataset mean/std |
| Original and edited SLAT | `(feature - TRELLIS_mean) / TRELLIS_std` |
| Missing original SLAT at a new target coordinate | Zero in normalized space |
| DINO image | Resize to 518, values in `[0,1]`, then ImageNet mean/std in the encoder |
| RGB render target | Full frame, `[0,1]`, alpha composited on white; no ImageNet normalization |
| SLAT decoder input | Undo SLAT normalization before Gaussian decoding |
| Coordinates and masks | Integer coordinates and keep weights in `[0,1]` |

RGB condition preprocessing uses plain
resize for RGB, alpha crop and premultiply for RGBA. For inference on the same
RGB training images, set `PARTFLOW_SKIP_REMBG=1` to select the matching legacy
preprocessing; the inference default segments and crops user images.

The dataset and render decoder normalization must agree. Do not normalize
already normalized NPZs a second time. Training defaults assume raw NPZs.

## Real part masks

`mask_source: sidecar` requires real part masks. Override the root with
`--mask_sidecar_root /path/to/mask_sidecars`. Sidecars are:

`<root>/<edit_type>/<shard>/<obj_id>/<edit_id>.npz`

Required keys are `mask_keep_ss [16,16,16]` and `mask_keep_slat [N_after]`.
The SLAT mask must use exactly the after-token order. Missing/misaligned local
masks raise an error. Global edits have zero keep masks and are filtered per
sample in the trainer, without disabling local samples in the same batch.

The downloaded semantic sidecars are used by both default configs. Global edits
intentionally have no sidecars and receive zero keep masks. Local edits must
have real sidecars; the loader raises an error for missing or invalid masks.

The loader intersects SS keep weights with source occupied coarse cells, and
SLAT keep weights with source/target coordinate overlap. This avoids preserving
empty space and discouraging additions.

`mask_source: latent_delta` is an optional, explicitly estimated mask mode.
It compares paired features in normalized space, dilates changes and keeps
only shared occupied tokens. It is not a semantic part mask and is not the
default training recipe.

## Losses

`x_t = (1-t)*x_edit + [sigma_min+(1-sigma_min)*t]*noise`

`v_edit = (1-sigma_min)*noise - x_edit`

Both stages use `MSE(v_pred, v_edit)` plus a masked velocity MSE against
`v_original`, using the same noise. Dense masked MSE divides by the expanded
mask weight sum exactly once.

- SS: `L_FM + 0.3*L_keep + lambda_render*L_silhouette`.
  The frozen SS decoder produces a soft occupancy volume. Camera rays render
  a differentiable silhouette, compared to the decoded GT SS geometry in the
  same camera. This is geometry supervision, not RGB texture supervision.
  Occupancy logits use temperature 10 to avoid saturation of pretrained
  logits that can exceed magnitude 100. GT logits are thresholded at zero.
- SLAT: `L_FM + 0.7*L_keep + lambda_render*L_RGB`.
  The frozen Gaussian decoder and differentiable rasterizer render the target
  view. `L_RGB` is pixel MSE by default; optional DreamSim is available with
  `render_loss.dreamsim > 0` and `pip install dreamsim`.

Clean-latent reconstruction uses the exact finite-sigma formula:
`x_pred = (1-sigma_min)*x_t - [sigma_min+(1-sigma_min)*t]*v_pred`.
Decoder execution is differentiable; it is never wrapped in `no_grad` for the
prediction. Gaussian decoding runs in FP32 to preserve small weighted render
gradients, using the sparse attention FP32 SDPA fallback.

Render weight ramps from zero at 30% of training to 0.01 at 60%. It is evaluated
every four steps, on the first eligible sample with `t < 0.5`. The default
includes all edit types. Before the weight becomes positive, render decoding
is skipped entirely. These defaults are a training recipe, not a convergence
claim.

## Run

Activate a TRELLIS environment, then install `requirements-training.txt`.
Mesh-only dependencies are imported lazily; SS/SLAT Gaussian training does not
require Kaolin. The optional DreamSim term is disabled by default.

From the repository root, after downloading/preparing data:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train_stage1_ss.json \
  --output_dir outputs/train_ss --num_gpus 1 --auto_retry 0

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train_stage2_slat.json \
  --output_dir outputs/train_slat --num_gpus 1 --auto_retry 0
```

Both stages default to `PartFlow/data/Pxform_v1/training` and automatically
find `PartFlow/data/Pxform_v1/mask_sidecars`. Pretrained TRELLIS weights resolve
through Hugging Face. For offline weights, add
`--pretrained_dir /path/to/TRELLIS-image-large` (must contain `ckpts/`), and
set `DINOV2_LOCAL_PATH` to a local DINOv2 source checkout with its weights cached.
Use `--mask_sidecar_root` only to override the automatically discovered mask root.

Default batch size is 2 per GPU, split into two microbatches. Select multiple
visible GPUs and adjust `--num_gpus` for DDP. Separate output directories are
required for SS and SLAT. Resume is automatic from the latest misc checkpoint;
use `--ckpt none` for a fresh run or `--load_dir ... --ckpt 5000` explicitly.

## Smoke and regression checks

```bash
GPU_ID=0 bash scripts/run_training_smoke.sh

PYTHONPATH=. python -m unittest discover -s tests -v
```

`--smoke_steps 2` uses the full pretrained model and real paired data, batch 1,
low-noise timesteps, real part masks and immediate render loss at resolution 64.
It asserts finite losses, nonempty masks, actual render execution, nonzero
gradients from each auxiliary loss and an optimizer update to a ControlNet gate
on every step. It saves raw/EMA/optimizer checkpoints and `smoke_report.json`.
Smoke starts with a conservative FP16 loss scale of 1; normal training retains
dynamic scaling. Smoke is a code-path/gradient witness, not a quality test.

To reuse an existing expanded dataset without copying it, create links once:

```bash
mkdir -p data/Pxform_v1
ln -s /path/to/expanded/training data/Pxform_v1/training
ln -s /path/to/semantic/mask_sidecars data/Pxform_v1/mask_sidecars
```

The repository-local sibling mask directory is discovered before resolving the
training symlink. Scripts/configs stay portable; the link targets are specific
to your machine. Do not run the downloader over linked existing data.
