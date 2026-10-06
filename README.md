<div align="center">

# 🧩 PartFlow

**Feedforward 3D Editing Learns from Semantic-Part Transformation**

[Jiawei Weng](mailto:jweng007@e.ntu.edu.sg)<sup>1,&ast;</sup>,
[Saining Zhang](https://sainingzhang.github.io/)<sup>1,&ast;,†</sup>,
[Zhenxin Diao](mailto:diaozhenxin2005@outlook.com)<sup>2,&ast;</sup>,
[Peishuo Li](mailto:peishuo001@e.ntu.edu.sg)<sup>1</sup>,
[Henghaofan Zhang](mailto:hhfzhang@outlook.com)<sup>2</sup>,
[Junhao Chen](https://yisuanwang.github.io/)<sup>2</sup>,
[Hao Zhao](https://sites.google.com/view/fromandto)<sup>2,†</sup>

<sup>1</sup>Nanyang Technological University, Singapore &nbsp;&nbsp;
<sup>2</sup>Tsinghua University, China

<sub>&ast;Equal contribution. †Corresponding author.</sub>

<a href="https://dennis-jwweng.github.io/pxform/"><img src="https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome" height="22" alt="Project page"></a>
<a href="https://arxiv.org/abs/2605.27351"><img src="https://img.shields.io/badge/arXiv-2605.27351-b5212f.svg?logo=arxiv" height="22" alt="Paper"></a>
<a href="https://huggingface.co/datasets/ART-3D/Pxform_v1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Pxform__v1-d96902.svg" height="22" alt="Dataset"></a>
<a href="https://huggingface.co/ART-3D/PartFlow_models"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-PartFlow__models-276cb4.svg" height="22" alt="Pretrained weights"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height="22" alt="License"></a>

</div>

<p align="center">
  <img src="assets/gallery.png" alt="PartFlow edited asset gallery" width="95%">
</p>

**Edit a 3D asset with a target image.** PartFlow uses two stages of flow matching,
with no per-asset optimization or 3D mask at inference. It learns from **Pxform**:
100K+ paired edits across seven edit types, grounded in semantic parts.

## 📰 News

- **October 2026:** Training code released, including data preparation, mask losses, and render losses.
- **August 2026:** Paper accepted to **SIGGRAPH Asia 2026**.
- **May 2026:** Inference code and pretrained weights released.

## 🔬 Method

<p align="center">
  <img src="assets/method.png" alt="PartFlow two-stage architecture" width="95%">
</p>

Both stages condition a pretrained [TRELLIS](https://github.com/microsoft/TRELLIS)
backbone on source latents and a target edit image.

| Stage | Prediction |
|---|---|
| **SS flow** | Edited sparse structure |
| **SLAT flow** | Edited structured latent, decoded into a textured 3D asset |

## 🛠️ Installation

Set up the [TRELLIS environment](https://github.com/microsoft/TRELLIS#-installation),
or use the bundled installer, then install PartFlow dependencies:

```bash
. ./setup.sh --new-env --basic --flash-attn --diffoctreerast --spconv \
             --mipgaussian --kaolin --nvdiffrast
pip install -r requirements.txt
```

Inference environment: **Python 3.10 · PyTorch 2.5.0 · CUDA 12.4**.

## 🚀 Inference

Download the stage weights and run the included example:

```bash
python download_weights.py
python inference.py --input examples/mod_glass_disc_table --output_dir outputs
```

Each case produces `outputs/<edit_id>/edit.glb` and `pred_slat.npz`.
Pass a parent directory to process multiple cases; use `--skip_existing` to resume.

<details>
<summary><b>📁 Prepare your own inputs</b></summary>

Each case contains source TRELLIS latents and a target image:

```text
<case_dir>/
├── ori_ss_latents.npz   # mean: float32 [8,16,16,16]
├── ori_latents.npz      # coords: int [N,3]; feats: float32 [N,8]
├── edit_img.png         # target image, RGB or RGBA
└── case_meta.json       # optional metadata
```

Provide SLAT features as **raw encoder outputs**; normalization and decoder
denormalization are handled internally. Ground-truth edited latents are optional
and ignored by inference. Sampling options: `--steps`, `--cfg_strength`, and
`--manifest`; see `python inference.py --help`.

</details>

## 🏋️ Training

Both stages support real part-mask losses, with **SS silhouette** and **SLAT RGB**
render supervision. Download the paired training data and matching masks together:

```bash
pip install -r requirements-training.txt
python download_data.py --shards 00  # start with one shard; use all for full data
GPU_ID=0 bash scripts/run_training_smoke.sh

python train.py --config configs/train_stage1_ss.json \
  --output_dir outputs/train_ss --num_gpus 1
python train.py --config configs/train_stage2_slat.json \
  --output_dir outputs/train_slat --num_gpus 1
```

Data and masks are discovered automatically under `data/Pxform_v1/`.
See the **[training guide](docs/training.md)** for normalization, losses,
offline weights, multi-GPU training, and checkpoint resume.

## 🎨 Results

<p align="center">
  <img src="assets/teaser_geometry.jpg" alt="Geometry editing comparisons" width="95%">
  <br><br>
  <img src="assets/teaser_colormat.jpg" alt="Appearance editing comparisons" width="95%">
</p>

## 📖 Citation

```bibtex
@article{weng2026partflow,
  title   = {Feedforward 3D Editing Learns from Semantic-Part Transformation},
  author  = {Weng, Jiawei and Zhang, Saining and Diao, Zhenxin and Li, Peishuo and Zhang, Henghaofan and Chen, Junhao and Zhao, Hao},
  journal = {arXiv preprint arXiv:2605.27351},
  year    = {2026}
}
```

## 🙏 Acknowledgements

Built on [TRELLIS](https://github.com/microsoft/TRELLIS).
