#!/usr/bin/env python3
"""PartFlow — two-stage image-conditioned 3D editing inference.

Pipeline (per edit case)::

    edit image --DINOv2--> condition tokens
    source SS latent  + cond --Stage 1 (sparse-structure flow)--> edited SS latent
    edited SS latent --ss_dec--> 16^3 occupancy --> edited voxel coords
    source SLAT (mapped to new coords) + cond --Stage 2 (structured-latent flow)--> edited SLAT
    edited SLAT --TRELLIS decoders--> mesh + gaussians --> edit.glb

Only the two PartFlow stage models (Stage 1 / Stage 2) are released weights;
the DINOv2 encoder and the TRELLIS SS/SLAT decoders are frozen pretrained
models fetched from their official Hugging Face repos.

Usage
-----
    # single case directory
    python inference.py --input examples/add_ring_band --output_dir outputs

    # a parent directory of many case sub-directories
    python inference.py --input /path/to/pxform/h3d_edit --output_dir outputs

See README.md for the expected per-case data layout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch
import torch.nn.functional as F

# Make the bundled `trellis` package importable regardless of CWD.
_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from PIL import Image
from torch.utils.data import DataLoader

from dataset import PxformDataset, collate_identity

from trellis import models
from trellis.datasets.editing_image_latent import _load_image_as_tensor
from trellis.datasets.editing_slat_text import map_ori_to_edit_coords
from trellis.models import from_pretrained
from trellis.modules import sparse as sp
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines.samplers.flow_euler import FlowEulerCfgSampler
from trellis.utils import postprocessing_utils

# Frozen pretrained models (auto-downloaded from Hugging Face on first run).
DINOV2_NAME = "dinov2_vitl14_reg"
TRELLIS_REPO = "JeffreyXiang/TRELLIS-image-large"
SS_DECODER = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"

# ImageNet normalisation used by DINOv2.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# Image conditioning
# -----------------------------------------------------------------------------
_REMBG_SESSION = None


def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        os.environ.setdefault("U2NET_HOME", os.path.join(_HERE, ".u2net_cache"))
        os.makedirs(os.environ["U2NET_HOME"], exist_ok=True)
        import rembg  # noqa: WPS433
        _REMBG_SESSION = rembg.new_session("u2net")
    return _REMBG_SESSION


def preprocess_edit_image(image_path: str, image_size: int = 518) -> torch.Tensor:
    """Match PartFlow training image conditioning.

    If the image has no real alpha channel, run rembg(u2net) to segment the
    foreground; then tight-crop on the alpha bbox with a 1.2x margin, resize to
    `image_size`, and alpha-premultiply onto a black background.

    Set ``PARTFLOW_SKIP_REMBG=1`` to fall back to a plain resize (legacy
    behaviour) — useful when inputs are already clean cut-outs.

    Returns an RGB tensor [3, S, S] in [0, 1].
    """
    if os.environ.get("PARTFLOW_SKIP_REMBG", "0").strip() in ("1", "true", "True"):
        return _load_image_as_tensor(image_path, image_size)

    image = Image.open(image_path)
    has_real_alpha = False
    if image.mode == "RGBA":
        has_real_alpha = bool((np.array(image.getchannel(3)) < 255).any())

    if not has_real_alpha:
        image = __import__("rembg").remove(
            image.convert("RGB"), session=_get_rembg_session()
        )
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha_np = np.array(image.getchannel(3))
    nz = (alpha_np > int(0.8 * 255)).nonzero()
    if nz[0].size > 0:
        x0, y0, x1, y1 = nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) / 2 * 1.2
        image = image.crop(
            (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        )

    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    rgb = torch.tensor(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
    return rgb * alpha.unsqueeze(0)


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
class PartFlowPipeline:
    """Loads all models once; call `.edit(case)` per Pxform case."""

    def __init__(
        self,
        s1_config: str,
        s2_config: str,
        s1_ckpt: str,
        s2_ckpt: str,
        device: str = "cuda",
    ) -> None:
        self.device = device

        print("[PartFlow] loading DINOv2 image encoder ...", flush=True)
        self.dinov2 = _load_dinov2(DINOV2_NAME).eval().to(device)

        print("[PartFlow] loading Stage 1 (sparse-structure flow) ...", flush=True)
        self.ss_model = _build_denoiser(s1_config, s1_ckpt, device)
        print("[PartFlow] loading Stage 2 (structured-latent flow) ...", flush=True)
        self.slat_model = _build_denoiser(s2_config, s2_ckpt, device)

        print("[PartFlow] loading frozen TRELLIS decoders ...", flush=True)
        self.ss_dec = from_pretrained(SS_DECODER).to(device).eval()
        self.decode_pipe = TrellisImageTo3DPipeline.from_pretrained(TRELLIS_REPO)
        self.decode_pipe.cuda()
        self.decode_pipe.models["slat_decoder_mesh"].eval()
        self.decode_pipe.models["slat_decoder_gs"].eval()

        self.sampler = FlowEulerCfgSampler(sigma_min=1e-5)

    @torch.no_grad()
    def encode_image(self, image_path: str):
        """Edit image -> (cond, neg_cond) DINOv2 patch-token conditions."""
        img = preprocess_edit_image(image_path).unsqueeze(0).to(self.device)
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device)[None, :, None, None]
        std = torch.tensor(_IMAGENET_STD, device=self.device)[None, :, None, None]
        feats = self.dinov2((img - mean) / std, is_training=True)["x_prenorm"]
        cond = F.layer_norm(feats, feats.shape[-1:])
        return cond, torch.zeros_like(cond)

    @torch.no_grad()
    def edit(
        self,
        case: dict,
        steps: int = 50,
        cfg_strength: float = 0.0,
        seed: int | None = None,
    ) -> dict:
        """Run the full two-stage edit for one Pxform case.

        Returns a dict with the predicted SLat sparse tensor, decoded mesh and
        gaussians, plus the predicted SS latent (numpy).
        """
        if seed is not None:
            torch.manual_seed(seed)

        cond, neg_cond = self.encode_image(case["edit_image_path"])

        # ---- Stage 1: sparse-structure flow ----
        ori_ss = torch.tensor(case["ori_ss"]).float().unsqueeze(0).to(self.device)
        z_ss = self.sampler.sample(
            self.ss_model,
            noise=torch.randn_like(ori_ss),
            cond=cond,
            neg_cond=neg_cond,
            ori_voxel=ori_ss,
            steps=steps,
            cfg_strength=cfg_strength,
            verbose=False,
        ).samples

        # Decode predicted SS latent to occupied voxel coords (batched [b,x,y,z]).
        ss_decoded = self.ss_dec(z_ss)
        edit_coords = torch.argwhere(ss_decoded > 0)[:, [0, 2, 3, 4]].int().to(self.device)

        # ---- Stage 2: structured-latent flow (raw SLAT space) ----
        mapped_ori = map_ori_to_edit_coords(
            case["ori_slat_coords"],
            case["ori_slat_feats"],
            edit_coords[:, 1:].cpu().numpy(),
        )
        in_ch = self.slat_model.in_channels
        noise_s2 = sp.SparseTensor(
            feats=torch.randn(edit_coords.shape[0], in_ch, device=self.device),
            coords=edit_coords,
        )
        ori_slat_t = sp.SparseTensor(
            feats=torch.tensor(mapped_ori).float().to(self.device),
            coords=edit_coords.clone(),
        )
        slat = self.sampler.sample(
            self.slat_model,
            noise=noise_s2,
            cond=cond,
            neg_cond=neg_cond,
            ori_slat=ori_slat_t,
            steps=steps,
            cfg_strength=cfg_strength,
            verbose=False,
        ).samples.detach()

        # ---- Decode SLat to mesh + gaussians ----
        decoded = self.decode_pipe.decode_slat(slat, ["mesh", "gaussian"])
        return {
            "slat": slat,
            "mesh": decoded["mesh"][0],
            "gaussian": decoded["gaussian"][0],
            "pred_ss": z_ss.detach().squeeze(0).cpu().numpy(),
        }

    @staticmethod
    def export_glb(result: dict, glb_path: str) -> str:
        """Export the decoded mesh + gaussians to a textured .glb."""
        os.makedirs(os.path.dirname(os.path.abspath(glb_path)), exist_ok=True)
        with torch.enable_grad():
            glb = postprocessing_utils.to_glb(
                result["gaussian"],
                result["mesh"],
                simplify=0.95,
                texture_size=1024,
                verbose=False,
            )
        glb.export(glb_path)
        return glb_path


# -----------------------------------------------------------------------------
# Model loading helpers
# -----------------------------------------------------------------------------
def _load_dinov2(model_name: str):
    """Load DINOv2 from a local hub checkout (DINOV2_LOCAL_PATH) or torch.hub."""
    local = os.environ.get("DINOV2_LOCAL_PATH", "").strip()
    if local and os.path.isdir(local):
        return torch.hub.load(local, model_name, source="local", pretrained=True)
    return torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)


def _build_denoiser(config_path: str, ckpt_path: str, device: str):
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    spec = cfg["models"]["denoiser"]
    model = getattr(models, spec["name"])(**spec["args"]).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PartFlow 3D editing inference.")
    p.add_argument("--input", required=True,
                   help="A single case directory, or a parent dir of cases.")
    p.add_argument("--output_dir", default="outputs",
                   help="Where to write <edit_id>/edit.glb.")
    p.add_argument("--manifest", default=None,
                   help="Optional JSON list of case ids to keep.")
    p.add_argument("--weights_dir", default=os.path.join(_HERE, "weights"),
                   help="Dir with stage1_ss/ and stage2_slat/ (see download_weights.py).")
    p.add_argument("--s1_config", default=os.path.join(_HERE, "configs/stage1_ss.json"))
    p.add_argument("--s2_config", default=os.path.join(_HERE, "configs/stage2_slat.json"))
    p.add_argument("--s1_ckpt", default=None, help="Override Stage1 checkpoint path.")
    p.add_argument("--s2_ckpt", default=None, help="Override Stage2 checkpoint path.")
    p.add_argument("--steps", type=int, default=50, help="Flow-Euler sampling steps.")
    p.add_argument("--cfg_strength", type=float, default=0.0,
                   help="Classifier-free guidance strength (0 = condition only).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip cases whose edit.glb already exists.")
    return p.parse_args()


def _resolve_ckpt(explicit: str | None, weights_dir: str, stage: str) -> str:
    if explicit:
        return explicit
    stage_dir = os.path.join(weights_dir, stage)
    if os.path.isdir(stage_dir):
        pts = sorted(f for f in os.listdir(stage_dir) if f.endswith(".pt"))
        if pts:
            return os.path.join(stage_dir, pts[-1])
    raise FileNotFoundError(
        f"No checkpoint for '{stage}' under {weights_dir}. "
        f"Run `python download_weights.py` first, or pass --s1_ckpt/--s2_ckpt."
    )


def main() -> None:
    args = parse_args()
    s1_ckpt = _resolve_ckpt(args.s1_ckpt, args.weights_dir, "stage1_ss")
    s2_ckpt = _resolve_ckpt(args.s2_ckpt, args.weights_dir, "stage2_slat")
    print(f"[PartFlow] Stage1 ckpt: {s1_ckpt}")
    print(f"[PartFlow] Stage2 ckpt: {s2_ckpt}")

    pipe = PartFlowPipeline(
        s1_config=args.s1_config,
        s2_config=args.s2_config,
        s1_ckpt=s1_ckpt,
        s2_ckpt=s2_ckpt,
        device=args.device,
    )

    dataset = PxformDataset(args.input, manifest=args.manifest)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_identity)
    print(f"[PartFlow] {len(dataset)} case(s) to process.\n")

    summary = []
    for (case,) in loader:
        eid = case["edit_id"]
        out_dir = os.path.join(args.output_dir, eid)
        glb_path = os.path.join(out_dir, "edit.glb")
        if args.skip_existing and os.path.isfile(glb_path):
            print(f"[skip] {eid} (edit.glb exists)")
            continue
        try:
            t0 = time.time()
            result = pipe.edit(
                case, steps=args.steps, cfg_strength=args.cfg_strength, seed=args.seed,
            )
            pipe.export_glb(result, glb_path)
            np.savez_compressed(
                os.path.join(out_dir, "pred_slat.npz"),
                coords=result["slat"].coords.cpu().numpy()[:, 1:].astype(np.uint8),
                feats=result["slat"].feats.cpu().numpy().astype(np.float32),
            )
            dt = time.time() - t0
            print(f"[ok]   {eid}  ->  {glb_path}  ({dt:.1f}s)")
            summary.append({"edit_id": eid, "glb": glb_path, "seconds": round(dt, 1)})
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {eid}: {exc!r}")
            summary.append({"edit_id": eid, "error": repr(exc)})

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    n_ok = sum("glb" in s for s in summary)
    print(f"\n[PartFlow] done: {n_ok}/{len(summary)} succeeded.")


if __name__ == "__main__":
    main()
