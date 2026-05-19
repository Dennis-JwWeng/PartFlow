"""
Single-view target render loss for Stage2 SLAT Flow training.

Pipeline (differentiable):
    x0_pred = x_t - t * v_pred       # SparseTensor with same coords as x_t
    rep     = slat_dec(x0_pred)      # List[Gaussian] (decoder frozen, NOT no_grad)
    img_hat = renderer.render(rep, extr, intr)['color']
    L_render = w_mse * MSE + w_dreamsim * DreamSim

DreamSim:
    Pretrained perceptual similarity model (Fu et al. 2023). Frozen at
    requires_grad=False; gradients still flow through the input image, so
    used as a perceptual term it shapes textures the way LPIPS does.
    Requires `pip install dreamsim`. The first call also downloads
    pretrained weights to ~/.cache/dreamsim/.

The decoder/renderer parameters are frozen (requires_grad=False) but executed
WITHOUT torch.no_grad so gradients flow x0_pred -> v_pred -> denoiser params.
Target image and camera pose do NOT receive gradients.

Camera-pose loading is intentionally minimal: `load_render_camera` reads from
`case_meta.json` or an explicit pose file; the precise field names are TODO and
will be filled in once the user supplies them. Image loading is finalised.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class RenderLossConfig:
    enabled: bool = False
    max_weight: float = 0.01
    start_ratio: float = 0.3
    end_ratio: float = 0.6
    t_max: float = 0.5
    every_n_steps: int = 4
    mse: float = 1.0
    dreamsim: float = 1.0
    target_image_key: str = 'after.png'
    pose_path: Optional[str] = None
    pose_key: Optional[str] = None
    resolution: int = 256
    first_only: bool = True   # debug: only compute on batch[0]
    # Only compute render loss for these edit types. None = all edit types.
    # Default skips addition/deletion because their after.png comes from a
    # Blender re-render pipeline (cross-pipeline domain shift, ~0.03 L1 floor).
    allowed_edit_types: Optional[List[str]] = field(
        default_factory=lambda: ['material', 'modification', 'scale']
    )

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> 'RenderLossConfig':
        d = dict(d or {})
        # accept both short and full names
        alias = {
            'enable_train_render_loss': 'enabled',
            'render_loss_max_weight': 'max_weight',
            'render_loss_start_ratio': 'start_ratio',
            'render_loss_end_ratio': 'end_ratio',
            'render_loss_t_max': 't_max',
            'render_loss_every_n_steps': 'every_n_steps',
            'render_loss_mse': 'mse',
            'render_loss_dreamsim': 'dreamsim',
            'render_target_image_key': 'target_image_key',
            'render_pose_path': 'pose_path',
            'render_pose_key': 'pose_key',
            'render_resolution': 'resolution',
            'render_first_only': 'first_only',
            'render_allowed_edit_types': 'allowed_edit_types',
        }
        clean = {}
        for k, v in d.items():
            if v is None:
                continue
            clean[alias.get(k, k)] = v
        return RenderLossConfig(**clean)


# --------------------------------------------------------------------------- #
# Lambda schedule                                                             #
# --------------------------------------------------------------------------- #

def schedule_lambda_render(step: int, max_steps: int, cfg: RenderLossConfig) -> float:
    """Linear ramp between start_ratio and end_ratio of training."""
    if not cfg.enabled or cfg.max_weight <= 0:
        return 0.0
    if max_steps <= 0:
        return cfg.max_weight
    s = step / float(max_steps)
    if s < cfg.start_ratio:
        return 0.0
    if s >= cfg.end_ratio:
        return cfg.max_weight
    span = max(cfg.end_ratio - cfg.start_ratio, 1e-8)
    return cfg.max_weight * (s - cfg.start_ratio) / span


# --------------------------------------------------------------------------- #
# Image / pose loaders                                                        #
# --------------------------------------------------------------------------- #

def load_target_image(case_dir: str, cfg: RenderLossConfig, render_resolution: int) -> torch.Tensor:
    """Load `case_dir/<target_image_key>` as float [3,H,W] in [0,1] resized to
    render_resolution. Strips alpha by pre-multiplying onto a black background
    (matches the renderer's default bg_color=(0,0,0))."""
    from PIL import Image
    p = os.path.join(case_dir, cfg.target_image_key)
    if not os.path.isfile(p):
        # tolerant fallback used elsewhere in the repo
        from ...datasets.editing_image_latent import _find_image_path
        alt = _find_image_path(case_dir, 'edit')
        if alt is None:
            raise FileNotFoundError(
                f'[render_loss] target image not found: tried {p!r} and edit_* fallbacks in {case_dir!r}'
            )
        p = alt
    img = Image.open(p)
    if img.mode == 'RGBA':
        rgba = np.array(img).astype(np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:4]   # premultiply onto black bg
    else:
        rgb = np.array(img.convert('RGB')).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    if t.shape[-1] != render_resolution or t.shape[-2] != render_resolution:
        t = F.interpolate(
            t.unsqueeze(0), size=(render_resolution, render_resolution),
            mode='bilinear', align_corners=False, antialias=True,
        ).squeeze(0)
    return t.clamp(0.0, 1.0)


def load_render_camera(case_dir: str, cfg: RenderLossConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load (extrinsics[4,4] in OpenCV w2c, intrinsics[3,3] normalised) for the
    single target view.

    H3D dataset convention (default):
        case_dir/view.meta.json -> {
            "frame": {
                "transform_matrix": 4x4 c2w in NeRF/Blender convention
                                    (+X right, +Y up, -Z forward),
                "camera_angle_x":   horizontal FoV in radians,
            }
        }
    Conversion:
        c2w_opencv = c2w_blender @ diag(1, -1, -1, 1)
        w2c_opencv = inv(c2w_opencv)
        fx = fy = 0.5 / tan(camera_angle_x / 2)   # normalised pinhole
        cx = cy = 0.5

    cfg.pose_path overrides the default file (relative or absolute). cfg.pose_key
    is reserved for future schemas where the camera is nested under another key
    inside view.meta.json.
    """
    if cfg.pose_path:
        path = cfg.pose_path if os.path.isabs(cfg.pose_path) else os.path.join(case_dir, cfg.pose_path)
    else:
        path = os.path.join(case_dir, 'view.meta.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'[render_loss] pose file not found: {path}')

    with open(path, 'r') as fp:
        meta = json.load(fp)

    obj = meta
    if cfg.pose_key:
        obj = meta.get(cfg.pose_key, meta)

    frame = obj.get('frame', obj)
    if 'transform_matrix' not in frame or 'camera_angle_x' not in frame:
        raise ValueError(
            f'[render_loss] {path}: expected frame.transform_matrix and '
            f'frame.camera_angle_x, got keys {list(frame.keys())}'
        )

    c2w_blender = np.asarray(frame['transform_matrix'], dtype=np.float32)
    if c2w_blender.shape != (4, 4):
        raise ValueError(f'[render_loss] transform_matrix must be 4x4, got {c2w_blender.shape}')

    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    c2w_cv = c2w_blender @ flip
    w2c_cv = np.linalg.inv(c2w_cv).astype(np.float32)

    fov_x = float(frame['camera_angle_x'])
    fx = 0.5 / math.tan(fov_x * 0.5)
    K = np.array([
        [fx, 0.0, 0.5],
        [0.0, fx, 0.5],   # square images: fy = fx
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return torch.from_numpy(w2c_cv), torch.from_numpy(K)


def read_edit_type(case_dir: str) -> Optional[str]:
    """Best-effort lookup of the case's edit_type:
        1. <case>/view.meta.json  (key 'edit_type')
        2. <case>/meta.json       (key 'edit_type')
        3. infer from path: .../H3D_v1/<edit_type>/<shard>/<obj_id>/<edit_id>
    Returns None if unknown."""
    for fn in ('view.meta.json', 'meta.json'):
        p = os.path.join(case_dir, fn)
        if os.path.isfile(p):
            try:
                with open(p, 'r') as fp:
                    d = json.load(fp)
                if isinstance(d, dict) and isinstance(d.get('edit_type'), str):
                    return d['edit_type']
            except Exception:
                pass
    parts = os.path.normpath(case_dir).split(os.sep)
    for known in (
        'addition', 'deletion', 'material', 'modification',
        'scale', 'color', 'global',
    ):
        if known in parts:
            return known
    return None


# --------------------------------------------------------------------------- #
# Image-space loss helpers                                                    #
# --------------------------------------------------------------------------- #

# Lazily-instantiated DreamSim module (never put under no_grad; params frozen).
# The first call downloads pretrained weights to ~/.cache/dreamsim/.
_PERCEPTUAL: Dict[str, Optional[nn.Module]] = {'dreamsim': None}


def _get_dreamsim(device) -> nn.Module:
    if _PERCEPTUAL['dreamsim'] is None:
        try:
            from dreamsim import dreamsim as _build  # type: ignore
        except ImportError as e:
            raise ImportError(
                "[render_loss] DreamSim requested (render_loss_dreamsim>0) but "
                "`dreamsim` package not installed. Run: pip install dreamsim"
            ) from e
        # dreamsim 0.2.x caches weights under `./models/` (relative to cwd).
        # Pin the cache to the repo root so weights pre-downloaded into
        #     <repo_root>/models/...
        # are reused regardless of where training is launched from.
        # If the caller has already pointed `DREAMSIM_CACHE_DIR` somewhere,
        # respect that.
        cache_dir = os.environ.get('DREAMSIM_CACHE_DIR')
        if cache_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            # render_loss.py -> trellis/trellis/trainers/flow_matching -> repo root is 4 levels up
            repo_root = os.path.abspath(os.path.join(here, '..', '..', '..', '..'))
            cand = os.path.join(repo_root, 'models')
            if os.path.isdir(cand):
                cache_dir = cand
        # `dreamsim()` returns (model, preprocess); we feed our own [0,1] tensors
        # and bypass the PIL preprocess (model accepts BCHW float in [0,1]).
        if cache_dir is not None:
            model, _preprocess = _build(pretrained=True, device=str(device), cache_dir=cache_dir)
        else:
            model, _preprocess = _build(pretrained=True, device=str(device))
        for p in model.parameters():
            p.requires_grad = False
        _PERCEPTUAL['dreamsim'] = model.eval()
    return _PERCEPTUAL['dreamsim']


# --------------------------------------------------------------------------- #
# Core entry point                                                            #
# --------------------------------------------------------------------------- #

def compute_single_view_train_render_loss(
    *,
    x_t,                         # SparseTensor
    v_pred,                      # SparseTensor (same coords/shape as x_t)
    t: torch.Tensor,             # [B] in [0,1]
    decoder: nn.Module,          # SLat decoder, eval(), requires_grad=False
    renderer,                    # GaussianRenderer (params none / fixed)
    target_images: torch.Tensor, # [B,3,H,W] in [0,1] (no grad)
    extrinsics: torch.Tensor,    # [B,4,4] (no grad)
    intrinsics: torch.Tensor,    # [B,3,3] (no grad)
    cfg: RenderLossConfig,
    decoder_normalization: Optional[Dict[str, torch.Tensor]] = None,
    sample_mask: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    """
    Returns a dict with keys: loss, loss_mse, loss_dreamsim,
    n_used, render_image (optional, detached), elapsed.

    Notes
    -----
    - Skips items with t_i >= cfg.t_max.
    - If cfg.first_only, only the first eligible sample in the batch is used.
    - All decode/render compute is performed in fp32 with autocast disabled.
    """
    start = time.time()
    device = target_images.device

    # 1) Build x0_pred SparseTensor (feats only; coords identical).
    if hasattr(x_t, 'layout') and x_t.layout is not None:
        idx_per_token = _layout_index_per_token(x_t)             # [N_total]
        t_per_token = t.view(-1)[idx_per_token].unsqueeze(-1)    # [N_total,1]
        x0_feats = x_t.feats - t_per_token * v_pred.feats
    else:
        t_b = t.view(-1, *([1] * (x_t.feats.dim() - 1)))
        x0_feats = x_t.feats - t_b * v_pred.feats
    x0_pred = x_t.replace(x0_feats)

    # 2) Pick eligible items in the batch (t < t_max && per-sample mask).
    layout = x_t.layout if hasattr(x_t, 'layout') else None
    B = target_images.shape[0]
    eligible = (t.view(-1) < cfg.t_max).nonzero(as_tuple=False).flatten().tolist()
    if sample_mask is not None:
        eligible = [i for i in eligible if i < len(sample_mask) and bool(sample_mask[i])]
    if cfg.first_only and len(eligible) > 0:
        eligible = eligible[:1]
    if len(eligible) == 0:
        zero = target_images.new_zeros(())
        return {
            'loss': zero, 'loss_mse': zero, 'loss_dreamsim': zero,
            'n_used': 0, 'render_image': None, 'elapsed': time.time() - start,
        }

    # 3) Decode (frozen, fp32, autocast disabled, NOT no_grad).
    with torch.amp.autocast('cuda', enabled=False):
        # Apply normalisation reverse if provided (matches dataset's decode_latent)
        if decoder_normalization is not None:
            x0_for_dec = x0_pred.replace(
                x0_pred.feats.float() * decoder_normalization['std'].to(device)
                + decoder_normalization['mean'].to(device)
            )
        else:
            x0_for_dec = x0_pred.replace(x0_pred.feats.float())
        reps = decoder(x0_for_dec)   # List[Gaussian], batch-aligned

    # 4) Render eligible reps and accumulate per-image losses (MSE + DreamSim).
    parts_mse, parts_ds = [], []
    last_render = None
    H = cfg.resolution

    for idx in eligible:
        rep = reps[idx]
        extr = extrinsics[idx].to(device, dtype=torch.float32)
        intr = intrinsics[idx].to(device, dtype=torch.float32)
        # Force the renderer's resolution (set on the fly; this is a struct, not a buffer).
        renderer.rendering_options.resolution = H
        with torch.amp.autocast('cuda', enabled=False):
            out = renderer.render(rep, extr, intr)
        img_hat = out['color']                # [3,H,W]
        if img_hat.shape[-1] != H or img_hat.shape[-2] != H:
            img_hat = F.interpolate(img_hat.unsqueeze(0), size=(H, H),
                                    mode='bilinear', align_corners=False).squeeze(0)
        img_hat = img_hat.clamp(0.0, 1.0)
        last_render = img_hat.detach()

        target = target_images[idx]
        if target.shape[-1] != H or target.shape[-2] != H:
            target = F.interpolate(target.unsqueeze(0), size=(H, H),
                                   mode='bilinear', align_corners=False).squeeze(0)

        if cfg.mse > 0:
            parts_mse.append(F.mse_loss(img_hat, target))
        if cfg.dreamsim > 0:
            net = _get_dreamsim(device)
            parts_ds.append(
                net(img_hat.unsqueeze(0), target.unsqueeze(0)).mean()
            )

    def _avg(xs):
        if not xs:
            return target_images.new_zeros(())
        return torch.stack(xs).mean()

    mse_avg = _avg(parts_mse)
    ds_avg = _avg(parts_ds)

    total = cfg.mse * mse_avg + cfg.dreamsim * ds_avg

    return {
        'loss': total,
        'loss_mse': mse_avg.detach(),
        'loss_dreamsim': ds_avg.detach(),
        'n_used': len(eligible),
        'render_image': last_render,
        'elapsed': time.time() - start,
    }


def _layout_index_per_token(x) -> torch.Tensor:
    """Return a [N_tokens] long tensor mapping each row of x.feats to its batch index,
    derived from x.layout (a list of slices)."""
    layout = x.layout
    n = x.feats.shape[0]
    out = torch.zeros(n, dtype=torch.long, device=x.feats.device)
    for i, sl in enumerate(layout):
        out[sl] = i
    return out
