"""Shared image conditioning and sparse-coordinate mapping for PartFlow."""


import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _match_prefixed_image(instance_dir: str, prefixes: Tuple[str, ...]) -> Optional[str]:
    """Return the first image file whose basename starts with one of the prefixes."""
    try:
        filenames = sorted(os.listdir(instance_dir))
    except FileNotFoundError:
        return None

    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    for fn in filenames:
        lower = fn.lower()
        if not lower.endswith(valid_exts):
            continue
        if any(fn.startswith(prefix) for prefix in prefixes):
            return os.path.join(instance_dir, fn)
    return None


def _find_image_path(instance_dir: str, name: str) -> Optional[str]:
    """Find an image inside the instance dir, tolerant to common naming variants."""
    if name == 'edit':
        candidates = (
            'edit_image.png',
            'edit_img.png',
            'after_edited_Flux.png',
        )
        prefix_fallbacks = ('edited_',)
    elif name == 'ori':
        candidates = (
            'ori_image.png',
            'ori.png',
            'original.png',
        )
        prefix_fallbacks = ('ori_',)
    else:
        raise ValueError(f"Unknown image kind: {name}")

    for cand in candidates:
        p = os.path.join(instance_dir, cand)
        if os.path.isfile(p):
            return p

    return _match_prefixed_image(instance_dir, prefix_fallbacks)


def _pil_to_tensor(image: Image.Image, image_size: int = 518) -> torch.Tensor:
    """Resize RGB images and crop/premultiply RGBA images for conditioning."""
    if image.mode == "RGBA":
        alpha = np.array(image.getchannel(3))
        nz = alpha.nonzero()
        if nz[0].size > 0:
            bbox = [nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_hsize = hsize * 1.2
            image = image.crop([
                int(cx - aug_hsize), int(cy - aug_hsize),
                int(cx + aug_hsize), int(cy + aug_hsize),
            ])
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        rgb = image.convert("RGB")
        rgb_t = torch.tensor(np.array(rgb)).permute(2, 0, 1).float() / 255.0
        alpha_t = torch.tensor(np.array(alpha)).float() / 255.0
        return rgb_t * alpha_t.unsqueeze(0)

    rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    return torch.tensor(np.array(rgb)).permute(2, 0, 1).float() / 255.0


def _load_image_as_tensor(path: str, image_size: int = 518) -> torch.Tensor:
    """Read a condition image as a float [3,H,W] tensor in [0,1]."""
    with Image.open(path) as image:
        return _pil_to_tensor(image, image_size)


def map_ori_to_edit_coords(ori_coords, ori_feats, edit_coords, return_occ=False):
    """
    Map original SLat features onto the edited coordinate system.
    For positions present in both, copy original features.
    For new positions (only in edited), fill with zeros.

    Args:
        ori_coords: [N_ori, 3] int
        ori_feats:  [N_ori, C] float
        edit_coords: [N_edit, 3] int
        return_occ: return a [N_edit, 1] overlap flag when True.
    Returns:
        mapped_feats: [N_edit, C] float on edit_coords
    """
    C = ori_feats.shape[1]
    N_edit = edit_coords.shape[0]
    mapped = np.zeros((N_edit, C), dtype=np.float32)
    occ = np.zeros((N_edit, 1), dtype=np.float32)

    ori_set = {}
    for i in range(ori_coords.shape[0]):
        key = (ori_coords[i, 0], ori_coords[i, 1], ori_coords[i, 2])
        ori_set[key] = i

    for j in range(N_edit):
        key = (edit_coords[j, 0], edit_coords[j, 1], edit_coords[j, 2])
        if key in ori_set:
            mapped[j] = ori_feats[ori_set[key]]
            occ[j] = 1

    return (mapped, occ) if return_occ else mapped
