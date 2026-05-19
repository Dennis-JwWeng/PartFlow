"""Pxform dataset for PartFlow 3D editing inference.

Each *case* is a directory holding the pre-encoded inputs of one edit, in the
Pxform benchmark layout:

    <case_dir>/
        ori_ss_latents.npz   # key `mean` (or `ss`): float32 [8, 16, 16, 16]
        ori_latents.npz      # keys `coords` [N,3 or 4] int, `feats` [N,8] float32
        edit_img.png         # the target / edit image (RGB or RGBA)
        case_meta.json       # optional, free-form metadata (e.g. prompt)

`edit_ss_latents.npz` / `edit_latents.npz` (ground truth) are NOT required for
inference and are ignored if present.

`PxformDataset` accepts either a single case directory or a parent directory
holding many case sub-directories, so it plugs straight into a
`torch.utils.data.DataLoader`. Use `batch_size=1` with `collate_identity`,
since each case carries a variable-length sparse SLat tensor.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

# Files that must exist for a directory to count as a valid inference case.
REQUIRED_FILES = ("ori_ss_latents.npz", "ori_latents.npz")
IMAGE_CANDIDATES = ("edit_img.png", "edit_image.png", "edit.png")


def _load_ss_latent(path: str) -> np.ndarray:
    """Load a sparse-structure latent, accepting either the `mean` or `ss` key."""
    z = np.load(path)
    for key in ("mean", "ss"):
        if key in z.files:
            return z[key].astype(np.float32)
    raise KeyError(f"{path}: expected key 'mean' or 'ss', got {list(z.files)}")


def _slat_coords_to_xyz(coords: np.ndarray) -> np.ndarray:
    """Normalise SLat coords to (N, 3); drop a leading batch column if present."""
    if coords.ndim != 2:
        raise ValueError(f"SLat coords must be 2D, got shape {coords.shape}")
    if coords.shape[1] == 3:
        return np.ascontiguousarray(coords)
    if coords.shape[1] == 4:
        return np.ascontiguousarray(coords[:, 1:])
    raise ValueError(f"SLat coords expected 3 or 4 columns, got {coords.shape}")


def _find_edit_image(case_dir: str) -> Optional[str]:
    for name in IMAGE_CANDIDATES:
        p = os.path.join(case_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _is_case_dir(path: str) -> bool:
    return os.path.isdir(path) and all(
        os.path.isfile(os.path.join(path, f)) for f in REQUIRED_FILES
    )


def _load_manifest_ids(path: str) -> List[str]:
    obj = json.loads(open(path, "r", encoding="utf-8").read())
    ids = obj.get("edit_ids") or obj.get("ids") if isinstance(obj, dict) else obj
    if not ids:
        raise ValueError(f"No edit ids found in manifest {path}")
    return [str(x).strip() for x in ids if str(x).strip()]


class PxformDataset(Dataset):
    """Yields one pre-encoded Pxform edit case per item.

    Parameters
    ----------
    root:
        A single case directory, or a parent directory containing many case
        sub-directories.
    manifest:
        Optional path to a JSON file listing case ids (folder names) to keep.
        Accepts a bare list, or a dict with an `edit_ids` / `ids` key.
    require_image:
        If True (default), cases without an edit image are dropped.
    """

    def __init__(
        self,
        root: str,
        manifest: Optional[str] = None,
        require_image: bool = True,
    ) -> None:
        self.root = os.path.abspath(root)
        self.require_image = require_image

        if _is_case_dir(self.root):
            case_dirs = [self.root]
        elif os.path.isdir(self.root):
            case_dirs = [
                os.path.join(self.root, n)
                for n in sorted(os.listdir(self.root))
                if _is_case_dir(os.path.join(self.root, n))
            ]
        else:
            raise FileNotFoundError(f"root does not exist: {self.root}")

        if manifest:
            keep = set(_load_manifest_ids(manifest))
            case_dirs = [d for d in case_dirs if os.path.basename(d) in keep]

        if require_image:
            case_dirs = [d for d in case_dirs if _find_edit_image(d) is not None]

        if not case_dirs:
            raise RuntimeError(f"No valid Pxform cases found under {self.root}")
        self.case_dirs: List[str] = case_dirs

    def __len__(self) -> int:
        return len(self.case_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_dir = self.case_dirs[idx]
        ori_slat = np.load(os.path.join(case_dir, "ori_latents.npz"))
        meta_path = os.path.join(case_dir, "case_meta.json")
        meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        return {
            "edit_id": os.path.basename(case_dir),
            "case_dir": case_dir,
            "edit_image_path": _find_edit_image(case_dir),
            "ori_ss": _load_ss_latent(os.path.join(case_dir, "ori_ss_latents.npz")),
            "ori_slat_coords": _slat_coords_to_xyz(ori_slat["coords"]),
            "ori_slat_feats": ori_slat["feats"].astype(np.float32),
            "meta": meta,
        }


def collate_identity(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collate fn for `DataLoader`: each case has a variable-length sparse
    tensor, so batches are kept as plain lists (use `batch_size=1`)."""
    return batch
