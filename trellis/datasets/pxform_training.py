"""Paired Pxform training: raw SS, TRELLIS-normalized SLAT, target image.

Mask sources are explicit: ``sidecar`` requires real part masks; ``latent_delta``
estimates preservation from the paired latents and is NOT a semantic part mask.
The latter keeps only overlapping occupied tokens outside dilated changes.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from typing import Any, Dict, Tuple, Union
from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from PIL import Image
from ..utils.editing_utils import _pil_to_tensor



def _slat_arrays(npz_dict: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = npz_dict["slat_coords"]
    if coords.ndim == 2 and coords.shape[1] == 4:
        coords = coords[:, 1:]
    feats = npz_dict["slat_feats"]
    return torch.tensor(coords).int(), torch.tensor(feats).float()

def _align_feats_to_coords(
    source_coords: torch.Tensor,
    source_feats: torch.Tensor,
    target_coords: torch.Tensor,
    return_occ: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Project source sparse features onto target coords.

    Missing source coords are filled with zeros on the target layout. When
    requested, also return a [N, 1] occupancy flag showing which target coords
    existed in the source. This helper is called per sample before batch indices
    are prepended, so coordinates never match across batch items.
    """
    if source_coords.shape == target_coords.shape and torch.equal(source_coords, target_coords):
        occ = source_feats.new_ones((target_coords.shape[0], 1))
        return (source_feats, occ) if return_occ else source_feats

    aligned_feats = source_feats.new_zeros((target_coords.shape[0], *source_feats.shape[1:]))
    occ = source_feats.new_zeros((target_coords.shape[0], 1))
    if source_coords.numel() == 0 or target_coords.numel() == 0:
        return (aligned_feats, occ) if return_occ else aligned_feats

    max_coord = torch.cat([source_coords.reshape(-1), target_coords.reshape(-1)]).max().item()
    base = int(max_coord) + 1

    def coord_keys(coords: torch.Tensor) -> torch.Tensor:
        coords = coords.to(torch.long)
        return coords[:, 0] * base * base + coords[:, 1] * base + coords[:, 2]

    source_keys = coord_keys(source_coords)
    target_keys = coord_keys(target_coords)
    order = torch.argsort(source_keys)
    sorted_keys = source_keys[order]
    pos = torch.searchsorted(sorted_keys, target_keys)

    valid_pos = pos < sorted_keys.numel()
    valid = valid_pos.clone()
    if valid_pos.any():
        valid[valid_pos] = sorted_keys[pos[valid_pos]] == target_keys[valid_pos]

    if valid.any():
        aligned_feats[valid] = source_feats[order[pos[valid]]]
        occ[valid] = 1
    return (aligned_feats, occ) if return_occ else aligned_feats

SLAT_NORMALIZATION = {
    "mean": [-2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
             -0.08418072760105133, -0.5271206498146057, 0.7238689064979553,
             -1.1414450407028198, 1.2039363384246826],
    "std": [2.377650737762451, 2.386378288269043, 2.124418020248413,
            2.1748552322387695, 2.663944721221924, 2.371192216873169,
            2.6217446327209473, 2.684523105621338],
}


def latent_delta_masks(before, after, threshold=0.15, dilation=1):
    """Infer conservative keep masks in normalized SLAT space, on CPU."""
    bc, bf = _slat_arrays(before)
    ac, af = _slat_arrays(after)
    mean = torch.tensor(SLAT_NORMALIZATION['mean'])
    std = torch.tensor(SLAT_NORMALIZATION['std'])
    bf, af = (bf - mean) / std, (af - mean) / std
    ba, ao = _align_feats_to_coords(bc, bf, ac, return_occ=True)
    ab, bo = _align_feats_to_coords(ac, af, bc, return_occ=True)
    changed_a = (ao[:, 0] == 0) | ((ba - af).square().mean(-1).sqrt() > threshold)
    changed_b = (bo[:, 0] == 0) | ((ab - bf).square().mean(-1).sqrt() > threshold)
    changed = torch.zeros(64, 64, 64)
    for coords, flags in ((ac, changed_a), (bc, changed_b)):
        c = coords[flags].long()
        changed[c[:, 0], c[:, 1], c[:, 2]] = 1
    changed = F.max_pool3d(changed[None, None], 2*dilation+1, 1, dilation)[0, 0]
    occupancy = torch.zeros_like(changed)
    shared = ac[ao[:, 0] > 0].long()
    occupancy[shared[:, 0], shared[:, 1], shared[:, 2]] = 1
    keep64 = occupancy * (1 - changed)
    keep_ss = (F.max_pool3d(occupancy[None, None], 4, 4)[0, 0] > 0) & (
        F.max_pool3d(changed[None, None], 4, 4)[0, 0] == 0)
    c = ac.long()
    return keep_ss.float(), keep64[c[:, 0], c[:, 1], c[:, 2]]


class _PxformPaired(Dataset):
    def __init__(self, roots, *, split='train', edit_types=None, image_size=518,
                 normalization=None, mask_source='sidecar', mask_sidecar_root=None,
                 mask_delta_threshold=0.15, mask_dilation=1, max_samples=None,
                 max_num_voxels=32768, manifest=None, edit_ids_file=None,
                 require_render_camera=True, **kwargs):
        self.root = Path(roots).resolve()
        self.image_size = image_size
        self.normalization = normalization or SLAT_NORMALIZATION
        self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
        self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
        if (self.std <= 0).any():
            raise ValueError('SLAT normalization std must be positive')
        if mask_source not in ('sidecar', 'latent_delta'):
            raise ValueError('mask_source must be sidecar or latent_delta')
        self.mask_source = mask_source
        default_masks = Path(roots).absolute().parent / 'mask_sidecars'
        self.mask_sidecar_root = (Path(mask_sidecar_root).resolve() if mask_sidecar_root
                                  else default_masks if default_masks.is_dir() else None)
        self.threshold, self.dilation = mask_delta_threshold, mask_dilation
        self.max_num_voxels = max_num_voxels
        keep_obj = None
        if split is not None:
            split_path = self.root / 'data' / 'splits' / f'{split}.obj_ids.txt'
            if split_path.is_file():
                keep_obj = set(split_path.read_text().splitlines())
            elif split == 'train' and (self.root / 'release.json').is_file():
                release = json.loads((self.root / 'release.json').read_text())
                if release.get('split') != 'train':
                    raise ValueError('Published release is not a training split')
            else:
                raise FileNotFoundError(f'Missing object split: {split_path}')
        keep_ids = None
        if edit_ids_file:
            obj = json.loads(Path(edit_ids_file).read_text())
            keep_ids = set(obj['edit_ids'] if isinstance(obj, dict) else obj)
        self.records = []
        with open(manifest or self.root / 'manifests' / 'all.jsonl') as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if edit_types is not None and r['edit_type'] not in edit_types:
                    continue
                if keep_obj is not None and r['obj_id'] not in keep_obj:
                    continue
                if keep_ids is not None and r['edit_id'] not in keep_ids:
                    continue
                case = self.root / r['edit_type'] / str(r['shard']) / r['obj_id'] / r['edit_id']
                required = ['before.npz', 'after.npz', 'after.png']
                if require_render_camera:
                    if not ((case / 'view.meta.json').is_file() or (case / 'meta.json').is_file()):
                        continue
                if not all((case / name).is_file() for name in required):
                    continue
                self.records.append((r, case))
                if max_samples is not None and len(self.records) >= max_samples:
                    break
        if not self.records:
            raise RuntimeError(f'No paired training cases in {self.root}, split={split}')
        self.loads = [1] * len(self.records)
        self.value_range = (0, 1)
        print(f'Pxform: {len(self)} cases, split={split}, mask_source={mask_source}; '
              'SS=raw, SLAT=TRELLIS-normalized')

    def __len__(self):
        return len(self.records)

    def _load(self, index):
        r, case = self.records[index]
        with np.load(case / 'before.npz') as z:
            before = dict(z)
        with np.load(case / 'after.npz') as z:
            after = dict(z)
        if self.max_num_voxels and len(after['slat_coords']) > self.max_num_voxels:
            return None
        for z in (before, after):
            if z['ss'].shape != (8, 16, 16, 16):
                raise ValueError(f'{case}: expected SS shape [8,16,16,16]')
            if not all(np.isfinite(z[k]).all() for k in ('ss', 'slat_feats')):
                raise ValueError(f'{case}: nonfinite latent')
            c = z['slat_coords'][:, -3:]
            if np.any(c < 0) or np.any(c >= 64):
                raise ValueError(f'{case}: SLAT coords outside [0,64)')
        if self.mask_source == 'latent_delta':
            ms, ml = latent_delta_masks(before, after, self.threshold, self.dilation)
        elif r['edit_type'] == 'global':
            # There is no unedited part in a global style transformation.
            ms, ml = torch.zeros(16, 16, 16), torch.zeros(len(after['slat_coords']))
        else:
            sidecar = (self.mask_sidecar_root / r['edit_type'] / str(r['shard']) /
                       r['obj_id'] / f"{r['edit_id']}.npz") if self.mask_sidecar_root else case / 'masks.npz'
            if sidecar.is_file():
                with np.load(sidecar) as z:
                    ms, ml = torch.from_numpy(z['mask_keep_ss'].copy()).float(), torch.from_numpy(z['mask_keep_slat'].copy()).float()
            else:
                if 'mask_keep_ss' not in before or 'mask_keep_slat' not in after:
                    raise FileNotFoundError(f'{case}: real masks missing; expected {sidecar}. '
                                            'Use mask_source=latent_delta explicitly for estimated masks.')
                ms, ml = torch.tensor(before['mask_keep_ss']).float(), torch.tensor(after['mask_keep_slat']).float()
        ms, ml = ms.reshape(16, 16, 16), ml.reshape(-1)
        if ml.numel() != len(after['slat_coords']):
            raise ValueError(f'{case}: SLAT mask length mismatch')
        if not torch.isfinite(ms).all() or not torch.isfinite(ml).all() or (ms < 0).any() or (ms > 1).any() or (ml < 0).any() or (ml > 1).any():
            raise ValueError(f'{case}: masks must be finite and in [0,1]')
        # Do not preserve the source's empty space (especially for additions).
        bc, bf = _slat_arrays(before)
        ac, af = _slat_arrays(after)
        occupied = torch.zeros(64, 64, 64)
        c = bc.long()
        occupied[c[:, 0], c[:, 1], c[:, 2]] = 1
        ms = ms * F.max_pool3d(occupied[None, None], 4, 4)[0, 0]
        _, overlap = _align_feats_to_coords(bc, bf, ac, return_occ=True)
        ml = ml * overlap[:, 0]
        with Image.open(case / 'after.png') as image:
            cond = _pil_to_tensor(image, self.image_size)
        return r, case, before, after, ms, ml, cond

    def visualize_sample(self, sample):
        return sample['cond']


class PxformSSDataset(_PxformPaired):
    def __getitem__(self, index):
        sample = self._load(index)
        if sample is None:
            return None
        r, case, before, after, ms, ml, cond = sample
        return dict(x_0=torch.tensor(after['ss']).float(),
                    ori_voxel=torch.tensor(before['ss']).float(), cond=cond,
                    mask_keep_ss=ms, edit_type=r['edit_type'], case_dirs=str(case))

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return {k: [b[k] for b in batch] if k in ('edit_type', 'case_dirs') else
                torch.stack([b[k] for b in batch]) for k in batch[0]}


class PxformSLatDataset(_PxformPaired):
    @staticmethod
    def collate_fn(batch, split_size=None):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b["coords"].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            coords_list, feats_list, ori_coords_list, ori_feats_list, ori_occ_list = [], [], [], [], []
            layout, ori_layout = [], []
            start = ori_start = 0
            for i, b in enumerate(sub_batch):
                n = b["coords"].shape[0]
                coords_list.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), b["coords"]], dim=-1))
                feats_list.append(b["feats"])
                layout.append(slice(start, start + n))
                start += n
                aligned_ori_feats, ori_occ = _align_feats_to_coords(
                    b["ori_coords"], b["ori_feats"], b["coords"], return_occ=True
                )
                m = b["coords"].shape[0]
                ori_coords_list.append(torch.cat([torch.full((m, 1), i, dtype=torch.int32), b["coords"]], dim=-1))
                ori_feats_list.append(aligned_ori_feats)
                ori_occ_list.append(ori_occ)
                ori_layout.append(slice(ori_start, ori_start + m))
                ori_start += m

            x_0 = SparseTensor(coords=torch.cat(coords_list), feats=torch.cat(feats_list))
            x_0._shape = torch.Size([len(group), *sub_batch[0]["feats"].shape[1:]])
            x_0.register_spatial_cache("layout", layout)
            ori_slat = SparseTensor(coords=torch.cat(ori_coords_list), feats=torch.cat(ori_feats_list))
            ori_slat._shape = torch.Size([len(group), *sub_batch[0]["ori_feats"].shape[1:]])
            ori_slat.register_spatial_cache("layout", ori_layout)
            ori_occ = SparseTensor(coords=ori_slat.coords.clone(), feats=torch.cat(ori_occ_list))
            ori_occ._shape = torch.Size([len(group), 1])
            ori_occ.register_spatial_cache("layout", ori_layout)
            pack: Dict[str, Any] = {
                "x_0": x_0,
                "ori_slat": ori_slat,
                "ori_occ": ori_occ,
                "cond": torch.stack([b["cond"] for b in sub_batch]),
                "edit_type": [b.get("edit_type", "") for b in sub_batch],
            }
            if "case_dir" in sub_batch[0]:
                pack["case_dirs"] = [b["case_dir"] for b in sub_batch]
            if all("mask_keep_slat" in b for b in sub_batch):
                pack["mask_keep_slat"] = torch.cat(
                    [b["mask_keep_slat"] for b in sub_batch], dim=0
                )
            packs.append(pack)
        return packs[0] if split_size is None else packs

    def __getitem__(self, index):
        sample = self._load(index)
        if sample is None:
            return None
        r, case, before, after, ms, ml, cond = sample
        bc, bf = _slat_arrays(before)
        ac, af = _slat_arrays(after)
        return dict(coords=ac, feats=(af-self.mean)/self.std,
                    ori_coords=bc, ori_feats=(bf-self.mean)/self.std,
                    cond=cond, mask_keep_slat=ml, edit_type=r['edit_type'], case_dir=str(case))
