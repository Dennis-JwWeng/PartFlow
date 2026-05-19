"""
Adapters for using 3DEditVerse/3DEditFormer-style image editing data with
Ori3DEdit ControlNet trainers.

This reads the filtered raw layout:
  <root>/edit_prompts.json
  <root>/encoded_ouput/{alpaca_encode,flux_encode}/<id>/{ori,edit}/
      ss_latent.npz      # key: ss or mean
      slat_latent.npz    # keys: coords, feats; coords may be [0, x, y, z]
  <root>/condition-image/{alpaca,flux}/<id>/
      ori_image.png
      after_edited_Flux.png or edited_*.png

and returns the batch keys expected by Ori3DEdit trainers:
  Stage 1: x_0, ori_voxel, cond
  Stage 2: x_0, ori_slat, cond
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from .editing_image_latent import _load_image_as_tensor
from .structured_latent import SLatVisMixin
from .sparse_structure_latent import SparseStructureLatentVisMixin


_DEFAULT_IMAGE_SIZE = 518
_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp')


def _load_ss(path: str) -> torch.Tensor:
    data = np.load(path)
    if 'mean' in data.files:
        arr = data['mean']
    elif 'ss' in data.files:
        arr = data['ss']
    else:
        raise KeyError(f"{path}: expected key 'mean' or 'ss', got {data.files}")
    return torch.tensor(arr).float()


def _load_slat(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path)
    if 'coords' not in data.files or 'feats' not in data.files:
        raise KeyError(f"{path}: expected keys 'coords' and 'feats', got {data.files}")
    coords = data['coords']
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"{path}: coords must be [N, 3] or [N, 4], got {coords.shape}")
    if coords.shape[1] == 4:
        coords = coords[:, 1:]
    feats = data['feats']
    return torch.tensor(coords).int(), torch.tensor(feats).float()


def _align_feats_to_coords(
    source_coords: torch.Tensor,
    source_feats: torch.Tensor,
    target_coords: torch.Tensor,
) -> torch.Tensor:
    if source_coords.shape == target_coords.shape and torch.equal(source_coords, target_coords):
        return source_feats

    aligned_feats = source_feats.new_zeros((target_coords.shape[0], *source_feats.shape[1:]))
    if source_coords.numel() == 0 or target_coords.numel() == 0:
        return aligned_feats

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
    return aligned_feats


def _find_image(cond_dir: str, kind: str) -> Optional[str]:
    if not os.path.isdir(cond_dir):
        return None

    if kind == 'ori':
        preferred = ('ori_image.png', 'original.png', 'ori_test.png')
        prefixes = ('ori_',)
    elif kind == 'edit':
        preferred = ('edit_image.png', 'after_edited_Flux.png')
        prefixes = ('edited_',)
    else:
        raise ValueError(f'unknown image kind: {kind}')

    for name in preferred:
        path = os.path.join(cond_dir, name)
        if os.path.isfile(path):
            return path

    names = sorted(os.listdir(cond_dir))
    for name in names:
        lower = name.lower()
        if lower.endswith(_IMAGE_EXTS) and any(lower.startswith(p) for p in prefixes):
            return os.path.join(cond_dir, name)

    if kind == 'edit':
        for name in names:
            if name.lower().endswith(_IMAGE_EXTS):
                return os.path.join(cond_dir, name)
    return None




def _iter_instances_from_dataset_manifest(
    root: str,
    *,
    manifest_file: str = 'dataset_info.json',
    split: str = 'both',
    max_items_per_split: int = 0,
) -> List[Dict[str, str]]:
    """Fast path: trust paths recorded in dataset_info.json (no per-sample stat storm)."""
    manifest_path = os.path.join(root, manifest_file)
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    specs = []
    if split in ('alpaca', 'both'):
        specs.append(('alpaca',))
    if split in ('flux_edit', 'both'):
        specs.append(('flux_edit',))

    instances: List[Dict[str, str]] = []
    for (prompt_key,) in specs:
        branch = manifest.get(prompt_key)
        if not isinstance(branch, dict):
            continue
        loaded = 0
        for sid, rec in branch.items():
            if max_items_per_split and loaded >= max_items_per_split:
                break
            if not isinstance(rec, dict):
                continue
            try:
                paths = {
                    'ori_ss': os.path.join(root, rec['ori_ss_latents_path']),
                    'edit_ss': os.path.join(root, rec['edit_ss_latents_path']),
                    'ori_slat': os.path.join(root, rec['ori_latents_path']),
                    'edit_slat': os.path.join(root, rec['edit_latents_path']),
                    'ori_img': os.path.join(root, rec['ori_img_path']),
                    'edit_img': os.path.join(root, rec['edit_img_path']),
                }
            except KeyError:
                continue
            paths['id'] = str(sid)
            paths['source'] = prompt_key
            instances.append(paths)
            loaded += 1
    return instances


def _iter_instances(
    root: str,
    *,
    prompts_file: str = 'edit_prompts.json',
    split: str = 'both',
    max_items_per_split: int = 0,
) -> List[Dict[str, str]]:
    prompts_path = os.path.join(root, prompts_file)

    specs = []
    if split in ('alpaca', 'both'):
        specs.append(('alpaca', 'alpaca_encode', 'alpaca'))
    if split in ('flux_edit', 'both'):
        specs.append(('flux_edit', 'flux_encode', 'flux'))

    prompts = None
    if prompts_file and os.path.isfile(prompts_path):
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)

    instances = []
    for prompt_key, encoded_subdir, cond_subdir in specs:
        if prompts is not None:
            items = prompts.get(prompt_key, {})
            if not isinstance(items, dict):
                continue
            sample_ids = [str(sid) for sid in items.keys()]
        else:
            encoded_root = os.path.join(root, 'encoded_ouput', encoded_subdir)
            if not os.path.isdir(encoded_root):
                continue
            sample_ids = sorted(
                name for name in os.listdir(encoded_root)
                if os.path.isdir(os.path.join(encoded_root, name))
            )

        loaded = 0
        for sid in sample_ids:
            if max_items_per_split and loaded >= max_items_per_split:
                break
            sample_root = os.path.join(root, 'encoded_ouput', encoded_subdir, sid)
            ori_dir = os.path.join(sample_root, 'ori')
            edit_dir = os.path.join(sample_root, 'edit')
            cond_dir = os.path.join(root, 'condition-image', cond_subdir, sid)

            paths = {
                'ori_ss': os.path.join(ori_dir, 'ss_latent.npz'),
                'edit_ss': os.path.join(edit_dir, 'ss_latent.npz'),
                'ori_slat': os.path.join(ori_dir, 'slat_latent.npz'),
                'edit_slat': os.path.join(edit_dir, 'slat_latent.npz'),
                'ori_img': _find_image(cond_dir, 'ori'),
                'edit_img': _find_image(cond_dir, 'edit'),
            }
            if not all(paths[k] and os.path.isfile(paths[k]) for k in paths):
                continue
            paths['id'] = sid
            paths['source'] = prompt_key
            instances.append(paths)
            loaded += 1
    return instances


class Editing3DEditFormerImageSparseStructureLatent(SparseStructureLatentVisMixin, Dataset):
    """Stage 1 adapter: raw 3DEditVerse -> Ori3DEdit SS ControlNet batch."""

    def __init__(
        self,
        roots: str,
        *,
        prompts_file: str = 'edit_prompts.json',
        split: str = 'both',
        max_items_per_split: int = 0,
        image_size: int = _DEFAULT_IMAGE_SIZE,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        dataset_manifest: Optional[str] = None,
        **kwargs,
    ):
        self.image_size = image_size
        self.normalization = normalization
        self.value_range = (0, 1)
        super().__init__(pretrained_ss_dec=pretrained_ss_dec, ss_dec_path=ss_dec_path, ss_dec_ckpt=ss_dec_ckpt)
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
        if dataset_manifest:
            self.instances = _iter_instances_from_dataset_manifest(
                roots, manifest_file=dataset_manifest, split=split, max_items_per_split=max_items_per_split
            )
        else:
            self.instances = _iter_instances(
                roots, prompts_file=prompts_file, split=split, max_items_per_split=max_items_per_split
            )
        print(f'{self.__class__.__name__}: {len(self.instances)} instances loaded from {roots}')

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        info = self.instances[index]
        ori = _load_ss(info['ori_ss'])
        edit = _load_ss(info['edit_ss'])
        if self.normalization is not None:
            ori = (ori - self.mean) / self.std
            edit = (edit - self.mean) / self.std
        return {
            'x_0': edit,
            'ori_voxel': ori,
            'cond': _load_image_as_tensor(info['edit_img'], self.image_size),
        }

    def __str__(self):
        return f'{self.__class__.__name__} ({len(self.instances)} instances)'


class Editing3DEditFormerSLatImage(SLatVisMixin, Dataset):
    """Stage 2 adapter: raw 3DEditVerse -> Ori3DEdit SLat ControlNet batch."""

    def __init__(
        self,
        roots: str,
        *,
        prompts_file: str = 'edit_prompts.json',
        split: str = 'both',
        max_items_per_split: int = 0,
        max_num_voxels: int = 32768,
        image_size: int = _DEFAULT_IMAGE_SIZE,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        dataset_manifest: Optional[str] = None,
        **kwargs,
    ):
        self.image_size = image_size
        self.max_num_voxels = max_num_voxels
        self.normalization = normalization
        self.value_range = (0, 1)
        super().__init__(pretrained_slat_dec=pretrained_slat_dec, slat_dec_path=slat_dec_path, slat_dec_ckpt=slat_dec_ckpt)
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
        if dataset_manifest:
            self.instances = _iter_instances_from_dataset_manifest(
                roots, manifest_file=dataset_manifest, split=split, max_items_per_split=max_items_per_split
            )
        else:
            self.instances = _iter_instances(
                roots, prompts_file=prompts_file, split=split, max_items_per_split=max_items_per_split
            )
        self.loads = [1000] * len(self.instances)
        print(f'{self.__class__.__name__}: {len(self.instances)} instances loaded from {roots}')

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        info = self.instances[index]
        ori_coords, ori_feats = _load_slat(info['ori_slat'])
        edit_coords, edit_feats = _load_slat(info['edit_slat'])
        if self.normalization is not None:
            ori_feats = (ori_feats - self.mean) / self.std
            edit_feats = (edit_feats - self.mean) / self.std
        return {
            'coords': edit_coords,
            'feats': edit_feats,
            'ori_coords': ori_coords,
            'ori_feats': ori_feats,
            'cond': _load_image_as_tensor(info['edit_img'], self.image_size),
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            coords_list, feats_list, ori_coords_list, ori_feats_list = [], [], [], []
            edit_layout, ori_layout = [], []
            edit_start = ori_start = 0
            for i, b in enumerate(sub_batch):
                n = b['coords'].shape[0]
                coords_list.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), b['coords']], dim=-1))
                feats_list.append(b['feats'])
                edit_layout.append(slice(edit_start, edit_start + n))
                edit_start += n

                m = b['ori_coords'].shape[0]
                ori_coords_list.append(torch.cat([torch.full((m, 1), i, dtype=torch.int32), b['ori_coords']], dim=-1))
                ori_feats_list.append(b['ori_feats'])
                ori_layout.append(slice(ori_start, ori_start + m))
                ori_start += m

            x_0 = SparseTensor(coords=torch.cat(coords_list), feats=torch.cat(feats_list))
            x_0._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            x_0.register_spatial_cache('layout', edit_layout)

            ori_slat = SparseTensor(coords=torch.cat(ori_coords_list), feats=torch.cat(ori_feats_list))
            ori_slat._shape = torch.Size([len(group), *sub_batch[0]['ori_feats'].shape[1:]])
            ori_slat.register_spatial_cache('layout', ori_layout)

            packs.append({
                'x_0': x_0,
                'ori_slat': ori_slat,
                'cond': torch.stack([b['cond'] for b in sub_batch], dim=0),
            })
        if split_size is None:
            return packs[0]
        return packs

    def __str__(self):
        return f'{self.__class__.__name__} ({len(self.instances)} instances)'

class Editing3DEditFormerAlignedSLatImage(Editing3DEditFormerSLatImage):
    """Stage 2 adapter for ControlNet: align original SLat features to edited coords."""

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            coords_list, feats_list, ori_coords_list, ori_feats_list = [], [], [], []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                n = b['coords'].shape[0]
                coords = torch.cat([torch.full((n, 1), i, dtype=torch.int32), b['coords']], dim=-1)
                coords_list.append(coords)
                feats_list.append(b['feats'])
                layout.append(slice(start, start + n))
                start += n

                aligned_ori_feats = _align_feats_to_coords(b['ori_coords'], b['ori_feats'], b['coords'])
                ori_coords_list.append(coords.clone())
                ori_feats_list.append(aligned_ori_feats)

            x_0 = SparseTensor(coords=torch.cat(coords_list), feats=torch.cat(feats_list))
            x_0._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            x_0.register_spatial_cache('layout', layout)

            ori_slat = SparseTensor(coords=torch.cat(ori_coords_list), feats=torch.cat(ori_feats_list))
            ori_slat._shape = torch.Size([len(group), *sub_batch[0]['ori_feats'].shape[1:]])
            ori_slat.register_spatial_cache('layout', layout)

            packs.append({
                'x_0': x_0,
                'ori_slat': ori_slat,
                'cond': torch.stack([b['cond'] for b in sub_batch], dim=0),
            })
        if split_size is None:
            return packs[0]
        return packs

